"""This recipe to train CLAP.
It supports distillation using tinyCLAP (https://arxiv.org/abs/2311.14517).

Authors
    * Francesco Paissan 2024
"""

import sys

import speechbrain as sb
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats

from clap_datasets import prepare_clap_datasets

torch.backends.cudnn.enabled = False

eps = 1e-10


class CLAPBrain(sb.Brain):
    def preprocess(self, wavs):
        """Pre-process wavs."""
        x = self.hparams.spectrogram_extractor(wavs)
        x = self.hparams.logmel_extractor(x)

        return x

    def prepare_txt_features(self, text):
        """Prepares text features to input in CLAP text encoder."""
        txt_inp = self.hparams.txt_tokenizer(
            text,
            max_length=self.hparams.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        return txt_inp

    def compute_sim(self, audio_embed, caption_embed):
        """Computes CLAP similarity metric."""
        logit_scale = self.hparams.clap.logit_scale.exp()
        similarity = logit_scale * audio_embed @ caption_embed.t()
        similarity = torch.clamp(similarity, max=100)

        return similarity

    def compute_forward(self, batch, stage):
        if len(batch) == 2:
            wavs, caption = batch
        else:
            wavs, caption, _, _ = batch

        wavs = wavs.to(self.device).squeeze(1)

        x_sb = self.preprocess(wavs)

        text_inp = self.prepare_txt_features(caption)

        txt_shared, aud_shared = self.hparams.clap(
            x_sb,
            text_inp.input_ids.data,
            text_inp.token_type_ids.data,
            text_inp.attention_mask.data,
        )

        aud_shared_student = None
        if not hasattr(self.modules, "clap"):
            aud_shared_student, _, _ = self.modules.clap_student(x_sb)
            aud_shared_student = aud_shared_student / aud_shared_student.norm(
                dim=1, keepdim=True
            )

        return txt_shared, aud_shared, aud_shared_student

    def contrastive_loss(self, aud_shared, txt_shared):
        """CLAP loss"""

        def l_i(C, ax):
            return -torch.diag(C.softmax(dim=ax)).log() * 0.5

        C = self.compute_sim(aud_shared, txt_shared)

        return (l_i(C, 0) + l_i(C, 1)).mean()

    def tinyCLAP_loss(self, aud_shared, aud_shared_student):
        """tinyCLAP loss"""
        return -F.cosine_similarity(
            aud_shared.detach(), aud_shared_student, dim=-1
        ).mean()

    def compute_objectives(self, pred, batch, stage):
        """Helper function to compute the objectives"""
        (
            txt_shared,
            aud_shared,
            aud_shared_student,
        ) = pred

        if self.hparams.audioenc_name_student is not None:
            assert aud_shared_student is not None, "Cannot distill. Check your hparams."
            loss = self.tinyCLAP_loss(aud_shared, aud_shared_student)
        else:
            loss = self.contrastive_loss(aud_shared, txt_shared)

        if self.hparams.zs_eval:
            _, _, labels, _ = batch
            labels = labels.to(self.device)
            if labels.dim() == 1:
                labels = labels[..., None]
            class_cap = self.prepare_txt_features(
                ["this is the sound of " + c for c in self.hparams.class_list]
            )

            e_t_class, _ = self.hparams.clap(
                None,
                class_cap.input_ids.data,
                class_cap.token_type_ids.data,
                class_cap.attention_mask.data,
                single="txt",
            )

            class_pred = self.compute_sim(aud_shared, e_t_class).softmax(-1)

            if aud_shared_student is not None:
                class_pred = self.compute_sim(aud_shared_student, e_t_class).softmax(-1)

            uttid = torch.ones(e_t_class.shape[0])
            self.acc_metric.append(
                uttid,
                predict=class_pred,
                target=labels,
            )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams, "lr_annealing"):
                if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                    self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Steps taken before stage start."""

        @torch.no_grad()
        def accuracy_value(predict, target):
            """Computes Accuracy"""
            predict = predict.argmax(1)

            return (predict.unsqueeze(1) == target).float().squeeze()

        self.acc_metric = MetricStats(metric=accuracy_value)

        return super().on_stage_start(stage, epoch)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Plots in subplots the values of `self.batch_to_plot` and saves the
        plot to the experiment folder `self.hparams.output_folder`."""
        tmp = {}
        tmp["acc"] = torch.Tensor(self.acc_metric.scores).mean()

        extra_m = {}
        for m in tmp:
            if not tmp[m].isnan():
                extra_m[m] = tmp[m]

        if stage == sb.Stage.TRAIN:
            old_lr = self.hparams.lr
            if hasattr(self.hparams, "lr_annealing"):
                old_lr, new_lr = self.hparams.lr_annealing(
                    [self.optimizer], epoch, stage_loss
                )
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
            }
            self.train_stats.update(extra_m)

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
            )

            # Save the current checkpoint and delete previous checkpoints
            self.checkpointer.save_and_keep_only(
                meta=self.train_stats, min_keys=["loss"]
            )

        if stage == sb.Stage.VALID:
            # current_fid = torch.Tensor(self.inp_fid.scores).mean()
            old_lr = self.hparams.lr
            if hasattr(self.hparams, "lr_annealing"):
                old_lr, new_lr = self.hparams.lr_annealing(
                    [self.optimizer], epoch, stage_loss
                )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            valid_stats = {
                "loss": stage_loss,
            }
            valid_stats.update(extra_m)

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

        if stage == sb.Stage.TEST:
            # current_fid = torch.Tensor(self.inp_fid.scores).mean()
            test_stats = {
                "loss": stage_loss,
            }
            test_stats.update(extra_m)

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=test_stats
            )

            test_stats = {
                k: (
                    test_stats[k].item()
                    if isinstance(test_stats[k], torch.Tensor)
                    else test_stats[k]
                )
                for k in test_stats
            }
            print(test_stats)


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets = prepare_clap_datasets(hparams)

    hparams["clap"].to(run_opts["device"])
    if hparams["audioenc_name_student"] is not None:
        hparams["clap"].requires_grad_(False)
    hparams["clap"].eval()

    if hparams["zs_eval"]:
        hparams["class_list"] = datasets["train"].dataset.classes

    if hparams["audioenc_name_student"] is not None:
        if hparams["projection_only"]:
            print("Freezing Base AudioEncoder. Updating only the projection layers.")
            hparams["student_model"].base.requires_grad_(False)

    hparams["spectrogram_extractor"].to(run_opts["device"])
    hparams["logmel_extractor"].to(run_opts["device"])

    clap_brain = CLAPBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["pretrained_CLAP"] is not None:
        print("Loading CLAP model...")
        run_on_main(hparams["load_CLAP"].collect_files)
        hparams["load_CLAP"].load_collected()

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    if not hparams["zs_eval"]:
        clap_brain.fit(
            epoch_counter=clap_brain.hparams.epoch_counter,
            train_set=datasets["train"],
        )

    if hparams["zs_eval"]:
        clap_brain.checkpointer.recover_if_possible(
            min_key="loss",
        )

        test_stats = clap_brain.evaluate(
            test_set=datasets["train"],
            min_key="loss",
            progressbar=True,
            test_loader_kwargs={"batch_size": 16},
        )
