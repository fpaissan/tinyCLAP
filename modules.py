"""
Code to define CLAP-related networks.
Some code inspired from here https://github.com/zhepeiw/clap_curation

Credits:
    * Francesco Paissan 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from micromind.networks import PhiNet
from speechbrain.utils.fetching import fetch
from torch import nn
from torchinfo import summary
from transformers import AutoModel, BatchEncoding


def get_model_from_str(s, vs=("alpha", "beta", "t0", "N")):
    def get_var(s, key):
        tmp = s.split("_")
        return tmp[tmp.index(key) + 1]

    verb = "PhiNet initialized with "
    ret = {}
    for k in vs:
        verb += f"{k}={get_var(s, k)} "
        ret[k] = float(get_var(s, k))

    ret["t_zero"] = ret["t0"]
    ret["num_layers"] = ret["N"]
    del ret["t0"]
    del ret["N"]

    return ret


def get_audio_encoder(name: str):
    if name == "Cnn14":
        return Cnn14
    elif "phinet" in name:
        phinet_conf = get_model_from_str(name)
        return PhiNet(input_shape=(1, 640, 64), compatibility=True, **phinet_conf)
    else:
        raise Exception(
            "The audio encoder name {} is incorrect or not supported".format(name)
        )


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class PhiNet(PhiNet):
    def __init__(self, embedding_dim=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bn0 = nn.BatchNorm2d(64)

        if embedding_dim is not None:
            in_channels_next = self._layers[-1]._layers[-2].weight.shape[0]
            self.pn_block = nn.Conv2d(
                in_channels_next,
                embedding_dim,
                kernel_size=1,
                stride=2,
            )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = super().forward(x)
        embedding = x

        x = self.pn_block(x)
        x = x.mean((-1, -2))

        return {"embedding": (x, embedding), "clipwise_output": x}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation="linear", temperature=1.0):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.bn_att = nn.BatchNorm1d(n_out)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class Cnn14(nn.Module):
    def __init__(
        self,
        classes_num,
        out_emb,
    ):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # out_emb is 2048 for best Cnn14
        self.fc1 = nn.Linear(2048, out_emb, bias=True)
        self.fc_audioset = nn.Linear(out_emb, classes_num, bias=True)

    def forward(self, x, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """
        # (batch_size, 1, time_steps, mel_bins)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x4_out = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x4_out, p=0.2, training=self.training)
        x3_out = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x3_out, p=0.2, training=self.training)
        x2_out = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x2_out, p=0.2, training=self.training)
        x1_out = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x1_out, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {
            "clipwise_output": clipwise_output,
            "embedding": (embedding, x1_out, x2_out, x3_out, x4_out),
        }

        return output_dict


class AudioEncoder(nn.Module):
    def __init__(
        self,
        audioenc_name: str,
        d_in: int,
        d_out: int,
        classes_num: int,
    ) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        if not "phinet" in audioenc_name:
            self.base = audio_encoder(
                classes_num,
                d_in,
            )
        else:
            self.base = audio_encoder

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = (
            out_dict["embedding"][0],
            out_dict["clipwise_output"],
        )
        projected_vec = self.projection(audio_features)

        return (
            projected_vec,
            out_dict["embedding"][1:],
            audio_classification_output,
        )


class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(text_model)

        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        out = self.base(**x)[0]
        hidden_state = out
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        self.hidden_state = hidden_state.detach()
        return projected_vec


class CLAP(nn.Module):
    def __init__(
        self,
        # audio
        audioenc_name: str,
        classes_num: int,
        out_emb: int,
        # text
        text_model: str,
        transformer_embed_dim: int,
        # common
        d_proj: int,
        pretrained_weights: bool = True,
        CLAP_weights: str = None,
        # audio student
        audioenc_name_student=None,
        out_emb_student=None,
    ):
        super().__init__()
        ckpt_path = None
        if pretrained_weights and CLAP_weights is not None:
            weights_path = "CLAP_weights.pth"
            tmp = CLAP_weights.split("/")
            print(
                " ".join(
                    """Fetching CLAP weights.
                The checkpoint is a ~2GB, so be patient.
                The process will start right after.
                """.split()
                )
            )
            fetch(
                tmp[-1],
                "/".join(tmp[:-1]),
                savedir=".",
                save_filename=weights_path,
            )

            ckpt_path = weights_path

        self.audio_encoder = AudioEncoder(
            audioenc_name,
            out_emb,
            d_proj,
            classes_num,
        )

        self.caption_encoder = TextEncoder(d_proj, text_model, transformer_embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        state_dict = torch.load(ckpt_path)["model"]
        self.load_state_dict(self.clean_state_dict(state_dict))
        print("Loaded pretrained CLAP checkpoint.")

    @staticmethod
    def clean_state_dict(state_dict):
        """Removes pre-processing keys from the state-dict."""
        keys_to_remove = []
        for k in state_dict:
            if "spectrogram" in k or "mel" in k:
                keys_to_remove.append(k)

        for k in keys_to_remove:
            state_dict.pop(
                k,
                None,
            )

        return state_dict

    def forward(self, audio, input_ids, token_type_ids, attention_mask, single=None):
        audio_embed = None
        caption_embed = None

        if not single == "txt":
            audio_embed, _, _ = self.audio_encoder(audio)
            audio_embed = audio_embed / audio_embed.norm(dim=1, keepdim=True)

        if not single == "aud":
            text = BatchEncoding(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
            )
            caption_embed = self.caption_encoder(text)
            caption_embed = caption_embed / caption_embed.norm(dim=1, keepdim=True)

        return caption_embed, audio_embed
