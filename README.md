# tinyCLAP: Distilling Contrastive Language-Audio Pretrained models

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/fpaissan/tinyCLAP) [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2311.14517) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/fpaissan/tinyCLAP/blob/main/LICENSE)

This repository contains the official implementation of [tinyCLAP](https://arxiv.org/abs/2311.14517).

![tinyCLAP overview](https://francescopaissan.it/tinyclapweb/assets/overview.png)

## Requirements

First of all, let's clone the repo and install the requirements:

```setup
git clone https://github.com/fpaissan/tinyCLAP & cd tinyCLAP
pip install -r extra_requirements.txt
```

## Training

You can distill your tinyCLAP model(s) using this command:

```bash
MODEL_NAME=phinet_alpha_1.50_beta_0.75_t0_6_N_7

./run_tinyCLAP.sh $MODEL_NAME $DATASET_LOCATION
```

Note that `MODEL_NAME` is formatted such that the script will automatically parse the configuration for the student model.
You can change parameters by changing the model name.

Please note:
- To use the original CLAP encoder in the distillation setting, replace the model name with `Cnn14`;
- To reproduce the variants of PhiNet from the manuscript, refer to the hyperparameters listed in Table 1.

## Evaluation

The command to evaluate the model on each dataset varies slightly among datasets.
Below are listed all the necessary commands.

### ESC50

```bash
python tinyclap.py hparams/distill_clap.yaml --experiment_name tinyCLAP_$MODEL_NAME --zs_eval True --esc_folder $PATH_TO_ESC
```

### UrbanSound8K

```bash
python tinyclap.py hparams/distill_clap.yaml --experiment_name tinyCLAP_$MODEL_NAME --zs_eval True --us8k_folder $PATH_TO_US8K
```

### TUT17

```bash
python tinyclap.py hparams/distill_clap.yaml --experiment_name tinyCLAP_$MODEL_NAME --zs_eval True --tut17_folder $PATH_TO_TUT17
```

## Pre-trained Models

You can download pretrained models from the [tinyCLAP HF](https://huggingface.co/fpaissan/tinyCLAP).

_Note_:  The checkpoints contain only the student model, so the text encoder will be downloaded separately.

To run inference using the pretrained models, use:

```bash
python tinyclap.py hparams/distill_clap.yaml --pretrained_clap fpaissan/tinyCLAP/$MODEL_NAME.ckpt --zs_eval True --esc_folder $PATH_TO_ESC
```

This command will automatically download the checkpoint if present in the zoo of pretrained models. Make sure to change the dataset configuration file based on the evaluation.
Please refer to the HF repo for a list of available tinyCLAP models.

## Citing tinyCLAP

```
@article{paissan2023tinyclap,
  title={tinyCLAP: Distilling Constrastive Language-Audio Pretrained Models},
  author={Paissan, Francesco and Farella, Elisabetta},
  journal={arXiv preprint arXiv:2311.14517},
  year={2023}
}
```
