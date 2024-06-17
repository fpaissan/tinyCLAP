#!/bin/bash

# This script runs the entire tinyCLAP optimization pipeline.
MODEL_NAME=$1
DATASET_LOCATION=$2

# Data configuration
AUDIOCAPS_FOLDER=$DATASET_LOCATION/AudioCaps/
MACS_FOLDER=$DATASET_LOCATION/MACs/
FSD50K_FOLDER=$DATASET_LOCATION/FSD50k/
CLOTHO_FOLDER=$DATASET_LOCATION/clotho/

find_best_model_ckpt() {
    local base_path="$1"
    local best_loss="inf"
    local best_model_ckpt_path=""

    # Ensure base_path is provided
    if [[ -z "$base_path" ]]; then
        echo "Error: base_path is required"
        return 1
    fi

    # Iterate over each CKPT.yaml file found in the base_path
    while IFS= read -r -d '' ckpt_path; do
        local model_ckpt_path
        model_ckpt_path=$(dirname "$ckpt_path")/student_model.ckpt

        # Extract the loss value from the CKPT.yaml file
        local loss
        loss=$(grep -oP '(?<=loss: )\S+' "$ckpt_path")

        # Default to inf if loss is not found
        if [[ -z "$loss" ]]; then
            loss="inf"
        fi

        # Compare losses (handling floating-point comparison)
        if [[ "$best_loss" == "inf" || $(awk "BEGIN {print ($loss < $best_loss)}") -eq 1 ]]; then
            best_loss="$loss"
            best_model_ckpt_path="$model_ckpt_path"
        fi
    done < <(find "$base_path" -type f -name "CKPT.yaml" -print0)

    # Output the best model checkpoint path
    echo "$best_model_ckpt_path"
}

python tinyclap.py hparams/distill_clap.yaml --audioenc_name_student $MODEL_NAME --experiment_name tinyCLAP_stage1_$MODEL_NAME \
      --audiocaps_folder $AUDIOCAPS_FOLDER --macs_folder $MACS_FOLDER \
      --fsd50k_folder $FSD50K_FOLDER --clotho_folder $CLOTHO_FOLDER

best_model=$(find_best_model_ckpt "results/tinyCLAP_stage1_$MODEL_NAME")
echo "Projection finetuning will start from $best_model"

python tinyclap.py hparams/distill_clap.yaml --audioenc_name_student $MODEL_NAME --experiment_name tinyCLAP_stage2_$MODEL_NAME \
      --audiocaps_folder $AUDIOCAPS_FOLDER --macs_folder $MACS_FOLDER \
      --fsd50k_folder $FSD50K_FOLDER --clotho_folder $CLOTHO_FOLDER \
      --projection_only True --number_of_epochs 10 --lr 0.004 --pretrained_CLAP $best_model
