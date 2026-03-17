#!/bin/bash

# This script prunes a machine learning model to reduce its size.
# Usage: ./prune_model.sh <config_file> <output_model_path> <device> <model_name> <no_prune> <no_quantize>

# Set variables
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Default values
CONFIG_FILE="${PROJECT_ROOT}/configs/prune_config.yaml"
OUTPUT_MODEL_PATH="${PROJECT_ROOT}/models/pruned_model.nemo"
DEVICE="cpu"
MODEL_NAME="titanet_large"
NO_PRUNE=false
NO_QUANTIZE=false

while [ $i -le $l ]
do
    # Checking for the parameters
    if [[ $1 == "--config" ]]; then
        CONFIG_FILE=$2
        shift 2
    elif [[ $1 == "--output" ]]; then
        OUTPUT_MODEL_PATH=$2
        shift 2
    elif [[ $1 == "--device" ]]; then
        DEVICE=$2
        shift 2
    elif [[ $1 == "--model-name" ]]; then
        MODEL_NAME=$2
        shift 2
    elif [[ $1 == "--no-prune" ]]; then
        NO_PRUNE=true
        shift 1
    elif [[ $1 == "--no-quantize" ]]; then
        NO_QUANTIZE=true
        shift 1
    else
        echo "Set to default value..."
    fi

    i=$((i + 1));
    shift 1;
done

# Run the pruning script
echo "Starting model pruning..."
python ${PROJECT_ROOT}/src/prune_model.py \
    --config ${PROJECT_ROOT}/configs/prune_config.yaml \
    --output ${OUTPUT_MODEL_PATH} \
    --device ${DEVICE} \
    --model-name ${MODEL_NAME} \
    $( [ "$NO_PRUNE" == "true" ] && echo "--no-prune" ) \
    $( [ "$NO_QUANTIZE" == "true" ] && echo "--no-quantize" )