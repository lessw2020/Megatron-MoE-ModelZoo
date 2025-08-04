#!/bin/bash
set -euxo pipefail



# Path to Megatron-MoE-Scripts
export WORKSPACE=$(dirname "$(readlink -f "$0")")

# Benchmarking configurations (must be set)
export MODEL=${MODEL:-"your_own_model"}
#export CLUSTER=${CLUSTER:-"your_own_cluster"}
export CLUSTER="todoMast"
#export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-"your_own_megatron_version"} # Version and release info
export MCORE_RELEASE_VERSION="0.14"
export MEGATRON_PATH="~/Megatron-LM"
#export MEGATRON_PATH=${MEGATRON_PATH:-"your_own_megatron_path"} # Path to Megatron-LM
#export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"} # Path to .sqsh or docker image url
#export WANDB_API_KEY=${WANDB_API_KEY:-"your_own_wandb_api_key"} # Wandb API key

# Load common configurations
source "${WORKSPACE}/runtime_configs/benchmarking/common.conf"
# Load model-specific configurations
source "${WORKSPACE}/runtime_configs/benchmarking/runtime.conf"
# Load cluster configurations
#source "${WORKSPACE}/cluster_configs/benchmarking/${CLUSTER}.conf"
# we manually set the cluster configs of relevance
export RUN_NAME=${RUN_NAME:-"DeepSeek v2 Lite"}
# Set DATA_PATH based on MODEL
export DATA_PATH=${DATA_PATH:-${DATA_PATHS["${DATASET}:${MODEL}"]}}

# Set TOKENIZER_MODEL based on MODEL
export TOKENIZER_MODEL=$(get_path_based_on_model "TOKENIZER_MODELS" "${MODEL}")




# Initialize training parameters
TRAINING_PARAMS=${TRAINING_PARAMS:-""}

# Process training parameters
if [[ -f ${TRAINING_PARAMS_PATH} ]]; then
    envsubst < ${TRAINING_PARAMS_PATH} > ${TRAINING_PARAMS_PATH}.tmp
    TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}.tmp
else
    echo "Error: TRAINING_PARAMS_PATH does not exist: ${TRAINING_PARAMS_PATH}."
    exit 1
fi

# Extract training parameters to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] |
    select(.value != "false") |
    with(select(.value == "true"); .value = "") |
    [.key + " " + .value] | join("")' ${TRAINING_PARAMS_PATH} | tr '\n' ' ')
TRAINING_PARAMS="${TRAINING_PARAMS} ${TRAINING_PARAMS_FROM_CONFIG}"

# Append any command line arguments to TRAINING_PARAMS
if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} $@"
fi

# Extract environment variables to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' ${TRAINING_PARAMS_PATH})
while IFS='=' read -r KEY VALUE; do
    if [[ -n ${KEY} ]]; then
        export "${KEY}"="${VALUE}"
        echo "${KEY}=${VALUE}"
    fi
done < <(echo "${ENV_VARS}" | tr ' ' '\n')

# Virtual pipeline parallelism arguments
if [[ ${VPP} -gt 1 ]]; then
    if [[ ! "${TRAINING_PARAMS}" =~ "--pipeline-model-parallel-layout" ]] && \
       [[ ! "${TRAINING_PARAMS}" =~ "--num-virtual-stages-per-pipeline-rank" ]] && \
       [[ ! "${TRAINING_PARAMS}" =~ "--num-layers-per-virtual-pipeline-stage" ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --num-virtual-stages-per-pipeline-rank ${VPP}"
    fi
fi

# Uneven pipeline parallelism arguments
if [[ $((NUM_LAYERS % PP)) -ne 0 ]]; then
    if [[ ! "${TRAINING_PARAMS}" =~ "--pipeline-model-parallel-layout" ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --decoder-first-pipeline-num-layers ${PP_FIRST} --decoder-last-pipeline-num-layers ${PP_LAST}"
    fi
fi

OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-0}
if [[ ${OPTIMIZER_OFFLOAD} == 1 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d"
fi

# FP8 arguments
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-recipe blockwise --fp8-format e4m3"
    if [[ ${OPTIMIZER_OFFLOAD} == 0 ]]; then
        TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-param-gather" # Optimizer CPU offload does not support fp8 param gather now.
    fi
    TRAINING_PARAMS="${TRAINING_PARAMS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-router-padding-for-fp8"
fi

# 1F1B overlapping arguments and environment variables
A2A_OVERLAP=${A2A_OVERLAP:-0}
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=20
    export NVTE_BWD_LAYERNORM_SM_MARGIN=20
    TRAINING_PARAMS="${TRAINING_PARAMS} --delay-wgrad-compute --overlap-moe-expert-parallel-comm"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
    TRAINING_PARAMS="${TRAINING_PARAMS} --overlap-grad-reduce --overlap-param-gather"
fi

# Long context arguments
if [[ ${SEQ_LEN} -gt 4096 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --max-position-embeddings ${SEQ_LEN}"
fi

# Profile command
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 20 --profile-step-end 22 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

# Export training command
export TRAINING_CMD="${PROFILE_CMD} python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

# Local execution logs
LOCAL_LOGS="${OUTPUT_PATH}/local_logs"
mkdir -p ${LOCAL_LOGS} || {
    echo "Error: Failed to create local logs directory ${LOCAL_LOGS}"
    exit 1
}

# Generate timestamp for log file
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
LOG_FILE="${LOCAL_LOGS}/${MODEL}-${RUN_NAME}-${TIMESTAMP}.log"

echo "Starting local benchmarking run..."
echo "Model: ${MODEL}"
echo "Run Name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Training command: ${TRAINING_CMD}"
echo "Working directory: ${MEGATRON_PATH}"

# Change to Megatron directory and run training
cd "${MEGATRON_PATH}"

# Check if we need to use container or run directly
if [[ "${CONTAINER_IMAGE}" != "your_own_container_image" ]] && [[ -n "${CONTAINER_IMAGE}" ]]; then
    # Container-based execution
    echo "Running with container: ${CONTAINER_IMAGE}"

    # Determine container runtime (Docker or Singularity/Apptainer)
    if command -v docker &> /dev/null && [[ "${CONTAINER_IMAGE}" =~ ^[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$ ]]; then
        # Docker execution
        CONTAINER_MOUNTS_DOCKER=""
        if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
            # Convert SLURM-style mounts to Docker format
            IFS=',' read -ra MOUNTS <<< "${CONTAINER_MOUNTS}"
            for mount in "${MOUNTS[@]}"; do
                CONTAINER_MOUNTS_DOCKER="${CONTAINER_MOUNTS_DOCKER} -v ${mount}"
            done
        fi

        docker run --rm --gpus all \
            ${CONTAINER_MOUNTS_DOCKER} \
            -w "${MEGATRON_PATH}" \
            -e WANDB_API_KEY="${WANDB_API_KEY}" \
            -e CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
            -e NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN}" \
            -e NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN}" \
            "${CONTAINER_IMAGE}" \
            bash -c "${TRAINING_CMD}" 2>&1 | tee "${LOG_FILE}"
    elif command -v singularity &> /dev/null || command -v apptainer &> /dev/null; then
        # Singularity/Apptainer execution
        CONTAINER_CMD="singularity"
        if command -v apptainer &> /dev/null; then
            CONTAINER_CMD="apptainer"
        fi

        BIND_MOUNTS=""
        if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
            BIND_MOUNTS="--bind ${CONTAINER_MOUNTS}"
        fi

        ${CONTAINER_CMD} exec --nv \
            ${BIND_MOUNTS} \
            --pwd "${MEGATRON_PATH}" \
            "${CONTAINER_IMAGE}" \
            bash -c "${TRAINING_CMD}" 2>&1 | tee "${LOG_FILE}"
    else
        echo "Error: Container image specified but no compatible container runtime found (Docker/Singularity/Apptainer)"
        exit 1
    fi
else
    # Direct execution without container
    echo "Running directly on host system"
    bash -c "${TRAINING_CMD}" 2>&1 | tee "${LOG_FILE}"
fi

echo "Benchmarking completed. Log saved to: ${LOG_FILE}"
