#!/bin/bash
set -euxo pipefail

# Direct run script for DeepSeek V2 Lite
# This script contains hardcoded values from model_configs/benchmarking/DeepSeek-V2-Lite.yaml
# and runtime_configs/benchmarking/runtime.conf and common.conf

# Basic configuration
export MODEL="DeepSeek-V2-Lite"
export DATASET="wikitext_full" # wikipedia_20k" # slimpajama_15k"
export WORKSPACE=$(dirname "$(readlink -f "$0")")
export CLUSTER="todoMast"
export MCORE_RELEASE_VERSION="0.14"
export MEGATRON_PATH="/home/less/Megatron-LM"

# Parallelism configurations (from runtime.conf DeepSeek-V2-Lite config)
export TP=1
export PP=1
export EP=8
export CP=1
export VPP=1

# Batch size configurations
export MBS=1
export GBS=512

# Model architecture configurations
export NUM_LAYERS=27

# MoE configurations
export MOE_TOKEN_DISPATCHER="alltoall"
export MOE_GROUPED_GEMM="true"

# Training configurations
export NNODES=1
export RUN_TIME="00:20:00"
export PRETRAIN=1

# Data configurations
export SEQ_LEN=4096

# Common configurations
export TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"
export COMMENT="v${MCORE_RELEASE_VERSION}"
export WANDB_PROJECT="${USER}-moe-benchmarking-v${MCORE_RELEASE_VERSION}"
export PROFILE=0
export PR="bf16"

# Paths
export RUN_NAME="DeepSeek v2 Lite"
export DATA_PATH="${DATA_PATH:-/home/less/datasets/wikitext_full/wikitext_text_document}" #/home/less/datasets/wikipedia_20k/wikipedia_en_text_document}" # slimpajama_15k/slimpajama_text_document}"
export TOKENIZER_MODEL="deepseek-ai/DeepSeek-V2"
export OUTPUT_PATH="${OUTPUT_PATH:-${WORKSPACE}/outputs}"
export LOAD_PATH="${LOAD_PATH:-}"

# Environment variables (from DeepSeek-V2-Lite.yaml ENV_VARS)
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Remove deprecated TORCH_NCCL_AVOID_RECORD_STREAMS (now default)
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1

# Additional environment variables for overlapping
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0

# Build training parameters (from DeepSeek-V2-Lite.yaml MODEL_ARGS)
TRAINING_PARAMS="--distributed-timeout-minutes 3"
TRAINING_PARAMS+=" --tensor-model-parallel-size ${TP}"
TRAINING_PARAMS+=" --pipeline-model-parallel-size ${PP}"
TRAINING_PARAMS+=" --expert-model-parallel-size ${EP}"
TRAINING_PARAMS+=" --context-parallel-size ${CP}"
TRAINING_PARAMS+=" --expert-tensor-parallel-size 1"
TRAINING_PARAMS+=" --use-distributed-optimizer"
TRAINING_PARAMS+=" --use-mcore-models"
# Remove sequence-parallel since TP=1 (causes warning)
# TRAINING_PARAMS+=" --sequence-parallel"
TRAINING_PARAMS+=" --use-flash-attn"
TRAINING_PARAMS+=" --disable-bias-linear"
TRAINING_PARAMS+=" --micro-batch-size ${MBS}"
TRAINING_PARAMS+=" --global-batch-size ${GBS}"
TRAINING_PARAMS+=" --train-iters 100"
TRAINING_PARAMS+=" --exit-duration-in-mins 230"
TRAINING_PARAMS+=" --no-bias-swiglu-fusion"
TRAINING_PARAMS+=" --no-check-for-nan-in-loss-and-grad"
TRAINING_PARAMS+=" --no-rope-fusion"
TRAINING_PARAMS+=" --transformer-impl transformer_engine"
TRAINING_PARAMS+=" --seq-length ${SEQ_LEN}"
TRAINING_PARAMS+=" --data-cache-path ${WORKSPACE}/data_cache"
TRAINING_PARAMS+=" --tokenizer-type HuggingFaceTokenizer"
TRAINING_PARAMS+=" --tokenizer-model ${TOKENIZER_MODEL}"
TRAINING_PARAMS+=" --data-path ${DATA_PATH}"
TRAINING_PARAMS+=" --split 99,1,0"
TRAINING_PARAMS+=" --no-mmap-bin-files"
TRAINING_PARAMS+=" --no-create-attention-mask-in-dataloader"
TRAINING_PARAMS+=" --num-workers 6"
TRAINING_PARAMS+=" --num-layers ${NUM_LAYERS}"
TRAINING_PARAMS+=" --hidden-size 2048"
TRAINING_PARAMS+=" --ffn-hidden-size 10944"
TRAINING_PARAMS+=" --num-attention-heads 16"
TRAINING_PARAMS+=" --kv-channels 128"
TRAINING_PARAMS+=" --max-position-embeddings 4096"
TRAINING_PARAMS+=" --position-embedding-type rope"
TRAINING_PARAMS+=" --rotary-base 10000"
TRAINING_PARAMS+=" --make-vocab-size-divisible-by 3200"
TRAINING_PARAMS+=" --normalization RMSNorm"
TRAINING_PARAMS+=" --norm-epsilon 1e-6"
TRAINING_PARAMS+=" --swiglu"
TRAINING_PARAMS+=" --untie-embeddings-and-output-weights"
TRAINING_PARAMS+=" --multi-latent-attention"
TRAINING_PARAMS+=" --attention-dropout 0.0"
TRAINING_PARAMS+=" --hidden-dropout 0.0"
TRAINING_PARAMS+=" --clip-grad 1.0"
TRAINING_PARAMS+=" --weight-decay 0.1"
TRAINING_PARAMS+=" --qk-layernorm"
TRAINING_PARAMS+=" --lr-decay-iters 90"
TRAINING_PARAMS+=" --lr-warmup-iters 10"
TRAINING_PARAMS+=" --lr-warmup-init 1.3e-7"
TRAINING_PARAMS+=" --lr 1.3e-6"
TRAINING_PARAMS+=" --min-lr 1.3e-7"
TRAINING_PARAMS+=" --lr-decay-style cosine"
TRAINING_PARAMS+=" --adam-beta1 0.9"
TRAINING_PARAMS+=" --adam-beta2 0.95"
TRAINING_PARAMS+=" --num-experts 64"
TRAINING_PARAMS+=" --moe-layer-freq [0]+[1]*26"
TRAINING_PARAMS+=" --moe-ffn-hidden-size 1408"
TRAINING_PARAMS+=" --moe-shared-expert-intermediate-size 2816"
TRAINING_PARAMS+=" --moe-router-load-balancing-type seq_aux_loss"
TRAINING_PARAMS+=" --moe-router-topk 6"
TRAINING_PARAMS+=" --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER}"
TRAINING_PARAMS+=" --moe-router-pre-softmax"
# Add moe-grouped-gemm flag if enabled
if [[ "${MOE_GROUPED_GEMM}" == "true" ]]; then
    TRAINING_PARAMS+=" --moe-grouped-gemm"
fi
TRAINING_PARAMS+=" --moe-aux-loss-coeff 1e-3"
TRAINING_PARAMS+=" --moe-router-topk-scaling-factor 1.0"
TRAINING_PARAMS+=" --moe-router-dtype fp32"
TRAINING_PARAMS+=" --moe-permute-fusion"
TRAINING_PARAMS+=" --kv-lora-rank 512"
TRAINING_PARAMS+=" --qk-head-dim 128"
TRAINING_PARAMS+=" --qk-pos-emb-head-dim 64"
TRAINING_PARAMS+=" --v-head-dim 128"
TRAINING_PARAMS+=" --rotary-scaling-factor 40"
TRAINING_PARAMS+=" --mscale 0.707"
TRAINING_PARAMS+=" --mscale-all-dim 0.707"
TRAINING_PARAMS+=" --eval-iters 32"
TRAINING_PARAMS+=" --eval-interval 200"
TRAINING_PARAMS+=" --auto-detect-ckpt-format"

# Add load path if specified (typically not used for pretraining from scratch)
if [[ -n "${LOAD_PATH}" ]]; then
    TRAINING_PARAMS+=" --load ${LOAD_PATH}"
fi

TRAINING_PARAMS+=" --save ${OUTPUT_PATH}/checkpoints"
TRAINING_PARAMS+=" --save-interval 500"
TRAINING_PARAMS+=" --dist-ckpt-strictness log_all"
TRAINING_PARAMS+=" --init-method-std 0.02"
TRAINING_PARAMS+=" --log-timers-to-tensorboard"
TRAINING_PARAMS+=" --log-memory-to-tensorboard"
TRAINING_PARAMS+=" --log-num-zeros-in-grad"
TRAINING_PARAMS+=" --log-params-norm"
TRAINING_PARAMS+=" --log-validation-ppl-to-tensorboard"
TRAINING_PARAMS+=" --log-throughput"
TRAINING_PARAMS+=" --log-interval 1"
TRAINING_PARAMS+=" --tensorboard-dir ${OUTPUT_PATH}/tensorboard"
TRAINING_PARAMS+=" --wandb-project ${WANDB_PROJECT}"
TRAINING_PARAMS+=" --wandb-exp-name DeepSeek-V2-Lite-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}"
TRAINING_PARAMS+=" --bf16"

# Add overlapping parameters (since A2A_OVERLAP is not set, use default overlap settings)
TRAINING_PARAMS+=" --overlap-grad-reduce --overlap-param-gather"

# Append any command line arguments to TRAINING_PARAMS
if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} $@"
fi

# Set default output path if not set
export OUTPUT_PATH=${OUTPUT_PATH:-"${WORKSPACE}/outputs"}

# Export training command with distributed launch
export TRAINING_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

# Local execution logs
LOCAL_LOGS="${OUTPUT_PATH}/local_logs"
mkdir -p ${LOCAL_LOGS} || {
    echo "Error: Failed to create local logs directory ${LOCAL_LOGS}"
    exit 1
}

# Generate timestamp for log file
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
LOG_FILE="${LOCAL_LOGS}/${MODEL}-${RUN_NAME// /-}-${TIMESTAMP}.log"

echo "Starting DeepSeek V2 Lite direct run..."
echo "Model: ${MODEL}"
echo "Run Name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Training command: ${TRAINING_CMD}"
echo "Working directory: ${MEGATRON_PATH}"
echo ""
echo "Configuration:"
echo "  TP=${TP}, PP=${PP}, EP=${EP}, CP=${CP}, VPP=${VPP}"
echo "  MBS=${MBS}, GBS=${GBS}"
echo "  SEQ_LEN=${SEQ_LEN}"
echo "  NUM_LAYERS=${NUM_LAYERS}"
echo "  MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER}"
echo "  MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM}"
echo ""

# Change to Megatron directory and run training
cd "${MEGATRON_PATH}"

# Direct execution without container
echo "Running directly on host system"
eval "${TRAINING_CMD}" 2>&1 | tee "${LOG_FILE}"

echo "DeepSeek V2 Lite training completed. Log saved to: ${LOG_FILE}"
