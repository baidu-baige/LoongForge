#! /bin/bash
# The script needs to be run on at least 1 nodes.

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/aiak-training-llm/dataset/mllm/demo/wds/"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/qwen2_5-vl/qwen2_5-vl-7b-tp1-pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/qwen2_5-vl-7b"}

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

CONFIG_PATH=../configs/qwen/qwen2_5_vl
CONFIG_NAME=pretrain_qwen2_5_vl_7b

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/aiak_training_omni/train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    data.hf_tokenizer_path=$TOKENIZER_PATH \
    data.data_path=$DATA_PATH \
    data.tokenizer_type=HFTokenizer \
    data.dataloader_type=external \
    data.split=[100,0,0] \
    data.add_question_in_pretrain=True \
    data.enable_discard_sample=True \
    data.num_workers=16 \
    train.load=$CHECKPOINT_PATH \
    train.dataloader_save=${CHECKPOINT_PATH}/dataloader \
    train.tensorboard_dir=$TENSORBOARD_PATH \
    train._target_=dev.aiak_training_omni.utils.schema.PretrainConfig \
    train.norm_epsilon=1.0e_6 \
    train.training_phase=pretrain \
    train.seq_length=1024 \
    train.max_position_embeddings=4096 \
    train.init_method_std=0.02 \
    train.micro_batch_size=1 \
    train.global_batch_size=512 \
    train.lr=0.0002 \
    train.min_lr=1.0e_5 \
    train.clip_grad=1.0 \
    train.weight_decay=0.01 \
    train.optimizer=adam \
    train.adam_beta1=0.9 \
    train.adam_beta2=0.95 \
    train.adam_eps=1.0e_5 \
    train.train_iters=50000 \
    train.lr_decay_iters=50000 \
    train.lr_decay_style=cosine \
    train.lr_warmup_fraction=0.002 \
    train.initial_loss_scale=65536 \
    train.bf16=true \
    train.save_interval=10000000 \
    train.ckpt_format=torch \
    train.attention_backend=flash \
    train.pipeline_model_parallel_size=1 \
    train.tensor_model_parallel_size=1 \
    train.use_distributed_optimizer=true \
    train.overlap_grad_reduce=true \
    train.overlap_param_gather=true \
    train.distributed_backend=nccl \
    train.log_interval=1 \
    train.log_timers_to_tensorboard=true \
