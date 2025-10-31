NNODES=2
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 8 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 4 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 2 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 1 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 8 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 4 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 2 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 1 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 8 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 4 | grep "iter/s" >> /tmp/performance
#
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 2 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 4096 --num_attention_heads 32 --num_query_groups 32 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 1 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 8 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 4 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 2 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 32768 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 1 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 8 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 4 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 2 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 131072 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 1 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 8 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 4 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 2 | grep "iter/s" >> /tmp/performance

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FLASH_ATTN=1 PYTHONPATH=/workspace/AIAK-Megatron:/workspace/AIAK-Training-Omni:$PYTHONPATH \
torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT benchmark.py --hidden_size 8192 --num_attention_heads 64 --num_query_groups 8 --seq_len 1048576 --batch_size 1 --context_parallel_size 8 --context_parallel_ulysses_degree 1 | grep "iter/s" >> /tmp/performance
