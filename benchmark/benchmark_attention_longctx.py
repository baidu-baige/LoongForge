"""
benchmark_attention_longctx.py
"""
import os
import torch

from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from aiak_training_omni.models.llama.llama_layer_spec import get_llama_layer_with_spec

from megatron.training.global_vars import set_args

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--hidden_size", type=int, default=8192, help="hidden size")
parser.add_argument("--num_query_groups", type=int, default=8, help="num_query_groups")
parser.add_argument("--num_attention_heads", type=int, default=64, help="num_attention_heads")
parser.add_argument("--seq_len", type=int, default=128 * 1024, help="seq len")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tensor_parallel_size", type=int, default=4, help="tensor_parallel_size")
parser.add_argument("--context_parallel_size", type=int, default=8, help="context_parallel_size")
parser.add_argument("--context_parallel_ulysses_degree", type=int, default=1,
                    help="context_parallel_ulysses_degree")

args = parser.parse_args()
set_args(args)


def test_gpu_forward(num_iter=100):
    """
    test gpu forward
    """
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    world_size = args.world_size
    rank = args.rank
    print(f"world_size:{world_size}, rank:{rank}", flush=True)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        init_method=init_method,
    )

    torch.distributed.barrier()
    Utils.initialize_model_parallel(args.tensor_parallel_size, 1, context_parallel_size=args.context_parallel_size,
                                    context_parallel_ulysses_degree=args.context_parallel_ulysses_degree)
    model_parallel_cuda_manual_seed(123)
    transformer_config = TransformerConfig(num_layers=1, hidden_size=args.hidden_size,
                                           num_attention_heads=args.num_attention_heads,
                                           num_query_groups=args.num_query_groups, use_cpu_initialization=True,
                                           bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
                                           tensor_model_parallel_size=args.tensor_parallel_size,
                                           autocast_dtype=torch.bfloat16,
                                           context_parallel_size=args.context_parallel_size,
                                           context_parallel_ulysses_degree=args.context_parallel_ulysses_degree,
                                           cp_comm_type="all_gather")
    parallel_attention = SelfAttention(transformer_config,
                                       get_llama_layer_with_spec().submodules.self_attention.submodules,
                                       layer_number=1,
                                       attn_mask_type=AttnMaskType.causal)

    config = parallel_attention.config
    sequence_length = int(args.seq_len / parallel_attention.config.context_parallel_size)
    micro_batch_size = args.batch_size

    parallel_attention.cuda(device)

    # [sequence length, batch size, hidden size]
    hidden_states = torch.ones((sequence_length, micro_batch_size,
                                parallel_attention.config.hidden_size), dtype=torch.bfloat16)
    hidden_states = hidden_states.cuda(device)

    attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda(device)

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    with torch.no_grad():
        for _ in range(num_iter):
            output, bias = parallel_attention(hidden_states, attention_mask)
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0
    if rank == 0:
        print(f"{num_iter / time:.3f} iter/s, {time:.3f} sec")
    Utils.destroy_model_parallel()


test_gpu_forward()
