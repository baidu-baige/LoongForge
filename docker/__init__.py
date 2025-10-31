#!/usr/bin/env python3
"""init"""
def load(args):
    """load"""
    import scaled_upper_triang_masked_softmax_cuda
    import scaled_masked_softmax_cuda
    import scaled_softmax_cuda
    import parallel_attention
    import rotary_positional_embedding_cuda
    import matmul_reduce_parallel_cuda