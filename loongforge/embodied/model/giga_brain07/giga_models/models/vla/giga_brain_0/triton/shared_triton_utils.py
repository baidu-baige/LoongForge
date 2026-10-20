import torch
import triton
import triton.language as tl


@triton.jit
def matmul_small_bias_res(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :], mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden), other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_small_bias_res_mod(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    i_mod: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] % i_mod) * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :], mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden), other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def scaled_matmul_small_bias_res(
    inp_ptr,
    inp_norm_factor_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M

        norm_factor = tl.load(inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N), mask=i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0)
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :], mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden), other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            x = x * norm_factor[:, None]
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_small_bias(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :], mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden), other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_small_bias_silu(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :], mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden), other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_small_res(
    inp_ptr,
    weight_ptr,
    out_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_small(
    inp_ptr,
    weight_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_small_bias_gelu(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :], mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden), other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def matmul_split_k(
    inp_ptr,
    weight_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    grid_k = SPLIT_K
    for p in range(pid, grid_i * grid_j * grid_k, psize):
        i = (p // (grid_j * grid_k)) * BLOCK_SIZE_N
        j = (p // grid_k % grid_j) * BLOCK_SIZE_M
        k_s = p % grid_k
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        k_beg = k_s * tl.cdiv(features, BLOCK_SIZE_K) // SPLIT_K * BLOCK_SIZE_K
        k_end = (k_s + 1) * tl.cdiv(features, BLOCK_SIZE_K) // SPLIT_K * BLOCK_SIZE_K
        for k in range(k_beg, k_end, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :] + k_s * seq_len * hidden,
            acc,
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.jit
def merge_split_k_bias_res(
    out_ptr, bias_ptr, res_ptr, final_out_ptr, seq_len: tl.constexpr, hidden: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr = 512
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK, seq_len * hidden, psize * BLOCK):
        acc = tl.load(res_ptr + i + tl.arange(0, BLOCK), mask=(i + tl.arange(0, BLOCK)) < (seq_len * hidden), other=0.0).to(tl.float32)
        acc += tl.load(bias_ptr + (i + tl.arange(0, BLOCK)) % hidden, mask=(i + tl.arange(0, BLOCK)) < (seq_len * hidden), other=0.0).to(tl.float32)
        for k in range(SPLIT_K):
            offset = k * seq_len * hidden
            mask = (i + tl.arange(0, BLOCK)) < (seq_len * hidden)
            vals = tl.load(out_ptr + offset + i + tl.arange(0, BLOCK), mask=mask, other=0.0)
            acc += vals.to(tl.float32)
        mask = (i + tl.arange(0, BLOCK)) < (seq_len * hidden)
        tl.store(final_out_ptr + i + tl.arange(0, BLOCK), acc.to(tl.bfloat16), mask=mask)


def conv2d_embed_n256_1152_res(images, patch_w, patch_b, pos_emb, out):
    nviews = images.shape[0]
    img_input = images.view(nviews, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5).contiguous()
    matmul_small_bias_res_mod[(256 * nviews // 64) * (1152 // 64),](
        img_input,
        patch_w,
        out,
        patch_b,
        pos_emb,
        seq_len=256 * nviews,
        features=3 * 14 * 14,
        hidden=1152,
        i_mod=256,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32,
    )


@triton.jit
def layer_norm_small_kernel(x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, seq_len: tl.constexpr, features: tl.constexpr, eps: tl.constexpr = 1e-5):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN: tl.constexpr = 2048

    for i in range(pid, seq_len, psize):
        x = tl.load(x_ptr + i * features + tl.arange(0, MAX_LEN), mask=tl.arange(0, MAX_LEN) < features, other=0.0)
        mean = tl.sum(x) * (1.0 / features)
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered * (tl.arange(0, MAX_LEN) < features)) * (1.0 / features)
        inv_std = 1.0 / tl.sqrt(var + eps)
        x = x_centered * inv_std
        x = x * tl.load(norm_w_ptr + tl.arange(0, MAX_LEN), mask=tl.arange(0, MAX_LEN) < features, other=0.0)
        x = x + tl.load(norm_b_ptr + tl.arange(0, MAX_LEN), mask=tl.arange(0, MAX_LEN) < features, other=0.0)
        tl.store(out_ptr + i * features + tl.arange(0, MAX_LEN), x.to(tl.bfloat16), mask=tl.arange(0, MAX_LEN) < features)


def layer_norm_n256_1152(x, norm_w, norm_b, out):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    layer_norm_small_kernel[seq_len,](x, out, norm_w, norm_b, seq_len=seq_len, features=1152)


def layer_norm_QKV_matmul_n256_1152_3456_bias(x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    layer_norm_small_kernel[seq_len,](x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152)

    matmul_small_bias[((seq_len + 63) // 64) * (3456 // 64),](
        x_norm, qkv_w, out, qkv_b, seq_len=seq_len, features=1152, hidden=3456, BLOCK_SIZE_N=64, BLOCK_SIZE_M=64, BLOCK_SIZE_K=32
    )


@triton.jit
def matmul_512x1152x1152_twopart_bias_res(
    old_ptr, inp_ptr, weight_ptr, bias_ptr, out_ptr, out2_ptr, seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    hidden1 = 1024

    BLOCK_SIZE_N1: tl.constexpr = 64
    BLOCK_SIZE_M1: tl.constexpr = 64
    BLOCK_SIZE_K1: tl.constexpr = 32
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N1)
    grid_j = tl.cdiv(hidden1, BLOCK_SIZE_M1)

    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N1
        j = (p % grid_j) * BLOCK_SIZE_M1
        acc = tl.load(old_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :]).to(tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :]).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_K1):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K1))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N1))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K1))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K1))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K1))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M1))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N1))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M1))[None, :] < hidden),
        )

    BLOCK_SIZE_N2: tl.constexpr = 32
    BLOCK_SIZE_M2: tl.constexpr = 32
    BLOCK_SIZE_K2: tl.constexpr = 32
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N2)
    grid_j = tl.cdiv(hidden - hidden1, BLOCK_SIZE_M2)
    grid_k = 4
    for p in range(pid, grid_i * grid_j * grid_k, psize):
        i = (p // (grid_j * grid_k)) * BLOCK_SIZE_N2
        j = ((p // grid_k) % grid_j) * BLOCK_SIZE_M2 + hidden1
        k0 = p % grid_k
        acc = tl.zeros((BLOCK_SIZE_N2, BLOCK_SIZE_M2), dtype=tl.float32)
        if k0 == 0:
            acc += tl.load(old_ptr + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M2))[None, :]).to(tl.float32)
            acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M2))[None, :]).to(tl.float32)
        for k in range(k0 * BLOCK_SIZE_K2, features, BLOCK_SIZE_K2 * grid_k):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K2))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N2))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K2))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K2))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M2))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K2))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M2))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out2_ptr
            + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * ((hidden - hidden1) * grid_k)
            + (j - hidden1 + tl.arange(0, BLOCK_SIZE_M2))[None, :]
            + k0 * (hidden - hidden1),
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N2))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M2))[None, :] < hidden),
        )


@triton.jit
def combine_1536_1152_twopart(out_ptr, inp_ptr, seq_len: tl.constexpr, hidden: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * 2, seq_len, psize * 2):
        inp = tl.load(inp_ptr + (i + tl.arange(0, 2)[:, None]) * 128 * 4 + tl.arange(0, 128)).to(tl.float32)
        for j in range(1, 4):
            inp += tl.load(inp_ptr + (i + tl.arange(0, 2)[:, None]) * 128 * 4 + tl.arange(0, 128) + 128 * j).to(tl.float32)
        tl.store(out_ptr + (i + tl.arange(0, 2)[:, None]) * hidden + 1024 + tl.arange(0, 128), inp.to(tl.bfloat16))


def matmul_n256_1152_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    if num_views == 2:
        matmul_512x1152x1152_twopart_bias_res[(256,)](
            old_ptr=res,
            inp_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            out2_ptr=buf,
            seq_len=512,
            features=1152,
            hidden=1152,
        )
        combine_1536_1152_twopart[(256,)](
            inp_ptr=buf,
            out_ptr=out,
            seq_len=512,
            hidden=1152,
        )
        return
    seq_len = 256 * num_views
    matmul_small_bias_res[((seq_len + 31) // 32) * (1152 // 64),](
        x, weight, out, bias, res, seq_len=seq_len, features=1152, hidden=1152, BLOCK_SIZE_N=32, BLOCK_SIZE_M=64, BLOCK_SIZE_K=32
    )


def layer_norm_matmul_n256_1152_4304_bias_gelu(x, norm_w, norm_b, weight, bias, out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views

    layer_norm_small_kernel[seq_len,](x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152)

    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    matmul_small_bias_gelu[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * ((4304 + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M),](
        x_norm,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=1152,
        hidden=4304,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


def matmul_n256_4304_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    matmul_split_k[((seq_len + 64) // 64) * (1152 // 64) * 4,](
        x, weight, buf, seq_len=seq_len, features=4304, hidden=1152, BLOCK_SIZE_N=64, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64, SPLIT_K=4
    )
    merge_split_k_bias_res[(seq_len * 1152 + 1023) // 1024,](buf, bias, res, out, seq_len=seq_len, hidden=1152, SPLIT_K=4)


@torch.compile
def AttnMultiKey(QKV):
    QKV = QKV.view(-1, 256, 3, 16, 72).permute(0, 2, 3, 1, 4)
    Q = QKV[:, 0]
    K = QKV[:, 1]
    V = QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(Q.shape[0], 256, 1152)
    return attn


@torch.compile
def AttnMultiKeyMask(Q, K, V, mask=None):
    Q = Q.view(1, -1, 8, 256).transpose(1, 2)
    K = K.view(1, 1, -1, 256)
    V = V.view(1, 1, -1, 256)
    if mask is not None:
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask[None, None], is_causal=False)
    else:
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    attn = attn.transpose(1, 2).reshape(-1, 2048)
    return attn


@torch.compile
def AttnSingleKey(Q, K, V, mask=None):
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    logits = torch.matmul(Q, K.T) * scale
    if mask is not None:
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.reshape(mask.shape[0], -1, logits.shape[-1])
        logits = torch.where(mask[:, None, :], logits, neg_inf)
        logits = logits.reshape(-1, logits.shape[-1])
    logits = torch.nn.functional.softmax(logits, dim=-1)
    attn = torch.matmul(logits, V).view(-1, 2048)
    return attn


@triton.jit
def rms_norm_kernel(inp_ptr, out_ptr, seq_len: tl.constexpr, features: tl.constexpr):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    BLOCK_SIZE: tl.constexpr = 512
    for i in range(pid, seq_len, psize):
        sum_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + 1e-6)

        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            x = x * factor
            tl.store(out_ptr + i * features + j + tl.arange(0, BLOCK_SIZE), x)


@triton.jit
def matmul_small_gate(
    inp_ptr,
    weight1_ptr,
    weight2_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr = 128,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
):
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)

    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, hidden, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                    mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                    other=0.0,
                )
                w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
                w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc2 = tl.dot(x, w2, acc2)
            acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
            acc = (acc * acc2).to(tl.bfloat16)
            tl.store(
                out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
                acc,
                mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
            )


@triton.jit
def scaled_matmul_small_gate(
    inp_ptr,
    inp_norm_factor_ptr,
    weight1_ptr,
    weight2_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N), mask=i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other=0.0,
            )
            x = x * norm_factor[:, None]
            w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc = tl.dot(x, w, acc)
            w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc2 = tl.dot(x, w2, acc2)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
            acc,
            mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
        )


@triton.jit
def scaled_matmul_small_bias_gate(
    inp_ptr,
    inp_norm_factor_ptr,
    weight1_ptr,
    bias1_ptr,
    weight2_ptr,
    bias2_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N), mask=i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other=0.0,
            )
            x = x * norm_factor[:, None]
            w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc = tl.dot(x, w, acc)
            w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc2 = tl.dot(x, w2, acc2)
        bias1 = tl.load(bias1_ptr + j + tl.arange(0, BLOCK_SIZE_M), mask=j + tl.arange(0, BLOCK_SIZE_M) < hidden, other=0)
        bias2 = tl.load(bias2_ptr + j + tl.arange(0, BLOCK_SIZE_M), mask=j + tl.arange(0, BLOCK_SIZE_M) < hidden, other=0)
        acc = acc + bias1[None]
        acc2 = acc2 + bias2[None]
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
            acc,
            mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
        )


def rms_matmul_n_2048_16384_gate(x, weight1, weight2, out, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    matmul_small_gate[((seq_len + 127) // 128, (16384 + 63) // 64)](x_norm, weight1, weight2, out, seq_len, 2048, 16384)


def matmul_n_16384_2048_res(x, weight, out):
    seq_len = x.shape[0]
    BLOCK_SIZE_N = 128
    if seq_len < 512:
        BLOCK_SIZE_N = 64
    matmul_small_res[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * (2048 // 64),](
        x, weight, out, out, seq_len=seq_len, features=16384, hidden=2048, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64
    )


def layer_norm_matmul_n256_1152_2048_bias(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    seq_len = x.shape[0] * 256
    layer_norm_small_kernel[seq_len,](x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152, eps=1e-5)
    matmul_small_bias[((seq_len + 63) // 64) * (2048 // 64),](
        x_norm, proj_w, out, proj_b, seq_len=seq_len, features=1152, hidden=2048, BLOCK_SIZE_N=64, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64
    )


@triton.jit
def matmul_n_2048_2560_qkv_rope(
    inp_ptr,
    weight_QKV_ptr,
    rope_weights_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    head_dim: tl.constexpr,
    num_head: tl.constexpr,
):
    BLOCK_SIZE_N: tl.constexpr = 64
    BLOCK_SIZE_M: tl.constexpr = 64
    BLOCK_SIZE_K: tl.constexpr = 64
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)
    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, (num_head + 2) * head_dim, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                    mask=tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i,
                    other=0.0,
                )
                w = tl.load(weight_QKV_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * ((num_head + 2) * head_dim) + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
            acc = acc.to(tl.bfloat16)
            if j < (num_head + 1) * head_dim:
                x0, x1 = tl.split(acc.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M // 2, 2))
                x_cossin = tl.load(
                    rope_weights_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * head_dim + j % head_dim + tl.arange(0, BLOCK_SIZE_M)[None, :],
                    mask=tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i,
                    other=0.0,
                )
                x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M // 2, 2))
                x0_ = x0 * x_cos - x1 * x_sin
                x1_ = x1 * x_cos + x0 * x_sin
                acc = tl.interleave(x0_, x1_)
            if j < num_head * head_dim:
                out_ptr = Q_ptr
                out_stride = num_head * head_dim
            elif j < (num_head + 1) * head_dim:
                out_ptr = K_ptr
                out_stride = head_dim
            else:
                out_ptr = V_ptr
                out_stride = head_dim
            tl.store(
                out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * out_stride + j % out_stride + tl.arange(0, BLOCK_SIZE_M),
                acc,
                mask=tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i,
            )


def rms_matmul_n_2048_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    matmul_n_2048_2560_qkv_rope[((seq_len + 63) // 64, 2560 // 64)](x_norm, weight_qkv, rope_weight, Q, K, V, seq_len, 2048, 256, 8)


def matmul_n_2048_2048_res(x, weight, out):
    seq_len = x.shape[0]
    matmul_small_res[((seq_len + 127) // 128) * (2048 // 64),](
        x, weight, out, out, seq_len=seq_len, features=2048, hidden=2048, BLOCK_SIZE_N=128, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64
    )


@triton.jit
def matvec_bias_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, features: tl.constexpr, hidden: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    BLOCK_SIZE_N: tl.constexpr = 32
    BLOCK_SIZE_M: tl.constexpr = 8
    for j in range(pid * BLOCK_SIZE_M, hidden, psize * BLOCK_SIZE_M):
        acc = tl.load(bias_ptr + j + tl.arange(0, BLOCK_SIZE_M)).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_N):
            x = tl.load(x_ptr + k + tl.arange(0, BLOCK_SIZE_N), mask=(k + tl.arange(0, BLOCK_SIZE_N) < features), other=0.0)
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=(k + tl.arange(0, BLOCK_SIZE_N)[:, None] < features) & (j + tl.arange(0, BLOCK_SIZE_M)[None, :] < hidden),
                other=0.0,
            )
            acc += tl.sum(x[:, None] * w, axis=0)
        tl.store(out_ptr + j + tl.arange(0, BLOCK_SIZE_M), acc.to(tl.bfloat16), mask=j + tl.arange(0, BLOCK_SIZE_M) < hidden)


def matmul_1_32_1024_bias(x, weight, bias, out):
    matvec_bias_kernel[((1024 + 7) // 8,)](x, weight, bias, out, features=32, hidden=1024)


def matmul_k_32_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),](
        x, weight, out, bias, seq_len=seq_len, features=32, hidden=1024, BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=32
    )


def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),](
        x, weight, out, bias, seq_len=seq_len, features=32, hidden=1024, BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=32
    )


def matmul_k_1024_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),](
        x, weight, out, bias, seq_len=seq_len, features=1024, hidden=1024, BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=64
    )


@triton.jit
def rmsnorm_factor_kernel(inp_ptr, factor_ptr, rows: tl.constexpr, features: tl.constexpr, eps: tl.constexpr = 1e-6, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    for i in range(pid, rows, psize):
        sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            offs = j + tl.arange(0, BLOCK_SIZE)
            x = tl.load(inp_ptr + i * features + offs, mask=offs < features, other=0.0)
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + eps)
        tl.store(factor_ptr + i, factor)


def rms_matmul_k_1024_32_bias_res(x, weight, bias, out, x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(seq_len,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),](
        x, x_norm_factor, weight, out, bias, out, seq_len=seq_len, features=1024, hidden=32, BLOCK_SIZE_N=16, BLOCK_SIZE_M=16, BLOCK_SIZE_K=256
    )


@triton.jit
def scaled_matmul_rope_qkv(
    inp_ptr,
    inp_norm_factor_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    weight_qkv_ptr,
    rope_weights_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = tl.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f'head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}'

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        norm_factor = tl.load(
            inp_norm_factor_ptr + start_i + tl.arange(0, BLOCK_SIZE_M), mask=start_i + tl.arange(0, BLOCK_SIZE_M) < seq_len, other=0
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(inp_ptr + offs_i * features + offs_k[None, :], mask=offs_k[None, :] < features, other=0)
            w = tl.load(weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j, mask=offs_k[:, None] < features, other=0)
            accumulator = tl.dot(x, w, accumulator)
        accumulator = accumulator * norm_factor[:, None]

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(out_ptr + offs_i * out_stride + offs_j % out_stride, accumulator, mask=(offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim))
        pid += psize


@triton.jit
def scaled_matmul_rope_qkv_bias(
    inp_ptr,
    inp_norm_factor_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    weight_qkv_ptr,
    bias_qkv_ptr,
    rope_weights_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    output_features = (num_heads + 2) * head_dim
    grid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = tl.cdiv(output_features, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f'head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}'

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        norm_factor = tl.load(
            inp_norm_factor_ptr + start_i + tl.arange(0, BLOCK_SIZE_M), mask=start_i + tl.arange(0, BLOCK_SIZE_M) < seq_len, other=0
        )
        bias_factor = tl.load(
            bias_qkv_ptr + start_j + tl.arange(0, BLOCK_SIZE_N), mask=start_j + tl.arange(0, BLOCK_SIZE_N) < output_features, other=0
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(inp_ptr + offs_i * features + offs_k[None, :], mask=offs_k[None, :] < features, other=0)
            w = tl.load(weight_qkv_ptr + offs_k[:, None] * output_features + offs_j, mask=offs_k[:, None] < features, other=0)
            accumulator = tl.dot(x, w, accumulator)
        accumulator = accumulator * norm_factor[:, None]
        accumulator = accumulator + bias_factor[None]

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(out_ptr + offs_i * out_stride + offs_j % out_stride, accumulator, mask=(offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim))
        pid += psize


def rms_matmul_k_1024_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm_factor, bias_qkv=None):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    if bias_qkv is None:
        scaled_matmul_rope_qkv[(128,)](
            x,
            x_norm_factor,
            seq_len,
            1024,
            256,
            8,
            weight_qkv,
            rope_weight,
            Q,
            K,
            V,
        )
    else:
        scaled_matmul_rope_qkv_bias[(128,)](
            x,
            x_norm_factor,
            seq_len,
            1024,
            256,
            8,
            weight_qkv,
            bias_qkv,
            rope_weight,
            Q,
            K,
            V,
        )


@triton.jit
def matmul_abT_scale(
    q_ptr,
    k_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    scale_factor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 32,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(q_ptr + offs_i[:, None] * K + offs_k[None, :], mask=offs_k[None, :] < K, other=0)
            w = tl.load(k_ptr + offs_j[:, None] * K + offs_k[None, :], mask=offs_k[None, :] < K, other=0)
            accumulator = tl.dot(x, tl.trans(w), accumulator)
        accumulator = accumulator * scale_factor
        tl.store(out_ptr + offs_i[:, None] * N + offs_j[None, :], accumulator.to(tl.bfloat16), mask=(offs_i[:, None] < M) & (offs_j[None, :] < N))
        pid += psize


@triton.jit
def softmax_kernel_mask0(
    inp_ptr,
    mask_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    num_heads: tl.constexpr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    assert BLOCK_SIZE >= queries, f'BLOCK_SIZE must be >= N, got {BLOCK_SIZE} < {queries}'

    masks: tl.constexpr = queries // num_heads

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=-float('inf'))
        # mask_val = tl.load(mask_ptr + offs_i * keys + offs_j, mask=attn_mask, other=0.0)
        offs_mask_i = offs_i // num_heads
        mask_val = tl.load(mask_ptr + offs_mask_i * keys + offs_j, mask=(offs_mask_i < masks) & (offs_j < keys), other=0.0)
        vals = tl.where(mask_val, vals, -float('inf'))
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        vals = vals.to(tl.bfloat16)
        tl.store(out_ptr + offs_i * keys + offs_j, vals, mask=attn_mask)


def matmul_k8_256_n_softmax_mask0(Q, K, out, mask):
    total_queries = Q.shape[0]
    total_keys = K.shape[0]
    num_heads = 8
    head_dim = 256
    matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
        Q, K, out, total_queries, total_keys, head_dim, head_dim**-0.5, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64
    )
    softmax_kernel_mask0[((total_queries + 3) // 4,)](out, mask, total_queries, total_keys, num_heads, out, BLOCK_SIZE_M=4, BLOCK_SIZE=1024)


def matmul_k8_n_256(x, V, out):
    total_queries = x.shape[0]
    total_keys = V.shape[0]
    head_dim = 256
    matmul_small[((total_keys + 31) // 32) * (head_dim // 32),](
        x, V, out, seq_len=total_queries, features=total_keys, hidden=head_dim, BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=64
    )


def matmul_k_2048_1024_res(x, weight, out):
    seq_len = x.shape[0]
    matmul_small_res[(128,)](x, weight, out, out, seq_len=seq_len, features=2048, hidden=1024, BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=128)


def rms_matmul_k_1024_4096_gate(x, weight1, weight2, out, x_norm_factor, bias1=None, bias2=None):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    if bias1 is None:
        assert bias2 is None
        scaled_matmul_small_gate[(128,)](
            x, x_norm_factor, weight1, weight2, out, seq_len=seq_len, features=1024, hidden=4096, BLOCK_SIZE_N=64, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64
        )
    else:
        assert bias2 is not None
        scaled_matmul_small_bias_gate[(128,)](
            x,
            x_norm_factor,
            weight1,
            bias1,
            weight2,
            bias2,
            out,
            seq_len=seq_len,
            features=1024,
            hidden=4096,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_K=64,
        )


def matmul_k_4096_1024_res(x, weight, out):
    seq_len = x.shape[0]
    matmul_small_res[(((seq_len + 15) // 16) * (1024 // 32),)](
        x, weight, out, out, seq_len=seq_len, features=4096, hidden=1024, BLOCK_SIZE_N=16, BLOCK_SIZE_M=32, BLOCK_SIZE_K=256
    )
