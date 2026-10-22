import torch
import triton
import triton.language as tl

from .shared_triton_utils import (
    conv2d_embed_n256_1152_res,
    layer_norm_matmul_n256_1152_4304_bias_gelu,
    layer_norm_QKV_matmul_n256_1152_3456_bias,
    layer_norm_small_kernel,
    matmul_k_32_1024_bias,
    matmul_n256_1152_1152_bias_res,
    matmul_n256_4304_1152_bias_res,
    matmul_small,
    matmul_small_bias,
    matmul_small_gate,
    rms_matmul_k_1024_32_bias_res,
    rmsnorm_factor_kernel,
    scaled_matmul_small_bias_gate,
    scaled_matmul_small_gate,
)

# Re-export kernels that are common and unchanged
__all__ = [
    'conv2d_embed_n256_1152_res',
    'layer_norm_QKV_matmul_n256_1152_3456_bias',
    'matmul_n256_1152_1152_bias_res',
    'layer_norm_matmul_n256_1152_4304_bias_gelu',
    'matmul_n256_4304_1152_bias_res',
    'layer_norm_matmul_n256_1152_2304_bias',
    'rms_matmul_n_2304_4096_qkv_rope',
    'matmul_n_2304_4096_qkv_rope',
    'rms_matmul_n_2304_9216_gate',
    'matmul_n_2048_2304',
    'matmul_n_9216_2304',
    'matmul_k_32_1024_bias',
    'rms_matmul_k_1024_4096_qkv_rope',
    'matmul_k_2048_1024',
    'rms_matmul_k_1024_2048_gate',
    'rms_matmul_k_1024_32_bias_res',
    'AttnMultiKey',
    'AttnGroupKeyMask',
]


@torch.compile
def AttnMultiKey(QKV):
    # QKV shape: (B*N, 256, 3*1152) -> (B*N, 256, 3, 16, 72)
    # But here we are using SigLip which has 16 heads, head_dim=72?
    # SigLipVisionConfig: num_attention_heads=16, hidden_size=1152 => head_dim=72.
    # This matches Pi0 implementation.
    QKV = QKV.view(-1, 256, 3, 16, 72).permute(0, 2, 3, 1, 4)
    Q = QKV[:, 0]
    K = QKV[:, 1]
    V = QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(Q.shape[0], 256, 1152)
    return attn


@torch.compile
def AttnGroupKeyMask(Q, K, V, mask=None):
    Q = Q.view(1, -1, 8, 256).transpose(1, 2)
    K = K.view(1, -1, 4, 256).transpose(1, 2)
    V = V.view(1, -1, 4, 256).transpose(1, 2)
    if mask is not None:
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask[None, None], is_causal=False, enable_gqa=True)
    else:
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True)
    attn = attn.transpose(1, 2).reshape(-1, 2048)
    return attn


@triton.jit
def rms_norm_kernel(inp_ptr, out_ptr, seq_len: tl.constexpr, features: tl.constexpr):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    BLOCK_SIZE: tl.constexpr = 256
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


def layer_norm_matmul_n256_1152_2304_bias(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    # For PaliGemma2 Projector: 1152 -> 2304
    seq_len = x.shape[0] * 256
    layer_norm_small_kernel[seq_len,](x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152, eps=1e-5)
    matmul_small_bias[((seq_len + 63) // 64) * (2304 // 64),](
        x_norm, proj_w, out, proj_b, seq_len=seq_len, features=1152, hidden=2304, BLOCK_SIZE_N=64, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64
    )


@triton.jit
def scaled_matmul_rope_qkv_bias(
    inp_ptr,
    inp_norm_factor_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
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

    output_features = (num_heads + 2 * num_kv_heads) * head_dim
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

        q_end = num_heads * head_dim
        k_end = q_end + num_kv_heads * head_dim

        if start_j < k_end:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < q_end:
            out_ptr_curr = q_ptr
            out_stride = num_heads * head_dim
            out_offset = start_j
        elif start_j < k_end:
            out_ptr_curr = k_ptr
            out_stride = num_kv_heads * head_dim
            out_offset = start_j - q_end
        else:
            out_ptr_curr = v_ptr
            out_stride = num_kv_heads * head_dim
            out_offset = start_j - k_end

        tl.store(
            out_ptr_curr + offs_i * out_stride + out_offset % out_stride + tl.arange(0, BLOCK_SIZE_N),
            accumulator,
            mask=(offs_i < seq_len) & (offs_j < output_features),
        )
        pid += psize


def rms_matmul_n_2304_4096_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm):
    # PaliGemma2: Hidden 2304.
    # Heads 8, KV Heads 4. Head Dim 256.
    # Out = (8+4+4)*256 = 16*256 = 4096.
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2304)

    matmul_n_2304_4096_qkv_rope[((seq_len + 63) // 64, (4096 + 63) // 64)](x_norm, weight_qkv, rope_weight, Q, K, V, seq_len, 2304, 256, 8, 4)


def rms_matmul_k_1024_4096_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm_factor, bias_qkv=None):
    # Expert: Hidden 1024.
    # Heads 8, KV Heads 4. Head Dim 256.
    # Out = 4096.
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)

    scaled_matmul_rope_qkv_bias[(128,)](x, x_norm_factor, seq_len, 1024, 256, 8, 4, weight_qkv, bias_qkv, rope_weight, Q, K, V)


def rms_matmul_n_2304_9216_gate(x, weight1, weight2, out, x_norm):
    # PaliGemma2 MLP: Hidden 2304 -> Intermediate 9216.
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2304)
    matmul_small_gate[((seq_len + 127) // 128, (9216 + 63) // 64)](x_norm, weight1, weight2, out, seq_len, 2304, 9216)


def rms_matmul_k_1024_2048_gate(x, weight1, weight2, out, x_norm_factor, bias1=None, bias2=None):
    # Expert MLP: 1024 -> 2048 (Interm).
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    if bias1 is None:
        scaled_matmul_small_gate[(128,)](
            x, x_norm_factor, weight1, weight2, out, seq_len=seq_len, features=1024, hidden=2048, BLOCK_SIZE_N=64, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64
        )
    else:
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
            hidden=2048,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_K=64,
        )


def matmul_k_2048_1024(x, weight, out):
    # Expert MLP Down / Attn Proj: 2048 -> 1024. No residual.
    seq_len = x.shape[0]
    matmul_small[(((seq_len + 15) // 16) * (1024 // 32),)](
        x, weight, out, seq_len=seq_len, features=2048, hidden=1024, BLOCK_SIZE_N=16, BLOCK_SIZE_M=32, BLOCK_SIZE_K=256
    )


def matmul_n_2048_2304(x, weight, out):
    # Encoder Output Proj: 2048 -> 2304
    seq_len = x.shape[0]
    # Grid: (seq_len // 32) * (2304 // 64)
    matmul_small[((seq_len + 31) // 32) * ((2304 + 63) // 64),](
        x,
        weight,
        out,
        seq_len,
        2048,
        2304,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
    )


def matmul_n_9216_2304(x, weight, out):
    # Encoder FFN Down: 9216 -> 2304
    seq_len = x.shape[0]
    matmul_small[((seq_len + 31) // 32) * ((2304 + 63) // 64),](
        x,
        weight,
        out,
        seq_len,
        9216,
        2304,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,  # K is large
    )


@triton.jit
def matmul_n_2304_4096_qkv_rope(
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
    num_kv_head: tl.constexpr,
):
    BLOCK_SIZE_N: tl.constexpr = 64
    BLOCK_SIZE_M: tl.constexpr = 64
    BLOCK_SIZE_K: tl.constexpr = 64
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)

    total_dim = (num_head + 2 * num_kv_head) * head_dim

    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, total_dim, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                    mask=tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i,
                    other=0.0,
                )
                w = tl.load(weight_QKV_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * total_dim + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)

            if j < (num_head + num_kv_head) * head_dim:
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
            elif j < (num_head + num_kv_head) * head_dim:
                out_ptr = K_ptr
                out_stride = num_kv_head * head_dim
            else:
                out_ptr = V_ptr
                out_stride = num_kv_head * head_dim

            tl.store(
                out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * out_stride + j % out_stride + tl.arange(0, BLOCK_SIZE_M),
                acc,
                mask=tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i,
            )
