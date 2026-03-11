import torch

from flash_mla_fwd import flash_mla_sparse_fwd
from flash_mla_bwd import flash_mla_sparse_bwd


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, q_start_index_s=0):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(q_start_index_s, q_start_index_s + sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True, q_start_index_s=0):
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_casual, q_start_index_s)
    o.backward(do)
    return q.grad, kv.grad


def test_sparse_mla_bwd(
    B=1,
    S=4096,
    SKV=8192,
    H=64,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    q_start_index_s=0,
):
    # Prepare data
    q = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                max_kv_i = min(SKV, max(1, q_start_index_s + t))
                i_i = torch.randperm(max_kv_i)[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    flash_out, _, flash_lse, _ = flash_mla_sparse_fwd(
        q.squeeze(0).contiguous(), kv.squeeze(0).contiguous(), indices.squeeze(0).contiguous(),
        sm_scale=576 ** -0.5, q_start_index_s=q_start_index_s
    )

    # flash_mla backward
    sm_scale = 1.0 / (DQKV ** 0.5)
    q_3d = q.squeeze(0)
    kv_3d = kv.squeeze(0)
    do_3d = do.squeeze(0)
    indices_3d = indices.squeeze(0)
    flash_dq, flash_dkv = flash_mla_sparse_bwd(
        q_3d, kv_3d, flash_out, do_3d, indices_3d, flash_lse,
        sm_scale=sm_scale,
        q_start_index_s=q_start_index_s,
    )
    torch.cuda.synchronize()

    if check_correctness:
        # Precision comparison: ref vs flash
        ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(
            q, kv, None, do, indices, None, q_start_index_s=q_start_index_s
        )
        ref_dq_3d = ref_dq.squeeze(0)
        ref_dkv_3d = ref_dkv.squeeze(0)
        flash_dq_max_diff, flash_dq_rel_diff = calc_diff(flash_dq, ref_dq_3d)
        flash_dkv_max_diff, flash_dkv_rel_diff = calc_diff(flash_dkv, ref_dkv_3d)
        print(f"[ref vs flash] dQ  max_diff={flash_dq_max_diff:.6f}, rel_diff={flash_dq_rel_diff:.6f}")
        print(f"[ref vs flash] dKV max_diff={flash_dkv_max_diff:.6f}, rel_diff={flash_dkv_rel_diff:.6f}")

    per_token_flop = 2 * sum(
        [
            H * DV * topk,
            H * DQKV * topk,
            H * DQKV * topk,
            H * DQKV * topk,
            H * DV * topk,
        ]
    )

    # Benchmark using torch.cuda.Event
    warmup = 10
    num_tests = 100
    for _ in range(warmup):
        flash_mla_sparse_bwd(
            q_3d, kv_3d, flash_out, do_3d, indices_3d, flash_lse,
            sm_scale=sm_scale,
            q_start_index_s=q_start_index_s,
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_tests):
        flash_mla_sparse_bwd(
            q_3d, kv_3d, flash_out, do_3d, indices_3d, flash_lse,
            sm_scale=sm_scale,
            q_start_index_s=q_start_index_s,
        )
    end_event.record()
    torch.cuda.synchronize()

    ms = start_event.elapsed_time(end_event) / num_tests
    print(f"Average time: {ms:.3f} ms")
    print(f"bwd io bandwidth = ", (B * S * max(DQKV * 2, DQKV + DV) * topk * 2) / (ms * 1e-3) / 1e12)
    print(f"bwd tflops = ", per_token_flop * S / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_mla_bwd(
        B=1,
        S=4096,
        SKV=8192,
        H=128,
        HKV=1,
        DQKV=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        q_start_index_s=2048,
    )