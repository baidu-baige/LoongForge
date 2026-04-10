import torch

from flash_mla_fwd import flash_mla_sparse_fwd

def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, d_v=512, q_start_index_s=0):
    q = q.float()
    kv = kv.float()

    b, sq, h, dim_q = q.shape
    _, skv, g, dim_kv = kv.shape
    topk = indices.shape[-1]
    dim_v = d_v

    assert h % g == 0, "h_q must be divisible by h_kv"
    assert dim_kv >= dim_v, "kv head dim must be >= value dim"

    grouped_heads = h // g
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale

    q = q.view(b, sq, g, grouped_heads, dim_q)

    safe_indices = indices.long().clamp(min=0, max=max(skv - 1, 0))
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1, 1)
    group_idx = torch.arange(g, device=q.device).view(1, 1, g, 1)
    gathered_kv = kv[batch_idx, safe_indices, group_idx]
    gathered_v = gathered_kv[..., :dim_v]

    qk = torch.einsum("bsghd,bsgtd->bghst", q, gathered_kv)

    causal_limit = torch.arange(q_start_index_s, q_start_index_s + sq, dtype=indices.dtype, device=indices.device).view(1, sq, 1, 1)
    valid_mask = (indices >= 0) & (indices < skv) & (indices <= causal_limit)
    valid_mask = valid_mask.permute(0, 2, 1, 3).unsqueeze(2)

    p_out = qk.masked_fill(~valid_mask, float("-inf"))
    score = qk.mul(sm_scale).masked_fill(~valid_mask, float("-inf"))
    max_logits = score.max(dim=-1).values
    lse = torch.logsumexp(score, dim=-1)

    lonely_q_mask = lse == float("-inf")
    lse_for_prob = lse.clone()
    lse_for_prob[lonely_q_mask] = float("+inf")
    p = torch.exp(score - lse_for_prob.unsqueeze(-1))

    o = torch.einsum("bghst,bsgtd->bsghd", p.type(gathered_v.dtype), gathered_v)
    o = o.reshape(b, sq, h, dim_v)
    p_out = p_out.permute(0, 3, 1, 2, 4).reshape(b, sq, h, topk)

    lse[lonely_q_mask] = float("+inf")
    max_logits = max_logits.permute(0, 3, 1, 2).reshape(b, sq, h)
    lse = lse.permute(0, 3, 1, 2).reshape(b, sq, h)

    return o.to(torch.bfloat16), max_logits, lse, p_out


def test_sparse_mla_fwd(
    B=1,
    S=4096,
    SKV=8192,
    H=64,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=2048,
    sm_scale=None,
    dtype=torch.bfloat16,
    check_correctness=True,
    q_start_index_s=0,
    write_p_out=False,
):
    sm_scale = DQKV ** -0.5 if sm_scale is None else sm_scale

    q = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                max_kv_i = min(SKV, max(1, q_start_index_s + t))
                i_i = torch.randperm(max_kv_i)[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    q_3d = q.squeeze(0).contiguous()
    kv_3d = kv.squeeze(0).contiguous()
    indices_3d = indices.squeeze(0).contiguous()

    flash_out, flash_max_logits, flash_lse, flash_p_out = flash_mla_sparse_fwd(
        q_3d,
        kv_3d,
        indices_3d,
        sm_scale=sm_scale,
        d_v=DV,
        q_start_index_s=q_start_index_s,
        write_p_out=write_p_out,
    )
    torch.cuda.synchronize()

    if check_correctness:
        ref_out, ref_max_logits, ref_lse, ref_p_out = ref_sparse_mla_fwd_interface(
            q, kv, indices, sm_scale=sm_scale, d_v=DV, q_start_index_s=q_start_index_s
        )
        ref_out_3d = ref_out.squeeze(0)
        ref_max_logits_3d = ref_max_logits.squeeze(0)
        ref_lse_3d = ref_lse.squeeze(0)

        flash_out_max_diff, flash_out_rel_diff = calc_diff(flash_out.float(), ref_out_3d.float())
        flash_max_logits_max_diff, flash_max_logits_rel_diff = calc_diff(flash_max_logits, ref_max_logits_3d)
        flash_lse_max_diff, flash_lse_rel_diff = calc_diff(flash_lse, ref_lse_3d)

        print(f"[ref vs flash] out        max_diff={flash_out_max_diff:.6f}, rel_diff={flash_out_rel_diff:.6f}")
        print(
            f"[ref vs flash] max_logits max_diff={flash_max_logits_max_diff:.6f}, rel_diff={flash_max_logits_rel_diff:.6f}"
        )
        print(f"[ref vs flash] lse        max_diff={flash_lse_max_diff:.6f}, rel_diff={flash_lse_rel_diff:.6f}")

        if write_p_out:
            ref_p_out_3d = ref_p_out.squeeze(0)
            flash_p_out_max_diff, flash_p_out_rel_diff = calc_diff(flash_p_out, ref_p_out_3d)
            print(f"[ref vs flash] p_out      max_diff={flash_p_out_max_diff:.6f}, rel_diff={flash_p_out_rel_diff:.6f}")

    per_token_flop = 2 * H * topk * (DQKV + DV)

    # Benchmark using torch.cuda.Event
    warmup = 10
    num_tests = 100
    for _ in range(warmup):
        flash_mla_sparse_fwd(
            q_3d,
            kv_3d,
            indices_3d,
            sm_scale=sm_scale,
            d_v=DV,
            q_start_index_s=q_start_index_s,
            write_p_out=write_p_out,
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_tests):
        flash_mla_sparse_fwd(
            q_3d,
            kv_3d,
            indices_3d,
            sm_scale=sm_scale,
            d_v=DV,
            q_start_index_s=q_start_index_s,
            write_p_out=write_p_out,
        )
    end_event.record()
    torch.cuda.synchronize()

    ms = start_event.elapsed_time(end_event) / num_tests
    print(f"Average time: {ms:.3f} ms")
    print(f"fwd io bandwidth = ", (B * S * (topk * DQKV + H * (DQKV + DV)) * 2) / (ms * 1e-3) / 1e12)
    print(f"fwd tflops = ", per_token_flop * B * S / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1,
        S=4096,
        SKV=8192,
        H=128,
        HKV=1,
        DQKV=576,
        DV=512,
        topk=2048,
        sm_scale=576 ** -0.5,
        dtype=torch.bfloat16,
        check_correctness=True,
        q_start_index_s=2048,
        write_p_out=True,
    )
