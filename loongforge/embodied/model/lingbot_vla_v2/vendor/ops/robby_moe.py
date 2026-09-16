"""Vendor module implementation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _zero_i32_kernel(out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Vendor interface."""
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.int32), mask=offs < N)


@triton.jit
def _zero_fp32_kernel(out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Vendor interface."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < N)


@triton.jit
def _moe_pack_selected_kernel(
    selected_ptr,
    route_ptr,
    counts_ptr,
    rows_ptr,
    slots_ptr,
    T: tl.constexpr,
    TOPK: tl.constexpr,
    MAX_ROUTES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Vendor interface."""
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_K)
    mask = slots < TOPK
    experts = tl.load(selected_ptr + row * TOPK + slots, mask=mask, other=0).to(tl.int32)
    routes = tl.load(route_ptr + row * TOPK + slots, mask=mask, other=0.0).to(tl.float32)
    pos = tl.atomic_add(counts_ptr + experts, 1, sem="relaxed", mask=mask)
    store_mask = mask & (pos < MAX_ROUTES)
    tl.store(rows_ptr + experts * MAX_ROUTES + pos, row, mask=store_mask)
    tl.store(slots_ptr + experts * MAX_ROUTES + pos, slots, mask=store_mask)


@triton.jit
def _moe_gate_up_grouped_kernel(
    x_ptr,
    gate_ptr,
    up_ptr,
    counts_ptr,
    rows_ptr,
    slots_ptr,
    route_ptr,
    inter_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    I: tl.constexpr,
    MAX_ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Vendor interface."""
    expert = tl.program_id(0)
    bid_m = tl.program_id(1)
    bid_i = tl.program_id(2)
    count = tl.load(counts_ptr + expert).to(tl.int32)
    start_m = bid_m * BLOCK_M
    if start_m >= count:
        return
    route_idx = start_m + tl.arange(0, BLOCK_M)
    offs_i = bid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_d = tl.arange(0, BLOCK_D)
    valid_m = route_idx < count
    rows = tl.load(rows_ptr + expert * MAX_ROUTES + route_idx, mask=valid_m, other=0).to(tl.int32)
    slots = tl.load(slots_ptr + expert * MAX_ROUTES + route_idx, mask=valid_m, other=0).to(tl.int32)
    acc_g = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        ds = d0 + offs_d
        x = tl.load(
            x_ptr + rows[:, None] * D + ds[None, :],
            mask=valid_m[:, None] & (ds[None, :] < D),
            other=0.0,
        )
        gw = tl.load(
            gate_ptr + (expert * I + offs_i[None, :]) * D + ds[:, None],
            mask=(offs_i[None, :] < I) & (ds[:, None] < D),
            other=0.0,
        )
        uw = tl.load(
            up_ptr + (expert * I + offs_i[None, :]) * D + ds[:, None],
            mask=(offs_i[None, :] < I) & (ds[:, None] < D),
            other=0.0,
        )
        acc_g += tl.dot(x, gw)
        acc_u += tl.dot(x, uw)
    route = tl.load(route_ptr + rows * TOPK + slots, mask=valid_m, other=0.0).to(tl.float32)
    silu = acc_g * (1.0 / (1.0 + tl.exp(-acc_g)))
    val = silu * acc_u * route[:, None]
    tl.store(
        inter_ptr + ((rows[:, None] * TOPK + slots[:, None]) * I + offs_i[None, :]),
        val.to(inter_ptr.dtype.element_ty),
        mask=valid_m[:, None] & (offs_i[None, :] < I),
    )


@triton.jit
def _moe_down_grouped_kernel(
    inter_ptr,
    down_ptr,
    counts_ptr,
    rows_ptr,
    slots_ptr,
    out_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    I: tl.constexpr,
    MAX_ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Vendor interface."""
    expert = tl.program_id(0)
    bid_m = tl.program_id(1)
    bid_d = tl.program_id(2)
    count = tl.load(counts_ptr + expert).to(tl.int32)
    start_m = bid_m * BLOCK_M
    if start_m >= count:
        return
    route_idx = start_m + tl.arange(0, BLOCK_M)
    offs_d = bid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_i = tl.arange(0, BLOCK_I)
    valid_m = route_idx < count
    rows = tl.load(rows_ptr + expert * MAX_ROUTES + route_idx, mask=valid_m, other=0).to(tl.int32)
    slots = tl.load(slots_ptr + expert * MAX_ROUTES + route_idx, mask=valid_m, other=0).to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        is_ = i0 + offs_i
        x = tl.load(
            inter_ptr + ((rows[:, None] * TOPK + slots[:, None]) * I + is_[None, :]),
            mask=valid_m[:, None] & (is_[None, :] < I),
            other=0.0,
        )
        w = tl.load(
            down_ptr + (expert * D + offs_d[None, :]) * I + is_[:, None],
            mask=(offs_d[None, :] < D) & (is_[:, None] < I),
            other=0.0,
        )
        acc += tl.dot(x, w)
    tl.atomic_add(
        out_ptr + rows[:, None] * D + offs_d[None, :],
        acc,
        sem="relaxed",
        mask=valid_m[:, None] & (offs_d[None, :] < D),
    )


def robby_moe_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Inference-only grouped MoE path migrated from robbyvla_infer _moe."""
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be 2D, got {tuple(hidden_states.shape)}")
    if selected_experts.ndim != 2 or routing_weights.ndim != 2:
        raise ValueError("selected_experts and routing_weights must be 2D")
    if not hidden_states.is_cuda:
        raise ValueError("robby_moe_forward requires CUDA tensors")

    T, D = hidden_states.shape
    E, I, weight_d = gate_weight.shape
    top_k = selected_experts.shape[1]
    if weight_d != D or up_weight.shape != gate_weight.shape or down_weight.shape != (E, D, I):
        raise ValueError(
            "Unexpected MoE weight shapes: "
            f"hidden={tuple(hidden_states.shape)} gate={tuple(gate_weight.shape)} "
            f"up={tuple(up_weight.shape)} down={tuple(down_weight.shape)}"
        )

    max_routes = T * top_k
    if workspace is None:
        counts = torch.empty((E,), device=hidden_states.device, dtype=torch.int32)
        rows = torch.empty((E, max_routes), device=hidden_states.device, dtype=torch.int32)
        slots = torch.empty((E, max_routes), device=hidden_states.device, dtype=torch.int32)
        inter = torch.empty((T, top_k, I), device=hidden_states.device, dtype=hidden_states.dtype)
        out = torch.empty((T, D), device=hidden_states.device, dtype=torch.float32)
    else:
        counts = workspace["counts"]
        rows = workspace["rows"]
        slots = workspace["slots"]
        inter = workspace["inter"]
        out = workspace["out"]

    selected_i32 = selected_experts.to(torch.int32).contiguous()
    route = routing_weights.contiguous()

    _zero_i32_kernel[(1,)](counts, E, BLOCK=triton.next_power_of_2(E), num_warps=1)
    _moe_pack_selected_kernel[(T,)](
        selected_i32,
        route,
        counts,
        rows,
        slots,
        T,
        top_k,
        max_routes,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    _moe_gate_up_grouped_kernel[
        (E, triton.cdiv(max_routes, 16), triton.cdiv(I, 32))
    ](
        hidden_states,
        gate_weight,
        up_weight,
        counts,
        rows,
        slots,
        route,
        inter,
        T,
        D,
        E,
        top_k,
        I,
        max_routes,
        BLOCK_M=16,
        BLOCK_I=32,
        BLOCK_D=64,
        num_warps=4,
    )
    _zero_fp32_kernel[(triton.cdiv(out.numel(), 1024),)](
        out,
        out.numel(),
        BLOCK=1024,
        num_warps=4,
    )
    _moe_down_grouped_kernel[
        (E, triton.cdiv(max_routes, 16), triton.cdiv(D, 64))
    ](
        inter,
        down_weight,
        counts,
        rows,
        slots,
        out,
        T,
        D,
        E,
        top_k,
        I,
        max_routes,
        BLOCK_M=16,
        BLOCK_D=64,
        BLOCK_I=64,
        num_warps=4,
    )
    return out.reshape_as(hidden_states)
