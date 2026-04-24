# ViT Encoder DP Load-Balancing
`vit_encoder_dp_balance` is a **lightweight, architecture-decoupled** engineering solution that targets the **DP-load-imbalance problem inside the Vision Encoder (ViT)** of Vision-Language Models.
It redistributes image tokens across DP ranks before the ViT forward pass so that every rank processes a similar workload, significantly improving GPU utilisation for **large-scale VLM training**.

---

## 1. Background & Problem
In **VLM (Vision-Language Model)** training, each sample may contain images/videos of vastly different resolutions. The Vision Encoder (ViT) processes these visual inputs **before** the LLM backbone.

The ViT's self-attention cost for a single image is quadratic in the number of visual tokens:

* load(image<sub>i</sub>) ∝ (T<sub>i</sub> × H<sub>i</sub> × W<sub>i</sub>)²

where **T**, **H**, **W** are the temporal, height, and width grid dimensions from \`image_grid_thw\`.

Because different DP ranks receive different images with different resolutions, ViT workloads across ranks can be **highly imbalanced** — even when the total LLM token counts per rank are balanced.

The lightly-loaded ranks must wait for the heavily-loaded ones at the synchronisation barrier after the ViT forward pass, creating **stragglers** that lower overall training throughput.

This problem differs from VLM-level DP load-balancing (\`--use-vlm-dp-balance\`), which reorders packed sequences for the LLM backbone. Here, the imbalance occurs **specifically in the ViT encoder stage**, and requires redistributing **visual tokens** rather than text tokens.

---

## 2. Solution Overview
The key idea is to **redistribute image tokens across DP ranks** according to their ViT compute cost before the ViT forward pass, and **reverse the redistribution** after the forward pass to restore the original data layout for downstream LLM processing.

The workflow has three phases:

1. **Gather & Plan** — Collect per-image token counts from all DP ranks, estimate per-image cost with a quadratic cost function, and solve a load-balanced assignment plan.
2. **Redistribute & Forward** — Redistribute \`pixel_values\` and \`image_grid_thw\` across ranks via \`all_to_all\`, then run the ViT forward pass on the balanced data.
3. **Reverse & Restore** — After the ViT forward, reverse-redistribute the output embeddings (including \`deepstack_pixel_embeds\` if present) back to their original DP ranks, and regenerate \`window_index\` if needed.

Key properties:

* **Decoupled from model architecture** — operates purely on ViT input/output tensors
* **No convergence impact** — only changes which rank computes which image; the mathematical result is identical
* **Supports gradient backpropagation** — uses \`torch.distributed.nn.functional.all_to_all\` for tensors that require gradients, ensuring correct backward pass through the redistribution
* **Automatic skip** — if the imbalance ratio is below 20% or a single image dominates the average load, redistribution is skipped to avoid unnecessary communication overhead
* **Independent micro-batch balancing** — unlike VLM-level balancing, ViT mode does not accumulate costs across micro-batches (\`cross_micro_batch_balance=False\`), since each micro-batch can be independently well-balanced without packing constraints

---

## 3. Usage
Add the flag in your training launcher:

\`\`\`bash
--use-vit-dp-balance
\`\`\`

This feature applies to **VLM models** using the \`OmniEncoderModel\` architecture (e.g., Qwen2-VL, Qwen3-VL, and other models with \`image_grid_thw\`-based ViT encoders).

It can be used **independently of or together with** VLM-level DP load-balancing (\`--use-vlm-dp-balance\`).

### Full CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| \`--use-vit-dp-balance\` | bool | \`False\` | Enable ViT encoder DP load balancing |
| \`--dp-balance-max-len-ratio-vit\` | float | \`None\` | Maximum sequence length ratio for ViT-level balancing. Limits each DP rank's max token count to (avg_len × ratio). \`None\` disables the constraint (default for ViT mode). |
| \`--dp-balance-trigger-threshold-vit\` | float | \`0.2\` | Minimum imbalance ratio threshold for triggering ViT-level balancing. Skips when imbalance ratio < threshold. |
| \`--dp-balance-verbose\` | bool | \`False\` | Print per-iteration diagnostics: imbalance ratio, per-DP load, reorder decisions |

---

## 4. Core Design

### 4.1 Entry Point & Integration

The ViT DP balance is invoked inside \`OmniEncoderModel.encode_images()\` via \`dp_balance_vit_encoder()\` (defined in \`vit_balance.py\`). The function wraps the ViT module call:

\`\`\`python
# Inside OmniEncoderModel.encode_images():
if args.use_vit_dp_balance:
    pixel_embeds, window_index, deepstack = dp_balance_vit_encoder(
        vit_module, pixel_values, image_grid_thw
    )
else:
    pixel_embeds, window_index, deepstack = vit_module(pixel_values, image_grid_thw)
\`\`\`

Unlike VLM-level balancing (which uses monkey-patching), ViT balancing is a direct function call — **no warm-up phase or cost model fitting is needed** because the ViT cost model is a simple quadratic that requires no calibration.

---

### 4.2 Cost Estimation
Each image's ViT compute cost is estimated as:

cost(image<sub>i</sub>) = num\_tokens<sub>i</sub>²

where num\_tokens<sub>i</sub> = T<sub>i</sub> × H<sub>i</sub> × W<sub>i</sub> from \`image_grid_thw\`.

This quadratic model reflects the self-attention complexity inside the ViT encoder. No warm-up profiling is needed — the quadratic assumption is a direct consequence of the ViT's architecture.

---

### 4.3 Load-Balanced Assignment
The solver (\`solve_sample_dp_reorder_plan\`) uses the same **greedy LPT (Longest Processing Time first)** algorithm as VLM-level balancing, but in **ViT mode** (no packing constraint):

1. **Sort** all images globally by cost in descending order
2. **Greedy assign** each image to the DP rank with the currently lowest total cost (no pack length constraint)
3. **Refine** by iteratively swapping/moving images between the most-loaded and least-loaded ranks until the imbalance falls below a tolerance threshold (default **5%**, up to **20** iterations)

**Skip conditions** — Redistribution is skipped when:
- **Imbalance ratio < threshold**: current distribution is already well-balanced (default threshold: **0.2**, controlled by \`--dp-balance-trigger-threshold-vit\`)
- **Single image dominates**: the highest single-image cost exceeds the average DP load, meaning no redistribution can meaningfully improve balance

The solver is shared with VLM-level balancing but configured differently:

| Parameter | VLM Mode | ViT Mode |
|-----------|----------|----------|
| \`cost_fn\` | Calibrated \`a·l²+b·l+c\` | Pure \`l²\` |
| \`pack_len_ratio\` | 1.2 (\`--dp-balance-max-len-ratio-vlm\`) | \`None\` (no constraint, \`--dp-balance-max-len-ratio-vit\`) |
| \`cross_micro_batch_balance\` | \`True\` | \`False\` |

---

### 4.4 Detailed Redistribution Flow

The redistribution is implemented via \`all_to_all\` communication in 6 steps:

**Step 1 — Compute per-image lengths and gather across DP:**
\`\`\`
vit_input_lengths[i] = T[i] * H[i] * W[i]  for each local image
→ gather_sample_info_across_dp() → global lengths, local indices, source ranks
\`\`\`

**Step 2 — Solve reorder plan:**
\`\`\`
solve_sample_dp_reorder_plan(cost_fn=λl: l², cross_micro_batch_balance=False)
→ plan[dst_rank] = [(local_idx, src_rank), ...] or None (skip)
\`\`\`

**Step 3 — Forward redistribution of \`pixel_values\`:**
\`\`\`
Split pixel_values by per-image lengths
→ redistribute_tensors() via all_to_all_single
→ Concatenate into balanced pixel_values
\`\`\`

**Step 4 — Forward redistribution of \`image_grid_thw\`:**
\`\`\`
Split image_grid_thw per row → redistribute → concatenate
\`\`\`

**Step 5 — ViT forward pass:**
\`\`\`
pixel_embeds, window_index, deepstack_pixel_embeds = vit_module(
    balanced_pixel_values, balanced_image_grid_thw
)
\`\`\`

**Step 6 — Reverse redistribution:**
\`\`\`
Compute reverse_reorder_plan from forward plan
→ Split pixel_embeds by per-image output lengths
→ redistribute back to original ranks via all_to_all_single
→ For each layer in deepstack_pixel_embeds:
    split by merged spatial dims → redistribute → concatenate
→ Regenerate window_index from original (un-reordered) image_grid_thw
\`\`\`

> **Note on \`deepstack_pixel_embeds\`**: When present (non-empty list), deep stack features have different spatial dimensions from \`pixel_embeds\` because the spatial merge reduces H and W by \`spatial_merge_size\`. The reverse redistribution handles this by computing separate feature lengths from the merged grid dimensions.

> **Note on \`window_index\`**: After reverse redistribution, \`window_index\` is regenerated from the **original** \`image_grid_thw\` (not the reordered one), since the downstream LLM processes images in their original DP assignment.

---

### 4.5 Gradient Support

For tensors that require gradients (e.g., \`pixel_embeds\` during backward), redistribution uses \`torch.distributed.nn.functional.all_to_all\` instead of \`dist.all_to_all_single\`. This ensures correct gradient flow through the redistribution operation, so that the ViT encoder receives proper gradients during backpropagation.

The implementation automatically detects \`requires_grad\` on the send tensor and selects the appropriate communication primitive.

---

### 4.6 Relationship to VLM DP Load-Balancing

| Feature | VLM DP Balance (\`--use-vlm-dp-balance\`) | ViT DP Balance (\`--use-vit-dp-balance\`) |
|---------|--------------------------------------|------------------------------------------|
| **Target** | LLM backbone (Attention + MLP) | ViT encoder |
| **Balancing unit** | Packed text sequences | Individual images/videos |
| **Cost model** | Warm-up profiled (a·l² + b·l + c) | Pure quadratic (l²) |
| **Warm-up required** | Yes (default 10 iterations) | No |
| **Constraint** | Pack length ratio ≤ 1.2× (\`--dp-balance-max-len-ratio-vlm\`) | None (\`--dp-balance-max-len-ratio-vit\`) |
| **Cross micro-batch** | Yes (accumulated costs) | No (independent per micro-batch) |
| **Integration** | Monkey-patch on pin_memory_loop | Direct call in encode_images() |
| **Timing** | After \`get_batch\`, before forward | Inside encoder forward |
| **Gradient support** | Not needed (before forward) | Yes (all_to_all with autograd) |
| **Trigger threshold** | \`--dp-balance-trigger-threshold-vlm\` (0.2) | \`--dp-balance-trigger-threshold-vit\` (0.2) |

Both features can be **enabled simultaneously** for maximum throughput improvement: ViT balance ensures even ViT compute, while VLM balance ensures even LLM compute.

---

## 5. Supported Models
The feature is integrated into \`OmniEncoderModel.encode_images()\` and supports all VLM models that use \`image_grid_thw\`-based ViT encoders, including:

* **Qwen2-VL** / **Qwen3-VL**
* Other models built on the \`OmniEncoderModel\` architecture

---

## 6. Quick Troubleshooting

| Symptom | Likely cause / fix |
|---------|-------------------|
| No visible speedup | Imbalance ratio < 0.2 → solver skips redistribution (expected behaviour). Check if images have similar resolutions. Use \`--dp-balance-verbose\` to inspect imbalance ratios. |
| OOM on some ranks after enabling | A rank received too many large images. This is unlikely with the LPT solver but can happen with extreme outliers. Consider combining with \`--use-vlm-dp-balance\`. |
| Shape mismatch after ViT forward | Ensure that the model returns \`(pixel_embeds, window_index, deepstack_pixel_embeds)\` from the ViT. Custom ViT architectures may need adaptation. |
| Redistribution applied but no speedup | Communication overhead may outweigh the balancing benefit. This typically happens with small DP sizes (< 8) or when all images are similar sizes. |
| \`--dp-balance-verbose\` shows frequent SKIP | Most batches are already well-balanced. This is normal and indicates that the feature is correctly avoiding unnecessary overhead. |
