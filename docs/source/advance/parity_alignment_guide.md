# Numerical Parity Alignment Guide

How to prove that a world model reimplemented on the five-layer architecture
under `loongforge/embodied/model/arch/` computes the same thing as the source
it replaces.

Three such proofs are in the tree, and this guide is written from them. Read
them alongside it:

| Harness | Source | Comparison | Status |
| --- | --- | --- | --- |
| `tests/embodied/unit/test_arch_vpp2_parity.py` | external checkout, `$VPP2_SOURCE_ROOT` | exact equality, CPU | 10 cases |
| `tests/embodied/unit/test_arch_internw0_delta_parity.py` | external checkout, `$INTERNW0_DELTA_SOURCE_ROOT` | exact equality, CPU | 12 cases, 1 skip |
| `tests/embodied/unit/test_arch_fastwam_parity.py` | in-repo, `loongforge/embodied/model/fastwam/` | `atol=1e-5`, CUDA | 12 cases |

Shared helpers live in `tests/embodied/unit/arch_parity.py`. It is deliberately
not a test module: each harness skips for its own reason, so a module-level
skip in one would take the others with it if they imported each other
(`arch_parity.py:6-9`).

## 1. What "parity" means here

**Two independent implementations, one set of weights.** Both sides run their
own full forward. Nothing is shared except the numbers in the parameters, which
travel from the source side to the new side through
`arch/factory/<model_name>/transfer.py`
(`test_arch_vpp2_parity.py:6-10`).

Two things follow, and both are load-bearing:

- **Nothing wraps or calls a source module inside the new code.** If the new
  side delegated even one step, a regression could hide behind the shim and the
  comparison would be testing the shim instead of the architecture.
- **The weight transfer is proved first, separately.** Every harness opens with
  `test_the_legacy_weights_load_with_nothing_missing_or_unexpected`
  (`test_arch_vpp2_parity.py:371-380`). Without it a parity failure cannot be
  told apart from a remap failure, and the two call for opposite fixes.
  `test_arch_transfer.py` pins the rule table against generated key names; this
  case pins it against the key names the real source modules produce.

Parity is not accuracy, convergence, or throughput. It answers one question:
given identical weights and identical input, do the two forwards produce the
same tensors?

## 2. Scope: what to compare

Four comparison points, chosen because a mismatch at each has a different
cause. Compare them separately even though the last one subsumes the others
numerically — an exit-only check tells you "somewhere upstream", which is the
least useful thing a failing test can say.

### Entry

`StreamModule.prepare` against the source's `pre_dit`. One call produces the
patchified tokens, the sinusoidal timestep embedding, the pre-combination
modulation, the RoPE table, and the projected per-slot context, so they are
asserted together with a separate name in each message
(`test_arch_fastwam_parity.py:495-516`).

Pin the entry details the spec is supposed to have absorbed. Three examples
from the tree:

- **The clean prefix.** `TokenTimestep(clean_prefix=N)` is one declaration
  replacing two pieces of source code that each contributed part of the zero
  block: the caller supplied `condition_frames - 1` leading zeros
  (`losses.py:624-634`) and `pre_dit` prepended one more
  (`wan_video_dit.py:826-836`). The case that pins the composition is
  `test_the_clean_prefix_covers_all_three_condition_frames`
  (`test_arch_internw0_delta_parity.py:497`); fastwam's one-frame version is
  `test_arch_fastwam_parity.py:521`.
- **The RoPE table.** Compared directly, and through whatever representation
  the source uses. fastwam splits it into cos/sin, so the harness reconstructs
  the split from the complex table
  (`test_arch_fastwam_parity.py:545-571`).
- **The key mask.** Every harness has
  `test_the_text_slot_mask_carries_the_padded_tail_and_the_proprio_token`,
  because a condition appended into a host slot changes that slot's valid
  length.

### Every joint layer

The source's per-layer tensors against the engine's `captures`, walked index by
index, reporting the **first** mismatch and how many layers differ
(`test_arch_vpp2_parity.py:477-503`).

Check that the two capture points coincide before trusting the index. For VPP2
the legacy callback fires right after the post-block (`mot.py:952-954`) and the
engine stores the hidden state at the same place (`engine.py:587-589`), so
layer *i* means the same thing on both sides and no offset correction is
needed. If they did not coincide, the correction belongs in the harness with a
comment, not in a tolerance.

### Engine exit

Both streams after `finalize`, worth asserting even with every layer already
compared: `finalize` runs after the last layer, and for this family it is
deliberately a no-op because the final norm sits in the head. A stray operation
there would show up only here (`test_arch_vpp2_parity.py:506-517`).

### Heads

The L4 heads against the source's head modules
(`test_arch_vpp2_parity.py:520-533`). The video head is the interesting half:
its modulation slots, its `x prod(patch_size)` exit width, and the unflatten
all have to line up, and the unflatten is the one step whose failure keeps the
shape right while stirring the time axis into the spatial axes. Assert the
final shape against `latent_channels` and `latent_grid`, not only the values.

## 3. Bit-exact vs tolerance: how to choose

**Default to exact equality.** When both sides issue the same primitive
operations in the same order, float32 agreement is bit-for-bit, and a tolerance
would let a genuine reordering through (`test_arch_vpp2_parity.py:28-30`). Two
of the three harnesses are exact.

`arch_parity.assert_identical` reports the largest deviation when it fails,
which is the number that tells a reordering (around `1e-7`) from a wrong tensor
(order 1) — `arch_parity.py:26-48`.

**Move to a tolerance only when a fused kernel forces it, and name the
kernel.** `arch_parity.assert_close` requires an explicit `atol` from every
caller for that reason (`arch_parity.py:56-79`). fastwam needs it twice over:
its RoPE is the FlashAttention Triton interleaved rotary with no eager
fallback (`wan/dit.py:27`, `:90-115`, `:132-148`), and the MoT path modulates
through `triton_norm_modulate`. Both compute the same expression in a different
order (`test_arch_fastwam_parity.py:14-22`).

Three rules for the tolerance itself:

1. **Measure it; do not guess it.** `_ATOL = 1e-5` carries a comment naming
   both sources of drift and the instruction to raise it only together with a
   note saying what moved (`test_arch_fastwam_parity.py:80-84`).
2. **Turn TF32 off.** TF32 keeps 10 mantissa bits, so every matmul in the stack
   would drift and the tolerance would have to be about a thousand times wider,
   at which point it stops meaning anything. The module-scoped autouse fixture
   `_real_float32` saves, clears, and restores both
   `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32`
   (`test_arch_fastwam_parity.py:430-447`).
3. **Test that the tolerance is still necessary.**
   `test_the_rotary_really_is_the_triton_kernel_with_no_fallback` asserts the
   rotary is an `autograd.Function` from `flash_attn` and that no eager
   fallback exists (`test_arch_fastwam_parity.py:478-490`). If the source ever
   grew one, or the harness stopped reaching the fused path, the right fix is to
   tighten every comparison to equality rather than keep a tolerance for no
   reason.

A mixed harness is fine. InternW0-Delta is exact everywhere except the one case
that folds a type embedding into a bias, where `W x + (b + e)` against
`(W x + b) + e` reassociates exactly one addition
(`test_arch_internw0_delta_parity.py:694-725`).

## 4. Capturing per-layer tensors

The new side is uniform: call `engine.run` with
`captures={name: tuple(range(num_layers))}` for every stream. Call the engine
directly rather than `WorldModel.forward`, because the harness needs its own
`[B]` timesteps and all layers, while `spec.captures()` requests only the layers
some head declared (`test_arch_vpp2_parity.py:248-273`).

The source side is where the three harnesses diverge. Look for a declarative
hook first; monkeypatch only when there is none.

### 4a. Declarative capture

InternW0-Delta's joint driver takes `capture_layers={"video": {...}, ...}` and
returns a 2-tuple instead of a tensor
(`mixture_of_transformers.py:730-739`), so the harness simply asks for both
streams (`test_arch_internw0_delta_parity.py:408-416`). VPP2 instead takes a
single `video_feature_callback=lambda index, hidden: ...`
(`test_arch_vpp2_parity.py:325-326`), which is why only its video stream is
walked layer by layer and the action stream is covered at the exit.

### 4b. Monkeypatching the call site

fastwam has no callback and no capture argument, and `nn.Module` forward hooks
do **not** fire, because `MoT` calls the blocks' pieces directly rather than
calling the blocks (`test_arch_fastwam_parity.py:173-178`).

The recipe, from `_capturing` (`test_arch_fastwam_parity.py:168-207`):

1. **Find the one place every expert passes through once per layer.** Here it is
   `_apply_post_with_optional_checkpoint` (`mot/model.py:653-663`).
2. **Build an identity map before the forward.** The call site receives
   `block=` as a keyword, so `id(block)` resolves the `(stream, layer)` pair:

   ```python
   index = {id(expert.blocks[layer]): (name, layer)
            for name, expert in legacy.mot.mixtures.items()
            for layer in range(len(expert.blocks))}
   ```

3. **Wrap on the instance, not the class.** Set the attribute on
   `legacy.mot`, and `del` it in a `finally` so the class method is uncovered
   again. Nothing is patched globally, so a failure cannot leak into another
   test.
4. **Call the real method, then file its output.** The wrapper must return the
   original return value unchanged:

   ```python
   def wrapper(**kwargs):
       tokens = original(**kwargs)
       name, layer = index[id(kwargs["block"])]
       captured[name][layer] = tokens
       return tokens
   ```

Before writing the wrapper, read the call site and confirm the argument is
passed by keyword and the return value is used. If either is untrue, `**kwargs`
and the pass-through return will silently break the forward.

## 5. Proving the harness is not a no-op

A parity harness is unusually easy to write so that it passes for the wrong
reason. Every input it feeds in that happens not to matter makes every
comparison above it free. Each harness therefore ends with a section headed
"the two inputs that would make all of the above vacuous"
(`test_arch_vpp2_parity.py:535`).

**The mask has to be exercised twice.**
`test_an_all_valid_text_mask_also_matches_and_gives_a_different_answer`
(`test_arch_vpp2_parity.py:538-555`) makes two claims in one extra forward:

- *also matches* rules out a shared bug that ignores `key_valid` on both sides,
  which would make every comparison pass with the mask doing nothing;
- *different answer* rules out the opposite reading of the same symptom — that
  this fixture's padded columns happen not to move the result, in which case
  the first claim would be free.

**The visibility matrix has to have structure.**
`test_the_joint_visibility_matrix_is_not_all_true`
(`test_arch_vpp2_parity.py:558-573`) asserts shape, `torch.bool`,
`not structural.all()`, and `structural.any()`. If it were all True, the parity
above would say nothing about visibility and the model's declared visibility
rule could be deleted without a single test noticing. This case is also where
the one shape conversion the harness must get right is recorded:
`MoT.forward` demands a 2-D square mask while `plan.materialize("dense")`
returns `[1, 1, S, S]`.

**Pass the new side's plan to the source.** The source gets
`model.plans[0].structural` rather than building its own
(`test_arch_internw0_delta_parity.py:386-389`), so a difference in the
comparison is a difference in the layers and not in which tokens may see which.

**Give a zero-initialized parameter a value before trusting a fold.** The
proprioception type embedding ships as `zeros`, which is why every other
comparison is exact. The one case that exercises the fold asserts
`not embedding.any()` first, then copies in random values, reloads, and makes
both halves of the claim: agreement to float32 rounding says the fold is the
right arithmetic, and inequality against the earlier run says the vector
reaches the output at all — without which the first half would hold for a
parameter that was being dropped
(`test_arch_internw0_delta_parity.py:694-727`).

The same principle governs the tools: `tests/embodied/unit/test_arch_style.py`
hands each source rule a snippet it must reject, because a rule that has
quietly stopped firing passes every file and looks exactly like a clean tree.

## 6. Recording what cannot be compared

Some parts of a source model have no numerical counterpart on the new side yet.
Leaving them out silently is the failure mode: six months later nobody can tell
an out-of-scope head from a forgotten one.

Write a case that measures the gap and then skips with the numbers in the skip
message. `test_the_distillation_heads_have_no_numerical_counterpart`
(`test_arch_internw0_delta_parity.py:730-769`) covers three heads with three
different reasons:

- **`future_delta` — a real disagreement.** The new side declares the per-frame
  width while the source flattens the frame axis into the projection. The case
  asserts both widths (`output_width == patch_dim` here,
  `patch_dim * frames` there) so the factor is recorded as a number rather than
  a claim, and names what is missing: `HeadSpec.output_width` has no frame axis,
  and the delta tokens have no declared position in `GeometrySpec`.
- **`semantic_future` — nothing left to compare.** The term is a cosine computed
  directly on two captured hidden states (`losses.py:58-71`), and both states
  are already compared layer by layer, so a head comparison would add nothing.
- **`track_align` — not buildable.** Its `track_query_head` raises
  `NotImplementedError`, which is also why `arch_parity.drop_heads` has to
  remove it before the model is constructed.

`drop_heads` (`arch_parity.py:82-103`) removes the head **and** the loss terms
that name it, because spec validation requires every `LossSpec.head` to exist;
dropping only the head yields a spec that fails validation rather than a
smaller model.

Record the environment gates the same way. An absent source checkout must
surface as a skip naming the path it looked for, not as a silent fall-through to
some other tree (`test_arch_vpp2_parity.py:61-79`). The pattern is
`$VPP2_SOURCE_ROOT` / `$INTERNW0_DELTA_SOURCE_ROOT` winning outright, a sibling
checkout as a convenience fallback, then `pytest.importorskip` per module with a
`reason` naming the line that needs it
(`test_arch_fastwam_parity.py:61-72`). CI never depends on an external tree.

## 7. Checklist for a new model

Work down the list. Each step is cheap to verify and expensive to skip, because
a failure at step *n* is nearly impossible to diagnose if step *n-1* was not
established first.

**Before writing any comparison**

1. Declare the provider under `arch/spec/providers/` and get
   `tests/embodied/unit/test_arch_providers.py` green.
2. Add the remap rules to `arch/factory/<model_name>/transfer.py` and pin them in
   `test_arch_transfer.py` against generated key names.
3. Run `arch_probe keys <model_type>` and diff its key list against the source
   checkpoint by eye. This is the step that turns a `strict=True` failure into
   a five-minute fix instead of an afternoon:

   ```bash
   PYTHONPATH=. python -m tools.embodied.arch_probe keys <model_type>
   ```

**The harness**

4. Locate the source: environment variable first, sibling checkout as
   fallback, `pytest.importorskip` with a reason per module.
5. Build both sides from one shrunken spec; load the source payload with
   `strict=True` and assert zero missing and zero unexpected keys.
6. Decide exact vs tolerance from section 3. If a tolerance, turn TF32 off and
   add the case that proves the fused kernel is still in the path.
7. Compare the entry. If it disagrees, run `arch_probe entry <model_type>`
   before reading any more code.
8. Compare every joint layer. Confirm the two capture points coincide first.
9. Compare the engine exit and every head that has a counterpart.

**Before calling it done**

10. Add the second-mask case (matches *and* differs).
11. Add the visibility-matrix case (not all True, not all False).
12. Write the skip case for everything out of scope, with the widths measured.
13. Run the full suite from the repository root:

    ```bash
    cd <repo-root>
    PYTHONPATH=. python -m pytest $(ls tests/embodied/unit/test_arch_*.py) -q
    ```

    Run it from the root or pytest collects nothing: the harnesses import
    `arch_parity` and `test_arch_blocks` as top-level modules.

**Related tools and pages**

- `tools/embodied/README.md` — the `arch_probe` subcommands and the three
  source rules enforced by `test_arch_style.py`.
- `advance/support_new_model` — registering a model family, config, and
  trainer.






