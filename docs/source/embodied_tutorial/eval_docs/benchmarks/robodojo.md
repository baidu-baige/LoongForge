# RoboDojo Evaluation

RoboDojo is an Isaac Sim 5.1 benchmark for the dual-arm ARX X5 (`arx_x5`): 42 tasks plus 12
`_random` layout variants, 54 configurations in total. The LoongForge eval module reuses
RoboDojo's official Isaac Sim client unchanged and plugs in through XPolicyLab, so success is
latched by the official reward modules and the evaluation protocol stays the benchmark's own.

Eight LoongForge models have a RoboDojo bridge; six of them were checked against the
benchmark's own published policies on the full 54-configuration sweep (see Verification).

## Step 0: Download weights

The benchmark publishes its reference checkpoints under
[RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
(`ckpt/RoboDojo/<policy>/`). Each bridge below names the policy directory it expects.

| `action_bridge` | Model | Reference checkpoint |
|---|---|---|
| `lingbot_va_joint30` | LingBot-VA | `Lingbot_VA/robodojo_sim_arx_x5_v21/checkpoint_step_5000` |
| `pi05_joint14` | pi0.5 | `Pi_05/` (openpi / orbax, convert to PyTorch first) |
| `pi05_sf_joint14` | Spatial_Forcing | `Spatial_Forcing/` (pi0.5 + a training-only VGGT alignment loss) |
| `xvla_ee20` | X-VLA | `X_VLA/` |
| `fastwam_joint14` | FastWAM | `FastWAM/` |
| `gr00t_n17_joint14` | GR00T-N1.7 | `GR00T_N17/` |
| `gr00t_n16_joint14` | GR00T-N1.6 | no official RoboDojo checkpoint — code parity only |
| `dreamzero_joint14` | DreamZero-AgiBot | public DreamZero-AgiBot weights — code parity only |

## Step 1: Environment setup

Install RoboDojo and its Isaac Sim 5.1 simulator following the official repository, then
point the runner at the checkout with `env.robodojo_root` (or `ROBODOJO_ROOT`).

Two Python environments are needed, and they are *not* interchangeable:

- **Sim env** (`env.robodojo_python`) — the Isaac Sim environment that runs
  `scripts/eval_policy.sh`. RoboDojo pins `websockets 12`.
- **Policy env** (`env.policy_python`, by default the LoongForge server's Python) — hosts
  the XPolicyLab policy server with the LoongForge bridge. XPolicyLab's server needs
  `websockets>=14` (`websockets.asyncio`), which the sim env cannot provide, so the runner
  starts the bridge in this second environment.

⚠️ Common issues:

- **Headless rendering.** The client renders through Vulkan; check the NVIDIA ICD before the
  run (`VK_ICD_FILENAMES`), and keep Isaac Sim's shader cache warm — a cold cache adds
  minutes to every process start.
- **Crash dialogs.** After a GPU or PhysX crash Isaac Sim's crash reporter keeps trying to
  open a `zenity` dialog and the client never exits. The runner watches for that and stops a
  client that makes no step progress for 240 s afterwards.
- **Proxies.** The runner strips `http_proxy` / `https_proxy` from both child environments:
  the bridge talks to `127.0.0.1` and a proxy breaks the WebSocket upgrade.

## Step 2: Run evaluation

Run from the LoongForge environment; the runner starts the sim client itself.

```bash
cd /path/to/LoongForge
examples/embodied/lingbot_va/eval/run_robodojo_eval.sh    # CONFIG=... to pick a config
```

Example configs (fill the `/path/to/...` placeholders first):

| Model | Config |
|---|---|
| LingBot-VA | `examples/embodied/lingbot_va/eval/configs/robodojo/stack_bowls_smoke.yaml`, `robodojo_all_tasks.yaml` |
| pi0.5 | `examples/embodied/pi05/eval/configs/robodojo/stack_bowls.yaml` |
| Spatial_Forcing | `examples/embodied/pi05/eval/configs/robodojo/stack_bowls_spatial_forcing.yaml` |
| X-VLA | `examples/embodied/xvla/eval/configs/robodojo/put_bottles_into_dustbin.yaml` |
| FastWAM | `examples/embodied/fastwam/eval/configs/robodojo/put_bottles_into_dustbin.yaml` |
| GR00T-N1.7 / N1.6 | `examples/embodied/gr00t_n1_7/eval/configs/robodojo/put_bottles_into_dustbin.yaml`, `gr00t_n1_6/...` |
| DreamZero | `examples/embodied/dreamzero/eval/configs/robodojo/put_bottles_into_dustbin.yaml` |

Key config fields:

| Key | Meaning |
|---|---|
| `benchmark.task_name` | RoboDojo task, e.g. `stack_bowls`; `benchmark.seed` selects the layout set (`Assets/Eval_Layout/<env>/<seed>`) |
| `benchmark.action_bridge` | Selects the per-model protocol (table above) |
| `benchmark.first_chunk_len` / `chunk_len` | The model's chunk schedule; the bridge flags the last action of each chunk so the plugin's loop matches the official one. LingBot-VA derives them from `model.frame_chunk_size` / `action_per_frame` |
| `benchmark.eval_num` | Cap on episodes (default: the task's native count) |
| `benchmark.eval_batch` / `num_envs` | RoboDojo's batched mode: several envs in one Isaac Sim process (below) |
| `env.robodojo_root` / `robodojo_python` / `policy_python` | The checkout and the two environments from Step 1 |

### Several envs per Isaac Sim process (`eval_batch`)

RoboDojo's own batched mode runs `num_envs` envs in one Isaac Sim process and calls
`eval_one_episode_batch`. Set `benchmark.eval_batch: true` and `benchmark.num_envs: <k>`; the
runner then links the second plugin (`LoongForgeBatch`, whose static `deploy.yml` is where
RoboDojo reads `eval_batch`) and writes a copy of `scripts/eval_policy.sh` with the env count
into the run directory — the official script takes `num_envs` from the sim config and drops
unknown flags, so it cannot be passed through. Exactly two lines of the copy differ from the
official script, both matched verbatim, so an upstream change fails the run instead of
silently evaluating something else.

The bridge keeps one episode id, PayloadBuilder, ActionDecoder and trace file per env, and
asks the server for a whole chunk per env per chunk (the official batched loop executes it
step by step). The per-step protocol of one env per process is unchanged.

Memory, not throughput, is the limit: with the ARX X5 scenes one Isaac Sim process costs
about 10 GB plus the policy server's own footprint, so a 96 GB card fits roughly
2 processes × 10 envs. An env without an episode still simulates and renders (its seed is
`None`, it is only marked done), so the runner never creates more envs than the run has
episodes.

## Verification

Full sweep, 54 configurations × seed 0 (2,100 episodes), LoongForge versus the benchmark's own
policy run on the same layouts. The CI is the 95% interval for the difference of two
independent proportions.

| Model | LoongForge | Official | Difference |
|---|---|---|---|
| Spatial_Forcing | 9.29% | 9.38% | −0.10 pp (±1.76) |
| pi0.5 | 8.43% | 9.19% | −0.76 pp (±1.71) |
| X-VLA | 7.05% | 6.95% | +0.10 pp |
| FastWAM | 1.43% | 1.33% | +0.10 pp (±0.71) |
| GR00T-N1.7 | 0.76% | 1.00% | −0.24 pp (±0.57) |
| LingBot-VA | 0.00% (0/2100) | 0.05% (1/2100) | −0.05 pp (±0.09) |

Every difference falls inside its CI, and no per-configuration difference is significant.
LingBot-VA's published SFT checkpoint is near zero on RoboDojo, so success rate carries no
signal there; the stronger evidence is per-episode agreement — 2,099 of 2,100 episodes end
with the same success flag as the official run.

GR00T-N1.6 and DreamZero have no official RoboDojo policy to compare against (no checkpoint
and no plugin): they are verified by a smoke run plus unit tests that check the payload and
the call sequence against the official source of the sibling policy.

> **Reproducibility note.** RoboDojo is not bit-reproducible: two runs of the same code and
> seed diverge from the first chunk (max |Δaction| ~3e-3 at step 0), so per-episode regression
> against another run is not an equivalence test. Compare aggregate success rates, and use
> offline action parity (same observations, same seed) when a change should be numerically
> neutral.

## Outputs

Besides the standard outputs (user guide §4), the runner copies the official `_result.json`
into `artifacts/robodojo/<task>/seed<seed>/`, records the official result directory and the
video paths, and writes one `results.jsonl` row per episode carrying RoboDojo's
partial-progress `score`. Videos stay where the official client wrote them
(`<robodojo_root>/eval_result/...`) — they are the bulk of the disk cost, so prune them
between sweeps. With `run.trace: true` the bridge also writes a per-episode action trace
(`trace/ep<NNNN>.jsonl`, or `trace/b<batch>e<env>.jsonl` under `eval_batch`), which is what
the offline parity check reads.

A client that exits 0 without its `(eval completed)` line counts as failed: the official
client also exits 0 when it fails while building the environment.
