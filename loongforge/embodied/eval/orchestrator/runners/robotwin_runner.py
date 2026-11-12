# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Launch the official RoboTwin evaluator against a standalone vla_eval server."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

import yaml

from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.orchestrator.config import load_config

# Official eval_policy.py prints (ANSI-colored):
#   Success rate: {suc}/{test_num} => {pct}%, current seed: {now_seed}
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_SUCCESS_RATE_LINE_RE = re.compile(
    r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>.*?current seed:\s*(\d+)",
    re.IGNORECASE,
)


def _write_deploy_policy(args: argparse.Namespace, path: pathlib.Path) -> None:
    """Run _write_deploy_policy."""
    fields: Dict[str, object] = {
        "policy_name": "loongforge.embodied.eval.bridges.robotwin_policy",
        "policy_ckpt_path": args.policy_ckpt_path,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "ckpt_setting": args.ckpt_setting,
        "seed": args.seed,
        "instruction_type": args.instruction_type,
        "host": args.host,
        "port": args.port,
        "unnorm_key": args.unnorm_key,
        "action_mode": args.action_mode,
        "reorder_action": not args.no_action_reorder,
        "max_steps": args.max_steps,
        "control_hz": args.control_hz,
        "disable_action_cache": args.disable_action_cache,
        "return_action_chunk": args.return_action_chunk,
        "skip_intra_chunk_obs": bool(getattr(args, "skip_intra_chunk_obs", False)),
        "chunk_execute_steps": int(getattr(args, "chunk_execute_steps", 0) or 0),
        "action_bridge": getattr(args, "action_bridge", "ee6d_dual"),
        "domain_id": getattr(args, "domain_id", None),
    }
    if getattr(args, "write_trace", True):
        # Per-step tracing costs ~5% of the sim CPU (tolist + JSON) and is only a
        # debugging artifact; run.write_trace=false drops it from the deploy config.
        fields["trace_path"] = str(_robotwin_artifact_dir(args) / "trace.json")
    if args.disable_eval_video_log:
        fields["eval_video_log"] = False
    path.write_text(yaml.safe_dump(fields, allow_unicode=True, sort_keys=False), encoding="utf-8")


def fill_missing_task_attrs(
    env: Any,
    task_name: str,
    cached: Optional[Dict[str, Any]] = None,
    arm_tag_factory: Any = None,
    face_prod: Any = None,
) -> None:
    """Seed task state that ``play_once()`` normally sets for ``check_success()``.

    The official loop only runs the expert demo when ``expert_check`` is on. With
    the offline seed cache (or ``--disable-expert-check``) it never runs, so the
    few tasks that initialise ``arm_tag`` / ``origin_z`` inside ``play_once``
    would fail in ``check_success``. The values mirror those ``play_once()``
    lines exactly, so the RoboTwin source can stay untouched.

    ``cached`` carries the values the offline scan recorded from ``play_once``
    itself and is authoritative: the worker calls :func:`clear_episode_task_attrs`
    before every ``setup_demo``, so whatever is cached is applied. When the cache
    has nothing for the seed, the pose-based tasks are re-derived from the scene
    exactly like ``play_once`` would. ``arm_tag_factory`` / ``face_prod`` are
    injectable for tests; production calls resolve them from ``envs.utils``
    (importable in the sim worker).
    """
    if arm_tag_factory is None or face_prod is None:
        from envs.utils import ArmTag as _ArmTag, get_face_prod as _get_face_prod

        arm_tag_factory = arm_tag_factory or _ArmTag
        face_prod = face_prod or _get_face_prod
    cached = cached or {}
    if cached.get("arm_tag") is not None:
        env.arm_tag = arm_tag_factory(cached["arm_tag"])
    if cached.get("origin_z") is not None:
        env.origin_z = float(cached["origin_z"])
    obj = getattr(env, "object", None)
    if task_name == "open_laptop" and not hasattr(env, "arm_tag"):
        face = face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        env.arm_tag = arm_tag_factory("left" if face > 0 else "right")
    elif task_name == "place_object_scale" and obj is not None and not hasattr(env, "arm_tag"):
        env.arm_tag = arm_tag_factory("right" if obj.get_pose().p[0] > 0 else "left")
    elif task_name == "put_object_cabinet" and obj is not None:
        if not hasattr(env, "arm_tag"):
            env.arm_tag = arm_tag_factory("right" if obj.get_pose().p[0] > 0 else "left")
        if not hasattr(env, "origin_z"):
            env.origin_z = obj.get_pose().p[2]


def clear_episode_task_attrs(env: Any) -> None:
    """Drop ``play_once``-seeded attributes before the next episode's scene build.

    The official loop runs ``setup_demo`` on a *new* ``TASK_ENV`` attribute state
    only once; from episode 2 on it reuses the same object, so ``hasattr`` stays
    true and a stale ``arm_tag`` from the previous episode would be scored
    against this episode's scene. Clearing them makes every episode take the
    same path as the first one.
    """
    for name in ("arm_tag", "origin_z"):
        try:
            delattr(env, name)
        except AttributeError:
            pass


def _write_worker_eval_step_limit(args: argparse.Namespace) -> Optional[pathlib.Path]:
    """Run _write_worker_eval_step_limit."""
    if not args.worker_local_files or args.step_limit_override is None:
        return None
    robotwin_path = pathlib.Path(args.robotwin_path)
    source_dir = robotwin_path / "task_config"
    config_dir = pathlib.Path(args.output_dir) / "worker_files" / "task_config"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    shutil.copytree(source_dir, config_dir)
    worker_step_limit = config_dir / "_eval_step_limit.yml"
    data = yaml.safe_load(worker_step_limit.read_text(encoding="utf-8")) or {}
    data[args.task_name] = int(args.step_limit_override)
    worker_step_limit.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return worker_step_limit


def _worker_eval_script(args: argparse.Namespace) -> Optional[pathlib.Path]:
    """Run _worker_eval_script."""
    if not args.worker_local_files:
        return None
    robotwin_path = pathlib.Path(args.robotwin_path)
    source = robotwin_path / "script" / "eval_policy.py"
    text = source.read_text(encoding="utf-8")
    script_source_dir = str(source.parent)
    # Cap native threads per sim process BEFORE anything imports torch/cuRobo.
    # Each process otherwise spawns ~300+ threads (torch/OMP/MKL follow the
    # host core count); with 50-64 concurrent sims that is tens of thousands of
    # threads fighting over the CPU while every sim is single-lane anyway.
    sim_threads = getattr(args, "sim_cpu_threads", None)
    if sim_threads:
        _n = int(sim_threads)
        text = (
            "# --- sim CPU thread cap (injected) ---\n"
            "import os as _os\n"
            "for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',\n"
            "           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):\n"
            f"    _os.environ[_v] = '{_n}'\n"
            "import torch as _torch\n"
            f"_torch.set_num_threads({_n})\n"
            "try:\n"
            f"    _torch.set_num_interop_threads({_n})\n"
            "except RuntimeError:\n"
            "    pass\n"
            "# --- end sim CPU thread cap ---\n" + text
        )
    text = text.replace(
        'sys.path.append("./description/utils")\n',
        'sys.path.append("./description/utils")\n' f"sys.path.insert(0, {script_source_dir!r})\n",
        1,
    )
    worker_step_limit = _write_worker_eval_step_limit(args)
    if worker_step_limit is not None:
        worker_config_dir = str(worker_step_limit.parent)
        text = text.replace(
            "from envs import CONFIGS_PATH\n",
            "from envs import CONFIGS_PATH\n"
            "import envs\n"
            "import envs._GLOBAL_CONFIGS as _vla_eval_global_configs\n"
            f"CONFIGS_PATH = {worker_config_dir + '/'!r}\n"
            "envs.CONFIGS_PATH = CONFIGS_PATH\n"
            "_vla_eval_global_configs.CONFIGS_PATH = CONFIGS_PATH\n",
            1,
        )
        text = text.replace(
            'camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")',
            f'camera_config_path = os.path.join({worker_config_dir!r}, "_camera_config.yml")',
            1,
        )
    if args.start_seed_override is not None:
        text = text.replace(
            "    st_seed = 100000 * (1 + seed)\n", f"    st_seed = {int(args.start_seed_override)}\n", 1
        )
    if args.test_num_override is not None:
        text = text.replace("    test_num = 100\n", f"    test_num = {int(args.test_num_override)}\n", 1)
    if args.disable_expert_check:
        text = text.replace("    expert_check = True\n", "    expert_check = False\n", 1)
        text = text.replace(
            "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
            '        episode_info_list = [episode_info["info"]]\n',
            "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
            "        episode_info = locals().get('episode_info', {'info': {}})\n"
            '        episode_info_list = [episode_info["info"]]\n',
            1,
        )
        text = text.replace(
            "        instruction = np.random.choice(results[0][instruction_type])\n",
            "        if not results or not results[0].get(instruction_type):\n"
            "            instruction = args[\"task_name\"].replace('_', ' ')\n"
            "        else:\n"
            "            instruction = np.random.choice(results[0][instruction_type])\n",
            1,
        )
    if args.disable_eval_video_log:
        text = text.replace(
            '    args["ckpt_setting"] = ckpt_setting\n',
            '    args["ckpt_setting"] = ckpt_setting\n' '    args["eval_video_log"] = False\n',
            1,
        )
    # The official script only catches UnStableError inside the expert-check
    # branch. With --disable-expert-check, an unstable seed in the eval loop
    # aborts the whole run (and the coordinator would retry the same
    # deterministic seed). Skip the seed instead, mirroring the official branch.
    unstable_old = (
        "\n        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
    )
    unstable_new = (
        "\n        try:\n"
        "            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        "        except UnStableError:\n"
        "            TASK_ENV.close_env()\n"
        # succ_seed/test_num are incremented BEFORE setup_demo in the official
        # loop, so a skipped seed must undo that count or the loop exits early.
        "            succ_seed -= 1\n"
        "            if suc_test_seed_list:\n"
        "                suc_test_seed_list.pop()\n"
        "            now_seed += 1\n"
        '            args["render_freq"] = render_freq\n'
        "            continue\n"
    )
    if unstable_old in text:
        text = text.replace(unstable_old, unstable_new, 1)
    # Chunk execution only needs a rendered frame when the next step will
    # actually hit the policy server. `model.needs_obs()` (the bridge) knows
    # whether the buffered chunk still has actions; everything else gets a
    # cheap proprio-only dict, which is all the decoders read. Camera
    # rendering is the dominant sim cost on GPU-sim.
    if getattr(args, "skip_intra_chunk_obs", False):
        loop_old = (
            "        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:\n"
            "            observation = TASK_ENV.get_obs()\n"
            "            eval_func(TASK_ENV, model, observation)\n"
        )
        loop_new = (
            "        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:\n"
            "            observation = (\n"
            "                TASK_ENV.get_obs() if model.needs_obs()\n"
            "                else _vla_proprio_observation(TASK_ENV)\n"
            "            )\n"
            "            eval_func(TASK_ENV, model, observation)\n"
        )
        helper = (
            "\ndef _vla_proprio_observation(TASK_ENV):\n"
            '    """Robot state only: a buffered chunk needs no camera render."""\n'
            "    left = TASK_ENV.robot.get_left_arm_jointState()\n"
            "    right = TASK_ENV.robot.get_right_arm_jointState()\n"
            "    return {\n"
            '        "joint_action": {\n'
            '            "left_arm": left[:-1],\n'
            '            "left_gripper": left[-1],\n'
            '            "right_arm": right[:-1],\n'
            '            "right_gripper": right[-1],\n'
            '            "vector": np.array(left + right),\n'
            "        },\n"
            "    }\n\n"
        )
        if loop_old in text:
            text = text.replace(loop_old, loop_new, 1)
            anchor = "def eval_policy(task_name,"
            if anchor in text:
                text = text.replace(anchor, helper + anchor, 1)
    # check_stable early-stop. Phase A (the 2000-step settle) is the dominant
    # setup cost (~60% of the ~195s check_stable); most scenes reach rest long
    # before step 2000. Every WINDOW steps we compare all actor poses against
    # the previous window: if nothing moved more than SETTLE_EPS, the scene has
    # settled — stop stepping immediately. A scene still moving at the 2000
    # step cap falls through to the original Phase-B stability verdict, so the
    # "reject unstable seeds" behaviour is preserved. Repeat-setup seeds
    # (expert_check pre-pass already settled them) skip Phase B. Instrumented:
    # logs the step count at which settling completed.
    if getattr(args, "settle_early_stop", False):
        phase_patch = (
            "\n# --- check_stable early-stop (injected) ---\n"
            "try:\n"
            "    from envs._base_task import Base_Task as _BT\n"
            "    import numpy as _np\n"
            "    _orig_init, _orig_check = _BT._init_task_env_, _BT.check_stable\n"
            "    _settled_seeds = set()\n"
            "    _WINDOW, _CAP, _EPS = 100, 2000, 1e-3\n"
            "    def _patched_init(self, *a, **k):\n"
            "        self._snap_seed = int(k.get('seed', 0))\n"
            "        return _orig_init(self, *a, **k)\n"
            "    def _poses(self):\n"
            "        return [_np.asarray(a.get_pose().p, _np.float64) for a in self.scene.get_all_actors()]\n"
            "    def _settled_at(self, ref):\n"
            "        # Per-actor ordered comparison (get_all_actors order is\n"
            "        # deterministic for a fixed seed): max per-actor displacement\n"
            "        # since the reference window. NOT an all-vs-all set distance.\n"
            "        _max = 0.0\n"
            "        for _p, _q in zip(_poses(self), ref):\n"
            "            _d = float(_np.linalg.norm(_p - _q))\n"
            "            _max = _max if _max > _d else _d\n"
            "        return _max\n"
            "    def _patched_check(self):\n"
            "        _seed = getattr(self, '_snap_seed', 0)\n"
            "        _second = _seed in _settled_seeds\n"
            "        _ref = _poses(self)\n"
            "        _i, _steps = 0, 0\n"
            "        while _steps < _CAP:\n"
            "            for _ in range(_WINDOW):\n"
            "                self.scene.step()\n"
            "            _steps += _WINDOW\n"
            "            _drift = _settled_at(self, _ref)\n"
            "            if _drift < _EPS:\n"
            "                break\n"
            "            _ref = _poses(self)\n"
            "        if _steps < _CAP:\n"
            "            print(f'[check_stable] task={self.task_name} seed={_seed} settled_at_step={_steps} '\n"
            "                  f'drift={_drift:.6f}m second={_second}', flush=True)\n"
            "            if _second:\n"
            "                return True, []\n"
            "            _settled_seeds.add(_seed)\n"
            "            return True, []\n"
            "        print(f'[check_stable] task={self.task_name} seed={_seed} NOT_SETTLED_by_{_CAP} '\n"
            "              f'drift={_drift:.6f}m -> official phase-B verdict', flush=True)\n"
            "        if _second:\n"
            "            return True, []\n"
            "        _ok, _unstable = _orig_check(self)\n"
            "        if _ok:\n"
            "            _settled_seeds.add(_seed)\n"
            "        return _ok, _unstable\n"
            "    _BT._init_task_env_, _BT.check_stable = _patched_init, _patched_check\n"
            "except Exception as _e:\n"
            "    print(f'[check_stable-patch] disabled: {_e}')\n"
            "# --- end check_stable patch ---\n\n"
        )
        anchor = "def eval_policy(task_name,"
        if anchor in text:
            text = text.replace(anchor, phase_patch + anchor, 1)
    # Setup attribution (diagnostics only, default off): times the three parts
    # of a setup — _init_task_env_ (scene build + check_stable), check_stable
    # alone, and play_once (expert rollout) — and prints them to the worker's
    # own eval log (same stdout the official loop uses, so no path issues).
    if getattr(args, "time_setup", False):
        timing_patch = (
            "\n# --- setup attribution (injected) ---\n"
            "try:\n"
            "    from envs._base_task import Base_Task as _TB\n"
            "    import time as _tm\n"
            "    _o_init, _o_play, _o_check = _TB._init_task_env_, _TB.play_once, _TB.check_stable\n"
            "    def _t_init(self, *a, **k):\n"
            "        self._t_seed = int(k.get('seed', 0))\n"
            "        _t0 = _tm.time(); _r = _o_init(self, *a, **k); _t1 = _tm.time()\n"
            "        print(f'[time] init_task_env seed={self._t_seed} {_t1-_t0:.2f}s', flush=True)\n"
            "        return _r\n"
            "    def _t_check(self):\n"
            "        _t0 = _tm.time(); _r = _o_check(self); _t1 = _tm.time()\n"
            "        print(f'[time] check_stable seed={getattr(self, \"_t_seed\", 0)} {_t1-_t0:.2f}s', flush=True)\n"
            "        return _r\n"
            "    def _t_play(self):\n"
            "        _t0 = _tm.time(); _r = _o_play(self); _t1 = _tm.time()\n"
            "        print(f'[time] play_once seed={getattr(self, \"_t_seed\", 0)} {_t1-_t0:.2f}s', flush=True)\n"
            "        return _r\n"
            "    _TB._init_task_env_, _TB.check_stable, _TB.play_once = _t_init, _t_check, _t_play\n"
            "except Exception as _e:\n"
            "    print(f'[setup-timing] disabled: {_e}')\n"
            "# --- end setup attribution ---\n\n"
        )
        anchor = "def eval_policy(task_name,"
        if anchor in text:
            text = text.replace(anchor, timing_patch + anchor, 1)
    # Advance by a full round of shard blocks, not by one seed: neighbouring
    # shards own adjacent ranges, so +1 would run another shard's seed.
    seed_stride = int(getattr(args, "seed_stride", 1) or 1)
    if seed_stride != 1:
        text = text.replace("now_seed += 1", f"now_seed += {seed_stride}")
    # Offline expert check (RoboTwin source untouched):
    #   benchmark.offline_expert_check_scan: <dir>  expert-only pass; records every admitted
    #       (seed, episode_info) and dumps <dir>/<task>.pkl as it goes.
    #   benchmark.offline_expert_check: <dir>   consume <dir>/<task>.pkl and skip the
    #       expert pre-pass (one setup_demo + policy rollout per cached seed).
    # The result is only valid for the environment that produced it (see doc 2.46).
    scan_dir = getattr(args, "offline_expert_check_scan", None)
    cache_dir = getattr(args, "offline_expert_check", None)
    if scan_dir or cache_dir:
        _task = str(getattr(args, "task_name", "task"))
        cache_path = str(pathlib.Path(str(cache_dir)) / f"{_task}.pkl") if cache_dir else None
        # Consume mode indexes the cache; that index is *not* a seed. Use the explicit
        # offset the planner puts on each unit, and only fall back to start_seed for
        # configs/plans produced before the key existed.
        _raw_offset = getattr(args, "offline_expert_check_offset", None)
        offset = (
            int(_raw_offset)
            if _raw_offset is not None
            else int(getattr(args, "start_seed_override", 0) or 0)
        )
        _raw_part = getattr(args, "offline_expert_check_part", None)
        _part_index = -1 if _raw_part is None else int(_raw_part)
        _scan_stride = int(getattr(args, "seed_stride", 1) or 1)
        header = (
            "\n# --- offline expert check (injected) ---\n"
            "try:\n"
            "    import os as _sos, pickle as _spk\n"
            f"    _S_SCAN_DIR = {str(scan_dir) if scan_dir else None!r}\n"
            f"    _S_CACHE_PATH = {cache_path!r}\n"
            f"    _S_OFFSET = {offset}\n"
            f"    _S_PART = {_part_index}\n"
            "    _S_CACHE = _spk.load(open(_S_CACHE_PATH, 'rb')) if _S_CACHE_PATH else None\n"
            "    _S_SEEDS = list(_S_CACHE['seeds']) if _S_CACHE else []\n"
            "    _S_INFO = dict(_S_CACHE.get('episode_info') or {}) if _S_CACHE else {}\n"
            "    _S_ATTRS = dict(_S_CACHE.get('attrs') or {}) if _S_CACHE else {}\n"
            "    _S_CURRENT_ATTRS = {}\n"
            "    _S_SCAN = []\n"
            "    def _s_task_attrs(env):\n"
            "        _d = {}\n"
            "        _a = getattr(env, 'arm_tag', None)\n"
            "        if _a is not None:\n"
            "            _d['arm_tag'] = str(getattr(_a, 'value', _a))\n"
            "        _z = getattr(env, 'origin_z', None)\n"
            "        if _z is not None:\n"
            "            _d['origin_z'] = float(_z)\n"
            "        return _d\n"
            "    def _s_dump_scan(task_name, task_config):\n"
            "        if not _S_SCAN_DIR:\n"
            "            return\n"
            "        _sos.makedirs(_S_SCAN_DIR, exist_ok=True)\n"
            "        _suffix = '' if _S_PART < 0 else '.part%d' % _S_PART\n"
            "        _p = _sos.path.join(_S_SCAN_DIR, str(task_name) + _suffix + '.pkl')\n"
            "        with open(_p, 'wb') as _f:\n"
            "            _spk.dump({'schema': 1, 'task': task_name, 'task_config': task_config,\n"
            "                       'n': len(_S_SCAN),\n"
            "                       'seeds': [int(_s) for _s, _i, _a in _S_SCAN],\n"
            "                       'episode_info': {str(int(_s)): _i for _s, _i, _a in _S_SCAN},\n"
            "                       'attrs': {str(int(_s)): _a for _s, _i, _a in _S_SCAN}}, _f)\n"
            "        print(f'[offline-expert-check] {_p} -> {len(_S_SCAN)} seeds', flush=True)\n"
            "except Exception as _e:\n"
            "    print(f'[offline-expert-check] disabled: {_e}')\n"
            "# --- end offline expert check ---\n\n"
        )
        anchor = "def eval_policy(task_name,"
        if anchor in text:
            text = text.replace(anchor, header + anchor, 1)
        if scan_dir:
            # No policy is needed to validate seeds: skip building the client
            # (the coordinator may still run a server; it stays idle).
            text = text.replace(
                "    model = get_model(usr_args)\n",
                "    model = None  # offline_expert_check scan: expert pass only\n",
                1,
            )
            scan_block = (
                "        if _S_SCAN_DIR:\n"
                "            _S_SCAN.append((int(now_seed), episode_info.get('info', {}),\n"
                "                            _s_task_attrs(TASK_ENV)))\n"
                "            _s_dump_scan(args['task_name'], args['task_config'])\n"
                "            args['render_freq'] = render_freq\n"
                f"            now_seed += {_scan_stride}\n"
                "            continue\n\n"
            )
            # The policy-phase setup is wrapped in try/except by the UnStableError
            # patch, so anchor on that wrapped form and insert the scan block
            # just before it (we skip the policy episode entirely in scan mode).
            anchor2 = (
                "        try:\n"
                "            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
            )
            if anchor2 in text:
                text = text.replace(anchor2, scan_block + anchor2, 1)
            else:
                # Fall back to the unwrapped form (when the UnStableError patch
                # did not apply, e.g. a stripped-down script in tests).
                anchor3 = (
                    "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
                    '        episode_info_list = [episode_info["info"]]\n'
                )
                if anchor3 not in text:
                    raise ValueError(
                        "offline_expert_check scan: eval loop anchor not found in worker script"
                    )
                text = text.replace(anchor3, scan_block + anchor3, 1)
        else:
            # Consume: no expert pre-pass; take seeds straight from the cache.
            text = text.replace(
                "    expert_check = True\n",
                "    expert_check = False  # offline_expert_check: admitted seeds come from the cache\n",
                1,
            )
            # A cached seed should be stable in the environment that scanned it;
            # if it is not, the cache is stale (environment drift) -> say so loudly.
            text = text.replace(
                "        except UnStableError:\n"
                "            TASK_ENV.close_env()\n",
                "        except UnStableError:\n"
                "            print('[offline-expert-check] CACHE STALE: cached seed %d is unstable '"
                " % now_seed + 'here; re-scan the cache', flush=True)\n"
                "            TASK_ENV.close_env()\n"
                "            raise SystemExit(3)  # stale cache: fail loud, do not skip\n",
                1,
            )
            loop_anchor = (
                "    while succ_seed < test_num:\n"
                '        render_freq = args["render_freq"]\n'
            )
            loop_new = (
                "    while succ_seed < test_num:\n"
                "        _idx = _S_OFFSET + succ_seed\n"
                "        if _idx >= len(_S_SEEDS):\n"
                "            raise SystemExit('[offline-expert-check] cache exhausted at index %d (n=%d)'\n"
                "                             % (_idx, len(_S_SEEDS)))\n"
                "        now_seed = _S_SEEDS[_idx]\n"
                "        _S_CURRENT_ATTRS.clear()\n"
                "        _S_CURRENT_ATTRS.update(_S_ATTRS.get(str(now_seed), {}))\n"
                "        episode_info = {'info': _S_INFO.get(now_seed, _S_INFO.get(str(now_seed), {}))}\n"
                '        render_freq = args["render_freq"]\n'
            )
            if loop_anchor in text:
                text = text.replace(loop_anchor, loop_new, 1)
            else:
                raise ValueError("offline_expert_check: eval loop anchor not found in worker script")
    # Without the expert pre-pass the official loop never calls play_once(), and a
    # few tasks initialise check_success() state (arm_tag / origin_z) there. Seed
    # it right after setup_demo instead of editing the RoboTwin source.
    if cache_dir or getattr(args, "disable_expert_check", False):
        env_anchor = '    TASK_ENV = class_decorator(args["task_name"])\n'
        if env_anchor not in text:
            raise ValueError("env init shim: task-env anchor not found in worker script")
        # The cache header defines _S_CURRENT_ATTRS at module level. With only
        # --disable-expert-check and no cache that header is never emitted, and
        # the shim below would raise NameError on the first setup_demo, leaving
        # the unit with zero output. Define the holder here in that case.
        attrs_init = "" if (scan_dir or cache_dir) else "    _S_CURRENT_ATTRS = {}\n"
        env_shim = (
            env_anchor
            + "\n    # --- env init shim (injected) ---\n"
            + attrs_init
            + "    from loongforge.embodied.eval.orchestrator.runners.robotwin_runner import (\n"
            "        clear_episode_task_attrs as _vla_clear_attrs,\n"
            "        fill_missing_task_attrs as _vla_fill_attrs,\n"
            "    )\n"
            '    _vla_task_name = args["task_name"]\n'
            "    _vla_orig_setup = TASK_ENV.setup_demo\n\n"
            "    def _vla_setup_with_attrs(*_a, **_k):\n"
            "        # Task envs are reused across episodes, so the previous episode's\n"
            "        # arm_tag / origin_z are still on the object; hasattr() would then\n"
            "        # shadow this episode's value and check_success would score\n"
            "        # against the wrong arm.\n"
            "        _vla_clear_attrs(TASK_ENV)\n"
            "        _ret = _vla_orig_setup(*_a, **_k)\n"
            "        _vla_fill_attrs(TASK_ENV, _vla_task_name, _S_CURRENT_ATTRS)\n"
            "        return _ret\n\n"
            "    TASK_ENV.setup_demo = _vla_setup_with_attrs\n"
            "    # --- end env init shim ---\n"
        )
        text = text.replace(env_anchor, env_shim, 1)
    # Instruction sampling must see the same candidate list as the single-process
    # protocol: ``generate_episode_descriptions(..., max_descriptions)`` fills a
    # list whose length feeds ``np.random.choice``, so a 25-episode unit would
    # otherwise draw a different paraphrase than a 100-episode task for the same
    # seed (observed as task-level success shifts on identical seeds).
    desc_count = int(getattr(args, "episodes_per_task", 0) or 0)
    if desc_count <= 0:
        desc_count = int(getattr(args, "test_num_override", 0) or 0)
    desc_old = 'generate_episode_descriptions(args["task_name"], episode_info_list, test_num)'
    if desc_count > 0 and desc_old in text:
        text = text.replace(
            desc_old,
            f'generate_episode_descriptions(args["task_name"], episode_info_list, {desc_count})',
            1,
        )
    script_dir = pathlib.Path(args.output_dir) / "worker_files"
    script_dir.mkdir(parents=True, exist_ok=True)
    worker_script = script_dir / f"eval_policy_{args.ckpt_setting}.py"
    worker_script.write_text(text, encoding="utf-8")
    return worker_script


def build_command(
    args: argparse.Namespace, config_path: pathlib.Path, eval_script: Optional[pathlib.Path] = None
) -> List[str]:
    """Run build_command."""
    return [
        args.robotwin_python,
        str(eval_script or (pathlib.Path(args.robotwin_path) / "script" / "eval_policy.py")),
        "--config",
        str(config_path),
        "--overrides",
        "--task_name",
        args.task_name,
        "--task_config",
        args.task_config,
        "--ckpt_setting",
        args.ckpt_setting,
        "--seed",
        str(args.seed),
        "--policy_name",
        "loongforge.embodied.eval.bridges.robotwin_policy",
    ]


def _robotwin_result_root(args: argparse.Namespace) -> pathlib.Path:
    """Run _robotwin_result_root."""
    return (
        pathlib.Path(args.robotwin_path)
        / "eval_result"
        / args.task_name
        / "loongforge.embodied.eval.bridges.robotwin_policy"
        / args.task_config
        / args.ckpt_setting
    )


def _robotwin_result_dirs(args: argparse.Namespace) -> Set[pathlib.Path]:
    """Run _robotwin_result_dirs."""
    result_root = _robotwin_result_root(args)
    if not result_root.is_dir():
        return set()
    return {path.resolve() for path in result_root.iterdir() if path.is_dir()}


def _new_robotwin_result_dir(args: argparse.Namespace, before_dirs: Set[pathlib.Path]) -> Optional[pathlib.Path]:
    """Run _new_robotwin_result_dir."""
    candidates = sorted(_robotwin_result_dirs(args) - before_dirs, key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _robotwin_artifact_dir(args: argparse.Namespace) -> pathlib.Path:
    """Run _robotwin_artifact_dir."""
    return pathlib.Path(args.output_dir) / "artifacts" / "robotwin" / args.task_name / args.task_config


def _copy_file_if_exists(source: pathlib.Path, target_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Run _copy_file_if_exists."""
    if not source.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _collect_robotwin_artifacts(
    args: argparse.Namespace,
    log_path: pathlib.Path,
    config_path: Optional[pathlib.Path],
    result_dir: Optional[pathlib.Path],
) -> Dict[str, object]:
    """Run _collect_robotwin_artifacts."""
    artifact_dir = _robotwin_artifact_dir(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, object] = {"artifact_dir": str(artifact_dir)}
    copied_log = _copy_file_if_exists(log_path, artifact_dir)
    if copied_log is not None:
        copied["official_eval_log"] = str(copied_log)
    if config_path is not None:
        copied_config = _copy_file_if_exists(config_path, artifact_dir)
        if copied_config is not None:
            copied["deploy_config"] = str(copied_config)

    if result_dir is None:
        return copied
    copied["robotwin_result_dir"] = str(result_dir)
    copied_result = _copy_file_if_exists(result_dir / "_result.txt", artifact_dir)
    if copied_result is not None:
        copied["result_txt"] = str(copied_result)

    trace_path = artifact_dir / "trace.json"
    if trace_path.is_file():
        copied["trace_path"] = str(trace_path)

    video_dir = artifact_dir / "videos"
    videos = []
    for video in sorted(result_dir.rglob("*.mp4")):
        copied_video = _copy_file_if_exists(video, video_dir)
        if copied_video is not None:
            videos.append(str(copied_video))
    if videos:
        copied["videos"] = videos
    return copied


def _cleanup_robotwin_work_files(
    args: argparse.Namespace,
    log_path: pathlib.Path,
    config_path: Optional[pathlib.Path],
) -> None:
    """Run _cleanup_robotwin_work_files."""
    artifact_dir = _robotwin_artifact_dir(args).resolve()
    for path in [log_path, config_path]:
        if path is None:
            continue
        resolved = path.resolve()
        if path.exists() and artifact_dir not in resolved.parents:
            path.unlink()
    worker_dir = pathlib.Path(args.output_dir) / "worker_files"
    if worker_dir.exists():
        shutil.rmtree(worker_dir)


def _override_step_limit(robotwin_path: pathlib.Path, task_name: str, step_limit: Optional[int]) -> Optional[str]:
    """Run _override_step_limit."""
    if step_limit is None:
        return None
    step_limit_path = robotwin_path / "task_config" / "_eval_step_limit.yml"
    original_text = step_limit_path.read_text(encoding="utf-8")
    data = yaml.safe_load(original_text) or {}
    data[task_name] = int(step_limit)
    step_limit_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return original_text


def _override_eval_policy_for_smoke(
    robotwin_path: pathlib.Path,
    test_num: Optional[int],
    disable_expert_check: bool,
    start_seed: Optional[int] = None,
) -> Optional[str]:
    """Run _override_eval_policy_for_smoke."""
    if test_num is None and not disable_expert_check and start_seed is None:
        return None
    eval_policy_path = robotwin_path / "script" / "eval_policy.py"
    original_text = eval_policy_path.read_text(encoding="utf-8")
    patched_text = original_text
    if start_seed is not None:
        patched_text = patched_text.replace(
            "    st_seed = 100000 * (1 + seed)\n", f"    st_seed = {int(start_seed)}\n", 1
        )
    if test_num is not None:
        patched_text = patched_text.replace("    test_num = 100\n", f"    test_num = {int(test_num)}\n", 1)
    if disable_expert_check:
        patched_text = patched_text.replace("    expert_check = True\n", "    expert_check = False\n", 1)
    eval_policy_path.write_text(patched_text, encoding="utf-8")
    return original_text


def _apply_config(args: argparse.Namespace, config: Dict[str, object]) -> argparse.Namespace:
    """Run _apply_config."""
    benchmark = config.get("benchmark") or {}
    model = config.get("model") or {}
    server = config.get("server") or {}
    env = config.get("env") or {}
    run = config.get("run") or {}
    timeouts = config.get("timeouts") or {}

    if not isinstance(benchmark, dict) or not isinstance(model, dict) or not isinstance(server, dict):
        raise ValueError("benchmark, model, and server sections must be mappings")

    mapping = [
        (benchmark, "task_name", "task_name"),
        (benchmark, "task_config", "task_config"),
        (benchmark, "max_steps", "max_steps"),
        (benchmark, "start_seed", "start_seed_override"),
        (benchmark, "seed_stride", "seed_stride"),
        # episodes_per_task is the whole-run default; a shard's explicit test_num
        # must win, so it is applied after this line.
        (benchmark, "episodes_per_task", "test_num_override"),
        (benchmark, "episodes_per_task", "episodes_per_task"),
        (benchmark, "test_num", "test_num_override"),
        (benchmark, "disable_expert_check", "disable_expert_check"),
        (benchmark, "domain_id", "domain_id"),
        (benchmark, "action_bridge", "action_bridge"),
        (benchmark, "instruction_type", "instruction_type"),
        (benchmark, "return_action_chunk", "return_action_chunk"),
        (benchmark, "disable_action_cache", "disable_action_cache"),
        (benchmark, "chunk_execute_steps", "chunk_execute_steps"),
        (benchmark, "skip_intra_chunk_obs", "skip_intra_chunk_obs"),
        (benchmark, "use_official_step_limits", "use_official_step_limits"),
        (benchmark, "settle_early_stop", "settle_early_stop"),
        (benchmark, "time_setup", "time_setup"),
        (benchmark, "sim_cpu_threads", "sim_cpu_threads"),
        (benchmark, "offline_expert_check", "offline_expert_check"),
        (benchmark, "offline_expert_check_offset", "offline_expert_check_offset"),
        (benchmark, "offline_expert_check_scan", "offline_expert_check_scan"),
        (benchmark, "offline_expert_check_part", "offline_expert_check_part"),
        (server, "chunk_execute_steps", "chunk_execute_steps"),
        (server, "ckpt_path", "policy_ckpt_path"),
        (model, "unnorm_key", "unnorm_key"),
        (model, "action_mode", "action_mode"),
        (model, "robotwin_action_bridge", "action_bridge"),
        (server, "host", "host"),
        (server, "port", "port"),
        (env, "loongforge_root", "loongforge_root"),
        (env, "eval_root", "eval_root"),
        (env, "robotwin_root", "robotwin_path"),
        (env, "robotwin_python", "robotwin_python"),
        (run, "seed", "seed"),
        (run, "write_trace", "write_trace"),
        (run, "output_dir", "output_dir"),
        (run, "ckpt_setting", "ckpt_setting"),
        (run, "disable_eval_video_log", "disable_eval_video_log"),
        (run, "keep_config", "keep_config"),
        (run, "worker_local_files", "worker_local_files"),
        (run, "cuda_visible_devices", "cuda_visible_devices"),
        (timeouts, "per_episode_sec", "timeout_sec"),
    ]
    for section, key, attr in mapping:
        value = section.get(key) if isinstance(section, dict) else None
        if value is not None:
            setattr(args, attr, value)

    # Official runs keep each task's own limit from task_config/_eval_step_limit.yml;
    # performance runs pin one horizon so every arm is comparable.
    if getattr(args, "use_official_step_limits", False):
        # Keep metadata (adapter context, trace, records) on the same horizon the
        # official script will use, without overriding the file itself.
        official = _official_step_limit(args)
        if official is not None:
            args.max_steps = official
    elif args.step_limit_override is None:
        args.step_limit_override = args.max_steps
    return args


def _official_step_limit(args: argparse.Namespace) -> Optional[int]:
    """Per-task step limit from RoboTwin's `_eval_step_limit.yml`, if present."""
    task = getattr(args, "task_name", "")
    root = getattr(args, "robotwin_path", "")
    if not task or not root:
        return None
    path = pathlib.Path(root) / "task_config" / "_eval_step_limit.yml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    value = data.get(task)
    return int(value) if value is not None else None


def _robotwin_rate_from_result(result_txt: Optional[str]) -> Optional[float]:
    """Parse official `_result.txt` last numeric line as success rate in [0, 1]."""
    if not result_txt:
        return None
    for line in reversed(str(result_txt).strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def _parse_official_episode_outcomes(log_text: str) -> List[Dict[str, Any]]:
    """Parse per-episode outcomes from official eval_policy log.

    Official loop prints after each valid episode:
      Success rate: {suc}/{test_num} => {pct}%, current seed: {now_seed}

    Episode success is inferred from the cumulative counter delta
    (suc increases by 1 on success, stays flat on fail).
    """
    episodes: List[Dict[str, Any]] = []
    prev_suc = 0
    for raw_line in str(log_text or "").splitlines():
        line = _ANSI_RE.sub("", raw_line)
        match = _SUCCESS_RATE_LINE_RE.search(line)
        if not match:
            continue
        suc = int(match.group(1))
        test_num = int(match.group(2))
        seed = int(match.group(3))
        success = 1 if suc > prev_suc else 0
        prev_suc = suc
        episodes.append(
            {
                "episode_idx": len(episodes),
                "seed": seed,
                "success": success,
                "cum_suc": suc,
                "cum_test_num": test_num,
            }
        )
    return episodes


def _trace_step_count(trace_path: Optional[str]) -> Optional[int]:
    """Executed steps of the (single) episode recorded in ``trace.json``.

    ``args.max_steps`` is the horizon cap, not the number of steps actually run:
    an episode that succeeds early stops before the cap. The trace has one entry
    per executed step, so it is the only faithful source for ``steps``.
    """
    if not trace_path:
        return None
    try:
        data = json.loads(pathlib.Path(str(trace_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    steps = data.get("steps") if isinstance(data, dict) else None
    return len(steps) if isinstance(steps, list) and steps else None


def _build_robotwin_records(
    args: argparse.Namespace,
    returncode: int,
    artifacts: Dict[str, object],
    episode_time_sec: float,
    result_txt: Optional[str],
    log_text: str,
) -> List[Dict[str, Any]]:
    """Build per-episode records aligned with LIBERO/SimplerEnv (1 ep = 1 row).

    Prefer official log episode lines; fall back to a single row from `_result.txt`
    rate when the log cannot be parsed (keeps smoke/partial runs usable).
    """
    official_rate = _robotwin_rate_from_result(result_txt)
    episodes = _parse_official_episode_outcomes(log_text)
    n_eps = len(episodes)
    # Wall-clock is for the whole official subprocess; split evenly when multi-ep.
    per_ep_time = float(episode_time_sec) / n_eps if n_eps > 0 else float(episode_time_sec)
    # A single-episode unit's trace is authoritative for the steps actually run;
    # multi-episode units only keep the last episode's trace, so fall back to the cap.
    traced_steps = _trace_step_count(artifacts.get("trace_path")) if n_eps <= 1 else None
    common = {
        "benchmark": "robotwin",
        "task_suite": args.task_name,
        "task_id": 0,
        "task_description": args.task_name.replace("_", " "),
        "task_config": args.task_config,
        "ckpt_setting": args.ckpt_setting,
        "steps": int(traced_steps or args.max_steps),
        "returncode": int(returncode),
        "result_path": artifacts.get("result_txt"),
        "official_eval_log": artifacts.get("official_eval_log"),
        "trace_path": artifacts.get("trace_path"),
        "videos": artifacts.get("videos", []),
    }

    if episodes:
        successes = sum(int(ep["success"]) for ep in episodes)
        rate = successes / n_eps
        # Prefer official _result.txt rate when present (canonical RoboTwin metric).
        if official_rate is not None:
            rate = float(official_rate)
        records: List[Dict[str, Any]] = []
        for ep in episodes:
            success = int(ep["success"])
            seed = int(ep["seed"])
            records.append(
                {
                    **common,
                    "episode_idx": int(ep["episode_idx"]),
                    "episode_id": (
                        f"robotwin/{args.task_name}/{args.task_config}/seed={seed}"
                    ),
                    "seed": seed,
                    "success": success,
                    "success_rate": rate,
                    "n_episodes": n_eps,
                    "episode_time_sec": per_ep_time,
                    "failure_reason": None if success else "not_successful",
                }
            )
        return records

    # Fallback: no parseable episodes — one synthetic row (legacy-compatible shape).
    # Fallback: no parseable episodes.
    if official_rate is None and int(returncode) != 0:
        # The official process died before completing an episode (e.g. an
        # uncaught UnStableError). Do not fabricate a model failure: return no
        # record so the coordinator retries/repaire the shard instead of
        # counting a fake 100-step episode.
        return []
    if official_rate is not None:
        success = int(official_rate > 0.0)
        rate = float(official_rate)
    else:
        success = 0
        rate = 0.0
    seed = int(getattr(args, "start_seed_override", None) or args.seed)
    return [
        {
            **common,
            "episode_idx": 0,
            "episode_id": f"robotwin/{args.task_name}/{args.task_config}/seed={seed}",
            "seed": seed,
            "success": success,
            "success_rate": rate,
            "n_episodes": 1,
            "episode_time_sec": float(episode_time_sec),
            "failure_reason": None if success else "not_successful",
        }
    ]


def _write_standard_outputs(
    args: argparse.Namespace,
    returncode: int,
    artifacts: Dict[str, object],
    episode_time_sec: float,
) -> None:
    """Write per-episode results.jsonl + summary CSVs (LIBERO-style aggregation)."""
    result_txt = None
    result_path = artifacts.get("result_txt")
    if result_path:
        path = pathlib.Path(str(result_path))
        if path.is_file():
            result_txt = path.read_text(encoding="utf-8")

    log_text = ""
    log_path = artifacts.get("official_eval_log")
    if log_path:
        lp = pathlib.Path(str(log_path))
        if lp.is_file():
            log_text = lp.read_text(encoding="utf-8", errors="replace")

    if getattr(args, "offline_expert_check_scan", None):
        # Scan units run no policy episodes, so they would look "short" to the
        # coordinator's coverage gate. Emit one placeholder record per expected
        # episode so the unit counts as complete (scan dirs are separate).
        n = max(1, int(getattr(args, "test_num_override", 1) or 1))
        records = [
            {
                "episode_id": (
                    f"robotwin/{args.task_name}/{args.task_config}"
                    f"/scan-part{int(getattr(args, 'offline_expert_check_part', -1) or 0)}={i}"
                ),
                "seed": None,
                "success": True,
                "success_rate": 1.0,
                "n_episodes": 1,
                "episode_time_sec": float(episode_time_sec),
                "failure_reason": None,
                "scan": True,
            }
            for i in range(n)
        ]
    else:
        records = _build_robotwin_records(
            args, returncode, artifacts, episode_time_sec, result_txt, log_text
        )
    output_dir = pathlib.Path(args.output_dir)
    for record in records:
        append_jsonl(output_dir / "results.jsonl", record)
    write_summary_csv(output_dir / "summary.csv", records)
    write_suite_summary_csv(output_dir / "suite_summary.csv", records)


def merge_offline_expert_check_scan(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Join ``<task>.part<i>.pkl`` scan shards into ``<task>.pkl``.

    Sharded scans each walk a residue class of the candidate seed space, so the
    parts are disjoint by construction; the merge sorts by seed and truncates to
    ``episodes_per_task``.
    """
    import pickle

    benchmark = config.get("benchmark") or {}
    scan_dir = benchmark.get("offline_expert_check_scan")
    if not scan_dir:
        return None
    base = pathlib.Path(str(scan_dir))
    parts = sorted(base.glob("*.part*.pkl"))
    if not parts:
        return None
    by_task: Dict[str, list] = {}
    for path in parts:
        task = path.name.split(".part")[0]
        try:
            by_task.setdefault(task, []).append(pickle.loads(path.read_bytes()))
        except Exception:
            continue
    want = int(benchmark.get("episodes_per_task") or benchmark.get("test_num") or 0)
    summary: Dict[str, int] = {}
    for task, chunks in by_task.items():
        seeds: list = []
        info: Dict[str, Any] = {}
        attrs: Dict[str, Any] = {}
        task_config = None
        for chunk in chunks:
            task_config = task_config or chunk.get("task_config")
            chunk_info = chunk.get("episode_info") or {}
            chunk_attrs = chunk.get("attrs") or {}
            for seed in chunk.get("seeds") or []:
                key = str(int(seed))
                if key in info:
                    continue
                info[key] = chunk_info.get(key, {})
                if key in chunk_attrs:
                    attrs[key] = chunk_attrs[key]
                seeds.append(int(seed))
        seeds.sort()
        if want > 0:
            seeds = seeds[:want]
        merged = {
            "schema": 1,
            "task": task,
            "task_config": task_config,
            "n": len(seeds),
            "seeds": seeds,
            "episode_info": {str(s): info[str(s)] for s in seeds},
            "attrs": {str(s): attrs[str(s)] for s in seeds if str(s) in attrs},
        }
        (base / f"{task}.pkl").write_bytes(pickle.dumps(merged))
        summary[task] = len(seeds)
    print(f"[offline-expert-check] merged {len(parts)} part(s) -> {summary}", flush=True)
    return summary


def run_evaluation(args: argparse.Namespace) -> int:
    """Run run_evaluation."""
    robotwin_path = pathlib.Path(args.robotwin_path)
    eval_script = robotwin_path / "script" / "eval_policy.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"RoboTwin eval entry does not exist: {eval_script}")
    if not pathlib.Path(args.policy_ckpt_path).exists():
        raise FileNotFoundError(f"Policy checkpoint path does not exist: {args.policy_ckpt_path}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path: Optional[pathlib.Path] = None
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    original_step_limit_text: Optional[str] = None
    original_eval_policy_text: Optional[str] = None
    worker_eval_script = _worker_eval_script(args)
    start_time = time.time()
    try:
        if not args.worker_local_files:
            original_step_limit_text = _override_step_limit(robotwin_path, args.task_name, args.step_limit_override)
            original_eval_policy_text = _override_eval_policy_for_smoke(
                robotwin_path,
                args.test_num_override,
                args.disable_expert_check,
                args.start_seed_override,
            )
        if args.keep_config:
            config_path = output_dir / "deploy_policy_vla_eval_robotwin.yml"
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="vla_eval_robotwin_")
            config_path = pathlib.Path(temp_dir.name) / "deploy_policy.yml"
        _write_deploy_policy(args, config_path)

        env = os.environ.copy()
        eval_root = pathlib.Path(getattr(args, "eval_root", "") or pathlib.Path(__file__).resolve().parents[2])
        loongforge_root = pathlib.Path(getattr(args, "loongforge_root", "") or eval_root.parent)
        py_paths = [str(robotwin_path), str(eval_root), str(loongforge_root)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            py_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = ":".join(py_paths)
        robotwin_bin = str(pathlib.Path(args.robotwin_python).resolve().parent)
        env["PATH"] = f"{robotwin_bin}:{env.get('PATH', '')}"
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        command = build_command(args, config_path, worker_eval_script)
        result_dirs_before = _robotwin_result_dirs(args)
        log_path = output_dir / f"{args.task_name}_{args.task_config}_official_eval.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND: " + " ".join(command) + "\n")
            log_file.write("PYTHONPATH: " + env["PYTHONPATH"] + "\n")
            log_file.flush()
            try:
                process = subprocess.run(
                    command,
                    cwd=str(robotwin_path),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None,
                )
                returncode = int(process.returncode)
            except subprocess.TimeoutExpired:
                log_file.write(f"\nTIMEOUT after {args.timeout_sec} seconds\n")
                log_file.flush()
                returncode = 124
        result_dir = _new_robotwin_result_dir(args, result_dirs_before)
        artifacts = _collect_robotwin_artifacts(args, log_path, config_path, result_dir)
        _write_standard_outputs(args, returncode, artifacts, time.time() - start_time)
        _cleanup_robotwin_work_files(args, log_path, config_path)
        print(
            json.dumps(
                {
                    "returncode": returncode,
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return returncode
    finally:
        if original_eval_policy_text is not None:
            (robotwin_path / "script" / "eval_policy.py").write_text(original_eval_policy_text, encoding="utf-8")
        if original_step_limit_text is not None:
            (robotwin_path / "task_config" / "_eval_step_limit.yml").write_text(
                original_step_limit_text, encoding="utf-8"
            )
        if temp_dir is not None:
            temp_dir.cleanup()


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--loongforge-root", default=os.environ.get("LOONGFORGE_ROOT", "/workspace/LoongForge-VLA"))
    parser.add_argument(
        "--eval-root",
        default=os.environ.get("EVAL_ROOT", "/workspace/LoongForge-VLA/loongforge/embodied/eval"),
    )
    parser.add_argument("--robotwin-path", default=os.environ.get("ROBOTWIN_PATH", "/workspace/RoboTwin"))
    parser.add_argument("--robotwin-python", default=os.environ.get("ROBOTWIN_PYTHON", sys.executable))
    parser.add_argument("--policy-ckpt-path", default="")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    parser.add_argument("--ckpt-setting", default="loongforge_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--action-mode", choices=["abs", "delta", "rel"], default="abs")
    # action_bridge / domain_id: YAML only (benchmark.action_bridge, benchmark.domain_id)
    parser.add_argument("--no-action-reorder", action="store_true")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--step-limit-override", type=int, default=None)
    parser.add_argument("--test-num-override", type=int, default=None)
    parser.add_argument("--start-seed-override", type=int, default=None)
    parser.add_argument("--disable-expert-check", action="store_true")
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--control-hz", type=int, default=10)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--disable-action-cache", action="store_true")
    parser.add_argument("--return-action-chunk", action="store_true")
    parser.add_argument("--skip-intra-chunk-obs", action="store_true")
    parser.add_argument("--use-official-step-limits", action="store_true")
    parser.add_argument("--settle-early-stop", action="store_true")
    parser.add_argument("--time-setup", action="store_true")
    parser.add_argument("--sim-cpu-threads", type=int, default=None)
    parser.add_argument("--disable-eval-video-log", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/robotwin/official_eval",
    )
    parser.add_argument("--keep-config", action="store_true")
    parser.add_argument("--worker-local-files", action="store_true")
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    if args.config:
        args = _apply_config(args, load_config(args.config))
    if not args.policy_ckpt_path:
        raise SystemExit("policy checkpoint must be set by server.ckpt_path in YAML")
    if not args.task_name:
        raise SystemExit("task name must be set by benchmark.task_name in YAML")
    raise SystemExit(run_evaluation(args))


if __name__ == "__main__":
    main()
