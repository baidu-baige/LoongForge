# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode loops executed by RoboDojo's sim client, for both env counts.

RoboDojo resolves ``--policy_name <name>`` to the package
``<RoboDojo>/XPolicyLab/policy/<name>/`` and imports ``deploy`` from it inside the Isaac Sim
process. The robodojo runner therefore **copies this file** into the plugin package it
generates (see ``robodojo_runner.ensure_plugin``) instead of importing it: the sim
environment is RoboDojo's own (``websockets`` 12, no LoongForge install), so the plugin must
stay stdlib-only. Keep it that way — no imports from ``loongforge``.

Both call orders mirror the official policies' ``deploy.py`` exactly; RoboDojo calls the one
that matches the plugin's ``eval_batch``:

- ``eval_one_episode`` (one env per process): one ``get_obs`` before every action, and
  ``is_episode_end`` once more at every chunk end (the outer ``while``). In between,
  ``is_episode_end`` follows the bridge's ``end_check``: after every action ("action",
  LingBot_VA / Pi_05 / X_VLA) or not at all ("chunk", FastWAM, whose remaining actions after
  an episode end are no-ops in ``take_action``). The bridge returns one action per call and
  flags the last action of each model chunk.
- ``eval_one_episode_batch`` (``eval_batch``, several envs per process; FastWAM / X_VLA / Pi_05
  share this loop officially): one batched observation, one ``get_action_batch`` returning each
  running env's chunk, then that chunk executed step by step on all running envs with
  ``is_episode_end`` after every step and a batched observation between steps for the envs
  still running. The bridge returns, per env, the rest of its current chunk.
"""


def eval_one_episode(TASK_ENV, model_client):
    """Run one episode against the LoongForge bridge."""
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        while True:
            out = model_client.call(func_name="get_action", obs=obs)
            for action in out["actions"]:
                TASK_ENV.take_action(action)
            per_action = out.get("end_check", "action") == "action"
            if (per_action and TASK_ENV.is_episode_end()) or out["chunk_end"]:
                break
            obs = TASK_ENV.get_obs()


def eval_one_episode_batch(TASK_ENV, model_client):
    """Run one batch of episodes (one per running env) against the LoongForge bridge."""
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            obs_list = TASK_ENV.get_obs_batch(env_idx_list)
            model_client.call(func_name="update_obs_batch", obs=obs_list)
