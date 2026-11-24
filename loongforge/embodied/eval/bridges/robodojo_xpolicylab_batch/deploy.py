# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Batched episode loop executed by RoboDojo's sim client (``XPolicyLab.policy.LoongForgeBatch.deploy``).

Replays the official policies' ``eval_one_episode_batch`` (FastWAM / X_VLA / Pi_05 share this
loop) call for call on the env: one batched observation, one ``get_action_batch`` returning each
running env's chunk, then that chunk executed step by step on all running envs with
``is_episode_end()`` after every step and a batched observation between steps for the envs still
running. The bridge returns, per env, the rest of its current chunk.
"""


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


def eval_one_episode(TASK_ENV, model_client):
    """Single-env runs use the ``LoongForge`` plugin (eval_batch: false)."""
    raise NotImplementedError("LoongForgeBatch is the eval_batch: true plugin; use LoongForge for one env per process.")
