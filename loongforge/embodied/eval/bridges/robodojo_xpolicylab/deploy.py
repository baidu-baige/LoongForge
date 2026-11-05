# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode loop executed by RoboDojo's sim client (``XPolicyLab.policy.LoongForge.deploy``).

Mirrors the official LingBot_VA ``deploy.py`` call order on the env so that
``get_obs`` / ``take_action`` / ``is_episode_end`` happen in exactly the same
sequence: one ``get_obs`` before every action, ``is_episode_end`` after every
action and once more at every chunk end (the outer ``while``). The bridge
returns one action per call and flags the last action of each model chunk.
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
            if TASK_ENV.is_episode_end() or out["chunk_end"]:
                break
            obs = TASK_ENV.get_obs()


def eval_one_episode_batch(TASK_ENV, model_client):
    """Batched envs are not supported: the LoongForge model keeps per-episode state."""
    raise NotImplementedError("LoongForge RoboDojo bridge evaluates one env per process; keep eval_batch: false.")
