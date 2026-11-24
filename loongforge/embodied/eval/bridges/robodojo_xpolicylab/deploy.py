# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode loop executed by RoboDojo's sim client (``XPolicyLab.policy.LoongForge.deploy``).

Mirrors the official policies' ``deploy.py`` call order on the env so that
``get_obs`` / ``take_action`` / ``is_episode_end`` happen in exactly the same
sequence: one ``get_obs`` before every action and ``is_episode_end`` once more
at every chunk end (the outer ``while``). In between, ``is_episode_end`` follows
the bridge's ``end_check``: after every action ("action", LingBot_VA / Pi_05 /
X_VLA) or not at all ("chunk", FastWAM, whose remaining actions after an
episode end are no-ops in ``take_action``). The bridge returns one action per
call and flags the last action of each model chunk.
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
    """Several envs per process use the ``LoongForgeBatch`` plugin (eval_batch: true)."""
    raise NotImplementedError("LoongForge is the eval_batch: false plugin; use LoongForgeBatch for several envs per process.")
