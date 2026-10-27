# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA reuses pi05 camera packing; robot conversion happens in the policy."""

from .gr00t_n1_7 import _encode_libero_ee_axis_angle
from .pi05 import Pi05PayloadBuilder
from .registry import register_payload_builder


@register_payload_builder("vlda")
class VLDAPayloadBuilder(Pi05PayloadBuilder):
    action_horizon: int = 10
    state_encoding: str = "libero_ee_axis_angle"

    def _encode_state(self, canonical):
        if self.state_encoding == "libero_ee_axis_angle":
            return _encode_libero_ee_axis_angle(canonical["state_raw"])
        if self.state_encoding == "aloha_pi":
            raise ValueError("VLDA applies Aloha conversion server-side; use state_encoding=passthrough")
        return super()._encode_state(canonical)
