from __future__ import annotations

import copy


def make_scaled_attack_specs(raw_specs: dict, scaler, scale_cols: list[str]) -> dict:
    return copy.deepcopy(raw_specs)
