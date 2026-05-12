from __future__ import annotations

import copy


def scale_value(col: str, raw_value: float, scaler, scale_cols: list[str]) -> float:
    i = scale_cols.index(col)
    return (raw_value - scaler.data_min_[i]) / (scaler.data_max_[i] - scaler.data_min_[i] + 1e-8)


def make_scaled_attack_specs(raw_specs: dict, scaler, scale_cols: list[str]) -> dict:
    specs = copy.deepcopy(raw_specs)
    scale_map = {
        "dos_http_flood": {
            "mal_time_elapsed_min": "time_elapsed",
            "mal_time_elapsed_max": "time_elapsed",
            "valid_pkt_size_total_min": "orig_bytes",
        },
    }
    for attack_name, key_to_col in scale_map.items():
        for spec_key, col in key_to_col.items():
            specs[attack_name][spec_key] = scale_value(col, raw_specs[attack_name][spec_key], scaler, scale_cols)
    return specs
