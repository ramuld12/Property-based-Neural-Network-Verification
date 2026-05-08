from __future__ import annotations

import torch
import torch.nn.functional as F
from property_driven_ml.constraints.constraints import Constraint
from property_driven_ml.constraints.postconditions import Postcondition
import property_driven_ml.constraints.preconditions as pml_preconditions


class TabularRuleConstraint(Constraint):
    def __init__(self, device, precondition, postcondition):
        super().__init__(device)
        self.precondition = precondition
        self.postcondition = postcondition


class FrozenFeaturePrecondition:
    def __init__(self, precondition, frozen_indices):
        self.precondition = precondition
        self.frozen_indices = frozen_indices

    def get_bounds(self, x):
        if hasattr(self.precondition, "get_bounds"):
            lo, hi = self.precondition.get_bounds(x)
        else:
            lo, hi = self.precondition.get_precondition(x)
        lo = lo.clone()
        hi = hi.clone()
        lo[..., self.frozen_indices] = x[..., self.frozen_indices]
        hi[..., self.frozen_indices] = x[..., self.frozen_indices]
        return lo, hi

    def get_precondition(self, x):
        return self.get_bounds(x)


class DoSHttpFloodPostcondition(Postcondition):
    def __init__(
        self,
        idx,
        class_idx,
        dos_http_flood_specs,
        validity_specs,
        scaler,
        scale_cols,
        min_prob=0.80,
    ):
        self.idx = idx
        self.class_idx = class_idx
        self.min_prob = min_prob
        self.dos_http_flood_specs = dos_http_flood_specs
        self.validity_specs = validity_specs
        self.scaler = scaler
        self.scale_cols = scale_cols

    def raw_col(self, x_col, col):
        i = self.scale_cols.index(col)
        min_ = torch.tensor(self.scaler.data_min_[i], device=x_col.device, dtype=x_col.dtype)
        max_ = torch.tensor(self.scaler.data_max_[i], device=x_col.device, dtype=x_col.dtype)
        return x_col * (max_ - min_ + 1e-8) + min_

    def rates_from_adv(self, x_adv):
        duration = self.raw_col(x_adv[:, self.idx["duration"]], "duration").clamp_min(1e-8)
        orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts")
        orig_bytes = self.raw_col(x_adv[:, self.idx["orig_bytes"]], "orig_bytes")
        return orig_bytes / duration, orig_pkts / duration

    def get_postcondition(self, N, x, x_adv):
        valid_tcp = x[:, self.idx["valid_tcp_handshake"]]
        valid_http = x[:, self.idx["valid_http_conn"]]
        time_elapsed = x[:, self.idx["time_elapsed"]]
        orig_bytes = x_adv[:, self.idx["orig_bytes"]]
        orig_pkts = x_adv[:, self.idx["orig_pkts"]]
        orig_byte_rate, orig_pkt_rate = self.rates_from_adv(x_adv)
        p = F.softmax(N(x_adv), dim=1)[:, self.class_idx]
        return lambda logic: logic.OR(
            logic.LT(orig_bytes, torch.full_like(orig_bytes, self.validity_specs["valid_packet_size_min_total_bytes"])),
            logic.LT(orig_pkts, torch.full_like(orig_pkts, self.validity_specs["valid_packet_size_min_pkts"])),
            logic.NEQ(valid_tcp, torch.ones_like(valid_tcp)),
            logic.NEQ(valid_http, torch.ones_like(valid_http)),
            logic.LT(orig_bytes, torch.full_like(orig_bytes, self.dos_http_flood_specs["valid_pkt_size_total_min"])),
            logic.LT(time_elapsed, torch.full_like(time_elapsed, self.dos_http_flood_specs["mal_time_elapsed_min"])),
            logic.GT(time_elapsed, torch.full_like(time_elapsed, self.dos_http_flood_specs["mal_time_elapsed_max"])),
            logic.AND(
                logic.LT(orig_byte_rate, torch.full_like(orig_byte_rate, self.dos_http_flood_specs["mal_byte_rate_min"])),
                logic.LT(orig_pkt_rate, torch.full_like(orig_pkt_rate, self.dos_http_flood_specs["mal_pkt_rate_min"])),
            ),
            logic.GEQ(p, torch.full_like(p, self.min_prob)),
        )

    @torch.no_grad()
    def debug_parts(self, N, x, x_adv):
        valid_tcp = x[:, self.idx["valid_tcp_handshake"]]
        valid_http = x[:, self.idx["valid_http_conn"]]
        time_elapsed = x[:, self.idx["time_elapsed"]]
        orig_bytes = x_adv[:, self.idx["orig_bytes"]]
        orig_pkts = x_adv[:, self.idx["orig_pkts"]]
        orig_byte_rate, orig_pkt_rate = self.rates_from_adv(x_adv)
        p = F.softmax(N(x_adv), dim=1)[:, self.class_idx]
        parts = {
            "valid_input": (orig_bytes >= self.validity_specs["valid_packet_size_min_total_bytes"])
            & (orig_pkts >= self.validity_specs["valid_packet_size_min_pkts"]),
            "valid_tcp_handshake": valid_tcp == 1,
            "valid_http_conn": valid_http == 1,
            "mal_time_elapsed_min": time_elapsed >= self.dos_http_flood_specs["mal_time_elapsed_min"],
            "mal_time_elapsed_max": time_elapsed <= self.dos_http_flood_specs["mal_time_elapsed_max"],
            "valid_pkt_size_total_min": orig_bytes >= self.dos_http_flood_specs["valid_pkt_size_total_min"],
            "mal_byte_rate_min": orig_byte_rate >= self.dos_http_flood_specs["mal_byte_rate_min"],
            "mal_pkt_rate_min": orig_pkt_rate >= self.dos_http_flood_specs["mal_pkt_rate_min"],
            "prediction_ok": p >= self.min_prob,
        }
        parts["antecedent_true"] = (
            parts["valid_input"]
            & parts["valid_tcp_handshake"]
            & parts["valid_http_conn"]
            & parts["mal_time_elapsed_min"]
            & parts["mal_time_elapsed_max"]
            & parts["valid_pkt_size_total_min"]
            & (parts["mal_byte_rate_min"] | parts["mal_pkt_rate_min"])
        )
        return parts


class PortscanPostcondition(Postcondition):
    def __init__(self, idx, class_idx, min_prob, portscan_specs):
        self.idx = idx
        self.class_idx = class_idx
        self.min_prob = min_prob
        self.portscan_specs = portscan_specs

    def get_postcondition(self, N, x, x_adv):
        uniq_dst_ports = x[:, self.idx["uniq_dst_ports"]]
        fail_ratio = x[:, self.idx["fail_ratio"]]
        pkts_per_port = x[:, self.idx["pkts_per_port"]]
        scan_duration = x[:, self.idx["scan_duration"]]
        p = F.softmax(N(x_adv), dim=1)[:, self.class_idx]
        return lambda logic: logic.OR(
            logic.LT(uniq_dst_ports, torch.full_like(uniq_dst_ports, self.portscan_specs["min_uniq_dst_ports"])),
            logic.AND(
                logic.LEQ(fail_ratio, torch.full_like(fail_ratio, self.portscan_specs["min_fail_ratio"])),
                logic.GEQ(pkts_per_port, torch.full_like(pkts_per_port, self.portscan_specs["max_pkts_per_port"])),
                logic.GEQ(scan_duration, torch.full_like(scan_duration, self.portscan_specs["max_scan_duration"])),
            ),
            logic.GEQ(p, torch.full_like(p, self.min_prob)),
        )

    @torch.no_grad()
    def debug_parts(self, N, x, x_adv):
        uniq_dst_ports = x[:, self.idx["uniq_dst_ports"]]
        fail_ratio = x[:, self.idx["fail_ratio"]]
        pkts_per_port = x[:, self.idx["pkts_per_port"]]
        scan_duration = x[:, self.idx["scan_duration"]]
        p = F.softmax(N(x_adv), dim=1)[:, self.class_idx]
        parts = {
            "min_uniq_dst_ports": uniq_dst_ports >= self.portscan_specs["min_uniq_dst_ports"],
            "min_fail_ratio": fail_ratio >= self.portscan_specs["min_fail_ratio"],
            "max_pkts_per_port": pkts_per_port <= self.portscan_specs["max_pkts_per_port"],
            "max_scan_duration": scan_duration <= self.portscan_specs["max_scan_duration"],
            "prediction_ok": p >= self.min_prob,
        }
        parts["scan_signal"] = parts["min_fail_ratio"] | parts["max_pkts_per_port"] | parts["max_scan_duration"]
        parts["antecedent_true"] = parts["min_uniq_dst_ports"] & parts["scan_signal"]
        return parts


def build_precondition(config: dict, device):
    cls = getattr(pml_preconditions, config["name"])
    return cls(device=device, **config.get("params", {}))


def pick_precondition_config(preconditions: dict, key: str) -> dict:
    return preconditions.get(key, preconditions["default"])


def build_constraints(
    feature_cols: list[str],
    labels: list[str],
    scaled_attack_specs: dict,
    preconditions: dict,
    device,
    scaler,
    scale_cols,
    frozen_features=None,
    min_prob: float = 0.8,
):
    idx = {name: i for i, name in enumerate(feature_cols)}
    frozen_indices = [idx[name] for name in (frozen_features or [])]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    dos_precondition = build_precondition(pick_precondition_config(preconditions, "dos_http_flood"), device)
    scan_precondition = build_precondition(pick_precondition_config(preconditions, "portscan"), device)
    dos_precondition = FrozenFeaturePrecondition(dos_precondition, frozen_indices)
    scan_precondition = FrozenFeaturePrecondition(scan_precondition, frozen_indices)

    target = label_to_idx["ATTACK"] if "ATTACK" in label_to_idx else None
    dos_target = target if target is not None else label_to_idx["DOS_HTTP_FLOOD"]
    scan_target = target if target is not None else label_to_idx["PORTSCAN"]

    return {
        "dos": TabularRuleConstraint(
            device,
            precondition=dos_precondition,
            postcondition=DoSHttpFloodPostcondition(
                idx=idx,
                class_idx=dos_target,
                dos_http_flood_specs=scaled_attack_specs["dos_http_flood"],
                validity_specs=scaled_attack_specs["validity"],
                scaler=scaler,
                scale_cols=scale_cols,
                min_prob=min_prob,
            ),
        ),
        "scan": TabularRuleConstraint(
            device,
            precondition=scan_precondition,
            postcondition=PortscanPostcondition(
                idx=idx,
                class_idx=scan_target,
                min_prob=min_prob,
                portscan_specs=scaled_attack_specs["portscan"],
            ),
        ),
        "idx": idx,
        "dos_class": dos_target,
        "scan_class": scan_target,
    }
