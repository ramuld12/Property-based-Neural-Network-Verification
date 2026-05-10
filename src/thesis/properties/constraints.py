from __future__ import annotations

"""Tabular property constraints for property-driven training."""

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
        scaler,
        scale_cols,
        model_feature_count,
        min_prob=0.80,
    ):
        self.idx = idx
        self.class_idx = class_idx
        self.min_prob = min_prob
        self.dos_http_flood_specs = dos_http_flood_specs
        self.scaler = scaler
        self.scale_cols = scale_cols
        self.model_feature_count = model_feature_count

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
        raw_orig_bytes = self.raw_col(x_adv[:, self.idx["orig_bytes"]], "orig_bytes")
        raw_orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts").clamp_min(1e-8)
        orig_bytes_per_packet = raw_orig_bytes / raw_orig_pkts
        orig_byte_rate, orig_pkt_rate = self.rates_from_adv(x_adv)
        # p = F.softmax(N(x_adv[:, : self.model_feature_count]), dim=1)[:, self.class_idx]
        logits = N(x_adv[:, : self.model_feature_count])
        dos_logit = logits[:, self.class_idx]

        other_logits = logits.clone()
        other_logits[:, self.class_idx] = -torch.inf
        max_other_logit = other_logits.max(dim=1).values
        return lambda logic: logic.OR(
            logic.LT(orig_bytes_per_packet, torch.full_like(orig_bytes_per_packet, self.dos_http_flood_specs["valid_packet_size_individual_min"])),
            logic.LT(orig_bytes, torch.full_like(orig_bytes, self.dos_http_flood_specs["valid_pkt_size_total_min"])),
            logic.NEQ(valid_tcp, torch.ones_like(valid_tcp)),
            logic.NEQ(valid_http, torch.ones_like(valid_http)),
            logic.LT(time_elapsed, torch.full_like(time_elapsed, self.dos_http_flood_specs["mal_time_elapsed_min"])),
            logic.GT(time_elapsed, torch.full_like(time_elapsed, self.dos_http_flood_specs["mal_time_elapsed_max"])),
            logic.OR(
                logic.LT(orig_byte_rate, torch.full_like(orig_byte_rate, self.dos_http_flood_specs["mal_byte_rate_min"])),
                logic.LT(orig_pkt_rate, torch.full_like(orig_pkt_rate, self.dos_http_flood_specs["mal_pkt_rate_min"])),
            ),

            # DOS_HTTP_FLOOD /prediction certainty
            # logic.GEQ(p, torch.full_like(p, self.min_prob))
            # Otherwise, model must predict DOS_HTTP_FLOOD
            logic.GEQ(dos_logit, max_other_logit),
        )

    @torch.no_grad()
    def debug_parts(self, N, x, x_adv):
        valid_tcp = x[:, self.idx["valid_tcp_handshake"]]
        valid_http = x[:, self.idx["valid_http_conn"]]
        time_elapsed = x[:, self.idx["time_elapsed"]]
        orig_bytes = x_adv[:, self.idx["orig_bytes"]]
        raw_orig_bytes = self.raw_col(x_adv[:, self.idx["orig_bytes"]], "orig_bytes")
        raw_orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts").clamp_min(1e-8)
        orig_bytes_per_packet = raw_orig_bytes / raw_orig_pkts
        orig_byte_rate, orig_pkt_rate = self.rates_from_adv(x_adv)
        # p = F.softmax(N(x_adv[:, : self.model_feature_count]), dim=1)[:, self.class_idx]
        logits = N(x_adv[:, : self.model_feature_count])
        dos_logit = logits[:, self.class_idx]

        other_logits = logits.clone()
        other_logits[:, self.class_idx] = -torch.inf
        max_other_logit = other_logits.max(dim=1).values
        parts = {
            "valid_input": (orig_bytes_per_packet >= self.dos_http_flood_specs["valid_packet_size_individual_min"])
            & (raw_orig_pkts >= 1.0),
            "valid_tcp_handshake": valid_tcp == 1,
            "valid_http_conn": valid_http == 1,
            "mal_time_elapsed_min": time_elapsed >= self.dos_http_flood_specs["mal_time_elapsed_min"],
            "mal_time_elapsed_max": time_elapsed <= self.dos_http_flood_specs["mal_time_elapsed_max"],
            "valid_pkt_size_total_min": orig_bytes >= self.dos_http_flood_specs["valid_pkt_size_total_min"],
            "mal_byte_rate_min": orig_byte_rate >= self.dos_http_flood_specs["mal_byte_rate_min"],
            "mal_pkt_rate_min": orig_pkt_rate >= self.dos_http_flood_specs["mal_pkt_rate_min"],
            "prediction_ok": dos_logit >= max_other_logit,
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
    def __init__(self, idx, class_idx, min_prob, portscan_specs, scaler, scale_cols, model_feature_count):
        self.idx = idx
        self.class_idx = class_idx
        self.min_prob = min_prob
        self.portscan_specs = portscan_specs
        self.scaler = scaler
        self.scale_cols = scale_cols
        self.model_feature_count = model_feature_count

    def raw_col(self, x_col, col):
        i = self.scale_cols.index(col)
        min_ = torch.tensor(self.scaler.data_min_[i], device=x_col.device, dtype=x_col.dtype)
        max_ = torch.tensor(self.scaler.data_max_[i], device=x_col.device, dtype=x_col.dtype)
        return x_col * (max_ - min_ + 1e-8) + min_

    def scale_col(self, raw_col, col):
        i = self.scale_cols.index(col)
        min_ = torch.tensor(self.scaler.data_min_[i], device=raw_col.device, dtype=raw_col.dtype)
        max_ = torch.tensor(self.scaler.data_max_[i], device=raw_col.device, dtype=raw_col.dtype)
        return (raw_col - min_) / (max_ - min_ + 1e-8)

    def round_ste(self, value):
        return value + (torch.round(value) - value).detach()

    def adv_values(self, x, x_adv):
        total_orig_pkts = self.raw_col(x[:, self.idx["total_orig_pkts"]], "total_orig_pkts")
        orig_pkts = self.raw_col(x[:, self.idx["orig_pkts"]], "orig_pkts")
        adv_orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts")
        uniq_dst_ports = self.round_ste(self.raw_col(x_adv[:, self.idx["uniq_dst_ports"]], "uniq_dst_ports")).clamp_min(1.0)
        adv_total_orig_pkts = (total_orig_pkts - orig_pkts + adv_orig_pkts).clamp_min(0.0)
        pkts_per_port = adv_total_orig_pkts / uniq_dst_ports.clamp_min(1e-8)
        max_duration_without_row = self.raw_col(
            x[:, self.idx["max_duration_without_current_row"]],
            "max_duration_without_current_row",
        )
        adv_duration = self.raw_col(x_adv[:, self.idx["duration"]], "duration")
        scan_duration = torch.maximum(adv_duration, max_duration_without_row)
        fail_ratio = self.raw_col(x_adv[:, self.idx["fail_ratio"]], "fail_ratio")
        corrected_adv = x_adv.clone()
        corrected_adv[:, self.idx["uniq_dst_ports"]] = self.scale_col(uniq_dst_ports, "uniq_dst_ports")
        corrected_adv[:, self.idx["scan_duration"]] = self.scale_col(scan_duration, "scan_duration")
        return uniq_dst_ports, fail_ratio, pkts_per_port, scan_duration, corrected_adv

    def get_postcondition(self, N, x, x_adv):
        uniq_dst_ports, fail_ratio, pkts_per_port, scan_duration, corrected_adv = self.adv_values(x, x_adv)
        # p = F.softmax(N(corrected_adv[:, : self.model_feature_count]), dim=1)[:, self.class_idx]
        logits = N(x_adv[:, : self.model_feature_count])
        scan_logit = logits[:, self.class_idx]

        other_logits = logits.clone()
        other_logits[:, self.class_idx] = -torch.inf
        max_other_logit = other_logits.max(dim=1).values
        return lambda logic: logic.OR(
            logic.LT(uniq_dst_ports, torch.full_like(uniq_dst_ports, self.portscan_specs["mal_uniq_dst_ports_min"])),
            logic.AND(
                logic.LEQ(fail_ratio, torch.full_like(fail_ratio, self.portscan_specs["mal_fail_ratio_min"])),
                logic.GT(pkts_per_port, torch.full_like(pkts_per_port, self.portscan_specs["mal_pkts_per_port_max"])),
                logic.GEQ(scan_duration, torch.full_like(scan_duration, self.portscan_specs["mal_scan_duration_max"])),
            ),
            logic.GEQ(scan_logit, max_other_logit)
        )

    @torch.no_grad()
    def debug_parts(self, N, x, x_adv):
        uniq_dst_ports, fail_ratio, pkts_per_port, scan_duration, corrected_adv = self.adv_values(x, x_adv)
        # p = F.softmax(N(corrected_adv[:, : self.model_feature_count]), dim=1)[:, self.class_idx]
        logits = N(x_adv[:, : self.model_feature_count])
        scan_logit = logits[:, self.class_idx]

        other_logits = logits.clone()
        other_logits[:, self.class_idx] = -torch.inf
        max_other_logit = other_logits.max(dim=1).values
        parts = {
            "mal_uniq_dst_ports_min": uniq_dst_ports >= self.portscan_specs["mal_uniq_dst_ports_min"],
            "mal_fail_ratio_min": fail_ratio >= self.portscan_specs["mal_fail_ratio_min"],
            "mal_pkts_per_port_max": pkts_per_port <= self.portscan_specs["mal_pkts_per_port_max"],
            "mal_scan_duration_max": scan_duration <= self.portscan_specs["mal_scan_duration_max"],
            "prediction_ok": scan_logit >= max_other_logit,
        }
        parts["scan_signal"] = parts["mal_fail_ratio_min"] | parts["mal_pkts_per_port_max"] | parts["mal_scan_duration_max"]
        parts["antecedent_true"] = parts["mal_uniq_dst_ports_min"] & parts["scan_signal"]
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
    model_feature_count,
    frozen_features=None,
    min_prob: float = 0.8,
):
    idx = {name: i for i, name in enumerate(feature_cols)}
    frozen_indices = sorted(set([idx[name] for name in (frozen_features or [])] + list(range(model_feature_count, len(feature_cols)))))
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
                scaler=scaler,
                scale_cols=scale_cols,
                model_feature_count=model_feature_count,
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
                scaler=scaler,
                scale_cols=scale_cols,
                model_feature_count=model_feature_count,
            ),
        ),
        "idx": idx,
        "dos_class": dos_target,
        "scan_class": scan_target,
    }
