from __future__ import annotations

"""Tabular property constraints for property-driven training."""

import torch
from property_driven_ml.constraints.constraints import Constraint
from property_driven_ml.constraints.postconditions import Postcondition
import property_driven_ml.constraints.preconditions as pml_preconditions


class TabularRuleConstraint(Constraint):
    def __init__(self, device, precondition, postcondition):
        super().__init__(device)
        self.precondition = precondition
        self.postcondition = postcondition


def valid_input_bounds(x):
    return ((x >= 0.0) & (x <= 1.0)).all(dim=1)


def raw_col(x_col, col, scaler, scale_idx):
    i = scale_idx[col]
    min_ = torch.tensor(scaler.data_min_[i], device=x_col.device, dtype=x_col.dtype)
    max_ = torch.tensor(scaler.data_max_[i], device=x_col.device, dtype=x_col.dtype)
    return x_col * (max_ - min_ + 1e-8) + min_


def scaled_col(raw_value, col, scaler, scale_idx):
    i = scale_idx[col]
    min_ = torch.tensor(scaler.data_min_[i], device=raw_value.device, dtype=raw_value.dtype)
    max_ = torch.tensor(scaler.data_max_[i], device=raw_value.device, dtype=raw_value.dtype)
    return (raw_value - min_) / (max_ - min_ + 1e-8)


def target_logits(N, x, class_idx, model_feature_count):
    logits = N(x[:, :model_feature_count])
    target_logit = logits[:, class_idx]
    other_logits = logits.clone()
    other_logits[:, class_idx] = -torch.inf
    return target_logit, other_logits.max(dim=1).values


def target_logit_wins(N, x, class_idx, model_feature_count):
    target_logit, max_other_logit = target_logits(N, x, class_idx, model_feature_count)
    return target_logit >= max_other_logit


def implies(logic, antecedent_violations, consequent):
    return logic.OR(*antecedent_violations, consequent)


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


class GlobalBoundsPrecondition:
    def __init__(self, device, lower_bound=0.0, upper_bound=1.0):
        self.device = device
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_bounds(self, x):
        lo = torch.full_like(x, self.lower_bound, device=self.device)
        hi = torch.full_like(x, self.upper_bound, device=self.device)
        return lo, hi

    def get_precondition(self, x):
        return self.get_bounds(x)


class ScaledFeatureMixin:
    def init_scaling(self, scaler, scale_idx):
        self.scaler = scaler
        self.scale_idx = scale_idx

    def raw_col(self, x_col, col):
        return raw_col(x_col, col, self.scaler, self.scale_idx)

    def scale_col(self, raw_value, col):
        return scaled_col(raw_value, col, self.scaler, self.scale_idx)

    def scaled_threshold(self, reference, raw_value, col):
        return self.scale_col(torch.full_like(reference, raw_value), col)


class DoSHttpFloodPostcondition(ScaledFeatureMixin, Postcondition):
    def __init__(
        self,
        idx,
        class_idx,
        dos_http_flood_specs,
        scaler,
        scale_idx,
        model_feature_count,
    ):
        self.idx = idx
        self.class_idx = class_idx
        self.dos_http_flood_specs = dos_http_flood_specs
        self.init_scaling(scaler, scale_idx)
        self.model_feature_count = model_feature_count

    def raw_time_elapsed(self, x):
        return self.raw_col(x[:, self.idx["time_elapsed"]], "time_elapsed")

    def build_postcondition(self, N, x, x_adv):
        valid_tcp = x[:, self.idx["valid_tcp_handshake"]]
        valid_http = x[:, self.idx["valid_http_conn"]]
        time_elapsed = x[:, self.idx["time_elapsed"]]
        scaled_orig_bytes = x_adv[:, self.idx["orig_bytes"]]
        raw_orig_bytes = self.raw_col(x_adv[:, self.idx["orig_bytes"]], "orig_bytes")
        raw_orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts").clamp_min(1e-8)
        orig_bytes_per_packet = raw_orig_bytes / raw_orig_pkts
        dos_logit, max_other_logit = target_logits(N, x_adv, self.class_idx, self.model_feature_count)
        return lambda logic: implies(
            logic,
            [
                logic.LT(orig_bytes_per_packet, torch.full_like(orig_bytes_per_packet, self.dos_http_flood_specs["valid_packet_size_individual_min"])),
                logic.LT(
                    scaled_orig_bytes,
                    self.scaled_threshold(scaled_orig_bytes, self.dos_http_flood_specs["valid_pkt_size_total_min"], "orig_bytes"),
                ),
                logic.LT(valid_tcp, torch.full_like(valid_tcp, 0.5)),
                logic.LT(valid_http, torch.full_like(valid_http, 0.5)),
                logic.LT(
                    time_elapsed,
                    self.scaled_threshold(time_elapsed, self.dos_http_flood_specs["mal_time_elapsed_min"], "time_elapsed"),
                ),
                logic.GT(
                    time_elapsed,
                    self.scaled_threshold(time_elapsed, self.dos_http_flood_specs["mal_time_elapsed_max"], "time_elapsed"),
                ),
            ],
            logic.GEQ(dos_logit, max_other_logit),
        )

    @torch.no_grad()
    def debug_parts(self, N, x, x_adv):
        valid_tcp = x[:, self.idx["valid_tcp_handshake"]]
        valid_http = x[:, self.idx["valid_http_conn"]]
        time_elapsed = self.raw_time_elapsed(x)
        raw_orig_bytes = self.raw_col(x_adv[:, self.idx["orig_bytes"]], "orig_bytes")
        raw_orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts").clamp_min(1e-8)
        orig_bytes_per_packet = raw_orig_bytes / raw_orig_pkts
        valid_sizes = (
            (orig_bytes_per_packet >= self.dos_http_flood_specs["valid_packet_size_individual_min"])
            & (raw_orig_pkts >= 1.0)
            & (raw_orig_bytes >= self.dos_http_flood_specs["valid_pkt_size_total_min"])
        )
        mal_elapsed_ok = (
            (time_elapsed >= self.dos_http_flood_specs["mal_time_elapsed_min"])
            & (time_elapsed <= self.dos_http_flood_specs["mal_time_elapsed_max"])
        )
        parts = {
            "ValidInput": valid_input_bounds(x_adv),
            "ValidSizes": valid_sizes,
            "ValidTCPHandshake": valid_tcp == 1,
            "ValidHTTPConn": valid_http == 1,
            "MalTimeElapsed": mal_elapsed_ok,
            "prediction_ok": target_logit_wins(N, x_adv, self.class_idx, self.model_feature_count),
        }
        parts["antecedent_true"] = (
            parts["ValidSizes"]
            & parts["ValidTCPHandshake"]
            & parts["ValidHTTPConn"]
            & parts["MalTimeElapsed"]
        )
        return parts


class PortscanPostcondition(ScaledFeatureMixin, Postcondition):
    def __init__(self, idx, class_idx, portscan_specs, scaler, scale_idx, model_feature_count):
        self.idx = idx
        self.class_idx = class_idx
        self.portscan_specs = portscan_specs
        self.init_scaling(scaler, scale_idx)
        self.model_feature_count = model_feature_count

    def round_ste(self, value):
        return value + (torch.round(value) - value).detach()

    def consistent_adv(self, x, x_adv):
        x_consistent = x_adv.clone()
        total_orig_pkts = self.raw_col(x[:, self.idx["total_orig_pkts"]], "total_orig_pkts")
        orig_pkts = self.raw_col(x[:, self.idx["orig_pkts"]], "orig_pkts")
        adv_orig_pkts = self.raw_col(x_adv[:, self.idx["orig_pkts"]], "orig_pkts")
        uniq_dst_ports = self.round_ste(self.raw_col(x[:, self.idx["uniq_dst_ports"]], "uniq_dst_ports")).clamp_min(1.0)
        adv_total_orig_pkts = (total_orig_pkts - orig_pkts + adv_orig_pkts).clamp_min(0.0)
        pkts_per_port = adv_total_orig_pkts / uniq_dst_ports.clamp_min(1e-8)

        ts = self.raw_col(x[:, self.idx["ts"]], "ts")
        window_min_ts = self.raw_col(x[:, self.idx["window_min_ts"]], "window_min_ts")
        max_flow_end_without_row = self.raw_col(
            x[:, self.idx["max_flow_end_without_current_row"]],
            "max_flow_end_without_current_row",
        )
        adv_duration = self.raw_col(x_adv[:, self.idx["duration"]], "duration")
        scan_duration = torch.maximum(ts + adv_duration, max_flow_end_without_row) - window_min_ts

        x_consistent[:, self.idx["uniq_dst_ports"]] = x[:, self.idx["uniq_dst_ports"]]
        x_consistent[:, self.idx["pkts_per_port"]] = self.scale_col(pkts_per_port, "pkts_per_port").clamp(0.0, 1.0)
        x_consistent[:, self.idx["scan_duration"]] = self.scale_col(scan_duration, "scan_duration").clamp(0.0, 1.0)
        x_consistent[:, self.idx["fail_ratio"]] = x[:, self.idx["fail_ratio"]]
        return x_consistent

    def adv_values(self, x, x_adv):
        x_consistent = self.consistent_adv(x, x_adv)
        uniq_dst_ports = self.round_ste(self.raw_col(x_consistent[:, self.idx["uniq_dst_ports"]], "uniq_dst_ports")).clamp_min(1.0)
        pkts_per_port = self.raw_col(x_consistent[:, self.idx["pkts_per_port"]], "pkts_per_port")
        scan_duration = self.raw_col(x_consistent[:, self.idx["scan_duration"]], "scan_duration")
        fail_ratio = self.raw_col(x_consistent[:, self.idx["fail_ratio"]], "fail_ratio")
        return uniq_dst_ports, fail_ratio, pkts_per_port, scan_duration

    def build_postcondition(self, N, x, x_adv):
        x_consistent = self.consistent_adv(x, x_adv)
        uniq_dst_ports, fail_ratio, pkts_per_port, scan_duration = self.adv_values(x, x_adv)
        scaled_uniq_dst_ports = self.scale_col(uniq_dst_ports, "uniq_dst_ports")
        scaled_fail_ratio = self.scale_col(fail_ratio, "fail_ratio")
        scaled_pkts_per_port = self.scale_col(pkts_per_port, "pkts_per_port")
        scaled_scan_duration = self.scale_col(scan_duration, "scan_duration")
        scan_logit, max_other_logit = target_logits(N, x_consistent, self.class_idx, self.model_feature_count)
        return lambda logic: implies(
            logic,
            [
                logic.LT(
                    scaled_uniq_dst_ports,
                    self.scaled_threshold(scaled_uniq_dst_ports, self.portscan_specs["mal_uniq_dst_ports_min"], "uniq_dst_ports"),
                ),
                logic.AND(
                    logic.LT(
                        scaled_fail_ratio,
                        self.scaled_threshold(scaled_fail_ratio, self.portscan_specs["mal_fail_ratio_min"], "fail_ratio"),
                    ),
                    logic.GT(
                        scaled_pkts_per_port,
                        self.scaled_threshold(scaled_pkts_per_port, self.portscan_specs["mal_pkts_per_port_max"], "pkts_per_port"),
                    ),
                    logic.GT(
                        scaled_scan_duration,
                        self.scaled_threshold(scaled_scan_duration, self.portscan_specs["mal_scan_duration_max"], "scan_duration"),
                    ),
                ),
            ],
            logic.GEQ(scan_logit, max_other_logit)
        )

    @torch.no_grad()
    def debug_parts(self, N, x, x_adv):
        x_consistent = self.consistent_adv(x, x_adv)
        uniq_dst_ports, fail_ratio, pkts_per_port, scan_duration = self.adv_values(x, x_adv)
        high_fail_ratio = fail_ratio >= self.portscan_specs["mal_fail_ratio_min"]
        low_pkts_per_port = pkts_per_port <= self.portscan_specs["mal_pkts_per_port_max"]
        short_scan_duration = scan_duration <= self.portscan_specs["mal_scan_duration_max"]
        parts = {
            "ValidInput": valid_input_bounds(x_consistent),
            "MalDstPorts": uniq_dst_ports >= self.portscan_specs["mal_uniq_dst_ports_min"],
            "MalFailRatio": high_fail_ratio,
            "MalPktsPerPort": low_pkts_per_port,
            "MalScanDuration": short_scan_duration,
            "prediction_ok": target_logit_wins(N, x_consistent, self.class_idx, self.model_feature_count),
        }
        parts["scan_signal"] = parts["MalFailRatio"] | parts["MalPktsPerPort"] | parts["MalScanDuration"]
        parts["antecedent_true"] = parts["MalDstPorts"] & parts["scan_signal"]
        return parts


def build_precondition(config: dict, device):
    if config["name"] == "GlobalBounds" and not hasattr(pml_preconditions, "GlobalBounds"):
        return GlobalBoundsPrecondition(device=device, **config.get("params", {}))
    cls = getattr(pml_preconditions, config["name"])
    return cls(device=device, **config.get("params", {}))


def pick_precondition_config(preconditions: dict, key: str) -> dict:
    return preconditions.get(key, preconditions["default"])


def build_constraints(
    feature_cols: list[str],
    labels: list[str],
    attack_specs: dict,
    preconditions: dict,
    device,
    scaler,
    scale_cols,
    model_feature_count,
    frozen_feature_names=None,
):
    idx = {name: i for i, name in enumerate(feature_cols)}
    scale_idx = {name: i for i, name in enumerate(scale_cols)}
    unknown_features = sorted(set(frozen_feature_names or []) - set(idx))
    if unknown_features:
        raise ValueError(f"Unknown frozen_features: {unknown_features}")
    frozen_indices = sorted(set([idx[name] for name in (frozen_feature_names or [])] + list(range(model_feature_count, len(feature_cols)))))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    dos_precondition = build_precondition(pick_precondition_config(preconditions, "dos_http_flood"), device)
    scan_precondition = build_precondition(pick_precondition_config(preconditions, "portscan"), device)
    dos_precondition = FrozenFeaturePrecondition(dos_precondition, frozen_indices)
    scan_precondition = FrozenFeaturePrecondition(scan_precondition, frozen_indices)

    attack_target = label_to_idx["ATTACK"] if "ATTACK" in label_to_idx else None
    dos_target = label_to_idx.get("DOS_HTTP_FLOOD", attack_target)
    scan_target = label_to_idx.get("PORTSCAN", attack_target)
    if dos_target is None:
        raise ValueError("labels must include DOS_HTTP_FLOOD or ATTACK for DoS constraints")
    if scan_target is None:
        raise ValueError("labels must include PORTSCAN or ATTACK for Portscan constraints")

    return {
        "dos": TabularRuleConstraint(
            device,
            precondition=dos_precondition,
            postcondition=DoSHttpFloodPostcondition(
                idx=idx,
                class_idx=dos_target,
                dos_http_flood_specs=attack_specs["dos_http_flood"],
                scaler=scaler,
                scale_idx=scale_idx,
                model_feature_count=model_feature_count,
            ),
        ),
        "scan": TabularRuleConstraint(
            device,
            precondition=scan_precondition,
            postcondition=PortscanPostcondition(
                idx=idx,
                class_idx=scan_target,
                portscan_specs=attack_specs["portscan"],
                scaler=scaler,
                scale_idx=scale_idx,
                model_feature_count=model_feature_count,
            ),
        ),
        "idx": idx,
        "dos_class": dos_target,
        "scan_class": scan_target,
    }
