from __future__ import annotations

import torch
import property_driven_ml.logics as pml_logics

from specs import ATTACK_SPECS


# ============================================================
# HELPERS
# ============================================================

def get_feature_index_map(feature_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def scaled_threshold(theta_raw: float, feature_name: str, scaler, feature_names: list[str]) -> float:
    idx = feature_names.index(feature_name)
    return (theta_raw - scaler.mean_[idx]) / scaler.scale_[idx]


def feat(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1)


def col(x: torch.Tensor, feat_idx: dict[str, int], name: str) -> torch.Tensor:
    return feat(x)[:, feat_idx[name]]


def class_margin(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    target_logit = logits[:, target_idx]
    other_indices = [i for i in range(logits.shape[1]) if i != target_idx]
    if not other_indices:
        return target_logit
    max_other = logits[:, other_indices].max(dim=1).values
    return target_logit - max_other


def active_margin_loss(margin: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    loss: penalize negative margin for active samples
    sat: fraction of active samples with nonnegative margin
    """
    active = active.bool()

    if active.sum() == 0:
        zero = torch.zeros((), device=margin.device)
        one = torch.ones((), device=margin.device)
        return zero, one

    active_margin = margin[active]
    loss = torch.relu(-active_margin).mean()
    sat = (active_margin >= 0).float().mean()
    return loss, sat


# ============================================================
# COLLECTION
# ============================================================

class Dl2PropertyCollection:
    def __init__(self, rules):
        self.rules = rules
        self.logic = pml_logics.DL2()

    def compute_loss(self, model, x_batch):
        logits = model(x_batch)

        losses = []
        sats = []
        stats = {}

        for i, rule in enumerate(self.rules):
            loss_i, sat_i = rule(logits, x_batch)
            losses.append(loss_i)
            sats.append(sat_i)
            stats[f"constraint_{i}_loss"] = float(loss_i.item())
            stats[f"constraint_{i}_sat"] = float(sat_i.item())

        total_loss = torch.stack(losses).mean() if losses else torch.zeros((), device=logits.device)
        return total_loss, stats


# ============================================================
# RULE BUILDERS
# ============================================================

def build_dos_http_rule(feat_idx, target_idx, scaler, feature_names):
    min_duration = scaled_threshold(ATTACK_SPECS["dos_http_flood"]["min_duration"], "duration", scaler, feature_names)
    max_duration = scaled_threshold(ATTACK_SPECS["dos_http_flood"]["max_duration"], "duration", scaler, feature_names)
    max_valid_pkt_rate = scaled_threshold(ATTACK_SPECS["dos_http_flood"]["max_valid_pkt_rate"], "orig_pkt_rate", scaler, feature_names)
    max_time_elapsed = scaled_threshold(ATTACK_SPECS["dos_http_flood"]["max_time_elapsed"], "time_elapsed", scaler, feature_names)
    min_flood_rate = scaled_threshold(ATTACK_SPECS["dos_http_flood"]["min_flood_rate"], "flood_rate", scaler, feature_names)

    def rule(logits, x):
        valid_input = torch.isfinite(feat(x)).all(dim=1)
        valid_tcp = col(x, feat_idx, "valid_tcp_handshake_feature") > 0
        valid_http = col(x, feat_idx, "service") >= 0
        valid_duration = (
            (col(x, feat_idx, "duration") >= min_duration)
            & (col(x, feat_idx, "duration") <= max_duration)
        )
        valid_packet_size = (
            (col(x, feat_idx, "orig_bytes") >= 0)
            & (col(x, feat_idx, "resp_bytes") >= 0)
            & (col(x, feat_idx, "orig_pkts") > 0)
        )
        valid_iat = col(x, feat_idx, "orig_pkt_rate") <= max_valid_pkt_rate

        mal_time_elapsed = col(x, feat_idx, "time_elapsed") <= max_time_elapsed
        mal_flood_rate = col(x, feat_idx, "flood_rate") >= min_flood_rate

        active = (
            valid_input
            & valid_tcp
            & valid_http
            & valid_duration
            & valid_packet_size
            & valid_iat
            & (mal_time_elapsed | mal_flood_rate)
        )

        margin = class_margin(logits, target_idx)
        return active_margin_loss(margin, active)

    return rule


def build_portscan_rule(feat_idx, target_idx, scaler, feature_names):
    min_ports = scaled_threshold(ATTACK_SPECS["portscan"]["min_ports"], "uniq_dst_ports", scaler, feature_names)
    max_pkts_per_port = scaled_threshold(ATTACK_SPECS["portscan"]["max_pkts_per_port"], "pkts_per_port", scaler, feature_names)
    max_scan_duration = scaled_threshold(ATTACK_SPECS["portscan"]["max_scan_duration"], "scan_duration", scaler, feature_names)
    min_fail_ratio = scaled_threshold(ATTACK_SPECS["portscan"]["min_fail_ratio"], "fail_ratio", scaler, feature_names)

    def rule(logits, x):
        many_ports = col(x, feat_idx, "uniq_dst_ports") >= min_ports
        few_pkts = col(x, feat_idx, "pkts_per_port") <= max_pkts_per_port
        short_scan = col(x, feat_idx, "scan_duration") <= max_scan_duration
        high_fail = col(x, feat_idx, "fail_ratio") >= min_fail_ratio

        active = many_ports | few_pkts | short_scan | high_fail
        margin = class_margin(logits, target_idx)
        return active_margin_loss(margin, active)

    return rule


def build_dos_udp_rule(feat_idx, target_idx, scaler, feature_names):
    max_udp_duration = scaled_threshold(ATTACK_SPECS["dos_udp_flood"]["max_udp_duration"], "duration", scaler, feature_names)
    min_udp_conn_count = scaled_threshold(ATTACK_SPECS["dos_udp_flood"]["min_udp_conn_count"], "udp_conn_count", scaler, feature_names)
    min_udp_packets = scaled_threshold(ATTACK_SPECS["dos_udp_flood"]["min_udp_packets"], "udp_packets", scaler, feature_names)
    min_udp_rate = scaled_threshold(ATTACK_SPECS["dos_udp_flood"]["min_udp_rate"], "udp_rate", scaler, feature_names)
    max_unique_src_ips = scaled_threshold(ATTACK_SPECS["dos_udp_flood"]["max_unique_src_ips"], "unique_src_ips", scaler, feature_names)

    def rule(logits, x):
        is_udp = col(x, feat_idx, "proto") >= 0
        udp_elapsed = (
            (col(x, feat_idx, "duration") >= 0)
            & (col(x, feat_idx, "duration") <= max_udp_duration)
        )
        udp_conn = col(x, feat_idx, "udp_conn_count") >= min_udp_conn_count
        udp_packets = col(x, feat_idx, "udp_packets") >= min_udp_packets
        udp_rate = col(x, feat_idx, "udp_rate") >= min_udp_rate
        single_source = col(x, feat_idx, "unique_src_ips") <= max_unique_src_ips

        active = is_udp & udp_elapsed & udp_conn & udp_packets & udp_rate & single_source
        margin = class_margin(logits, target_idx)
        return active_margin_loss(margin, active)

    return rule


def build_ddos_udp_rule(feat_idx, target_idx, scaler, feature_names):
    max_udp_duration = scaled_threshold(ATTACK_SPECS["ddos_udp_flood"]["max_udp_duration"], "duration", scaler, feature_names)
    min_udp_conn_count = scaled_threshold(ATTACK_SPECS["ddos_udp_flood"]["min_udp_conn_count"], "udp_conn_count", scaler, feature_names)
    min_udp_packets = scaled_threshold(ATTACK_SPECS["ddos_udp_flood"]["min_udp_packets"], "udp_packets", scaler, feature_names)
    min_udp_rate = scaled_threshold(ATTACK_SPECS["ddos_udp_flood"]["min_udp_rate"], "udp_rate", scaler, feature_names)
    min_unique_src_ips = scaled_threshold(ATTACK_SPECS["ddos_udp_flood"]["min_unique_src_ips"], "unique_src_ips", scaler, feature_names)

    def rule(logits, x):
        is_udp = col(x, feat_idx, "proto") >= 0
        udp_elapsed = (
            (col(x, feat_idx, "duration") >= 0)
            & (col(x, feat_idx, "duration") <= max_udp_duration)
        )
        udp_conn = col(x, feat_idx, "udp_conn_count") >= min_udp_conn_count
        udp_packets = col(x, feat_idx, "udp_packets") >= min_udp_packets
        udp_rate = col(x, feat_idx, "udp_rate") >= min_udp_rate
        multi_source = col(x, feat_idx, "unique_src_ips") >= min_unique_src_ips

        active = is_udp & udp_elapsed & udp_conn & udp_packets & udp_rate & multi_source
        margin = class_margin(logits, target_idx)
        return active_margin_loss(margin, active)

    return rule


# ============================================================
# BUILDER
# ============================================================

def build_properties(
    device: torch.device,
    scaler,
    feature_names: list[str],
    label_encoder,
    logic=None,   # kept for factory compatibility
):
    feat_idx = get_feature_index_map(feature_names)
    rules = []

    if "DOS_HTTP_FLOOD" in label_encoder.classes_:
        target_idx = int(label_encoder.transform(["DOS_HTTP_FLOOD"])[0])
        rules.append(build_dos_http_rule(feat_idx, target_idx, scaler, feature_names))

    if "PORTSCAN" in label_encoder.classes_:
        target_idx = int(label_encoder.transform(["PORTSCAN"])[0])
        rules.append(build_portscan_rule(feat_idx, target_idx, scaler, feature_names))

    if "DOS_UDP_FLOOD" in label_encoder.classes_ and all(
        f in feature_names for f in ["duration", "udp_conn_count", "udp_packets", "udp_rate", "unique_src_ips"]
    ):
        target_idx = int(label_encoder.transform(["DOS_UDP_FLOOD"])[0])
        rules.append(build_dos_udp_rule(feat_idx, target_idx, scaler, feature_names))

    if "DDOS_UDP_FLOOD" in label_encoder.classes_ and all(
        f in feature_names for f in ["duration", "udp_conn_count", "udp_packets", "udp_rate", "unique_src_ips"]
    ):
        target_idx = int(label_encoder.transform(["DDOS_UDP_FLOOD"])[0])
        rules.append(build_ddos_udp_rule(feat_idx, target_idx, scaler, feature_names))

    return Dl2PropertyCollection(rules)