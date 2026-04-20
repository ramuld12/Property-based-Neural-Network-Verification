from __future__ import annotations

import torch
import property_driven_ml.logics as pml_logics
from property_driven_ml.constraints import Constraint
from property_driven_ml.constraints.preconditions import Precondition
from property_driven_ml.constraints.postconditions import Postcondition

from specs import ATTACK_SPECS

# ============================================================
# BASE
# ============================================================

class IdentityPrecondition(Precondition):
    def get_precondition(self, x: torch.Tensor):
        return x, x


class SimpleConstraint(Constraint):
    def __init__(self, device: torch.device, postcondition: Postcondition):
        super().__init__(device)
        self.precondition = IdentityPrecondition()
        self.postcondition = postcondition


class GoedelPropertyCollection:
    def __init__(self, constraints, constraint_names, logic=pml_logics.GoedelFuzzyLogic()):
        self.constraints = constraints
        self.constraint_names = constraint_names
        self.logic = logic

    def compute_loss(self, model, x_batch):
        losses = []
        sats = []

        for i, constraint in enumerate(self.constraints):
            loss_i, sat_i = constraint.eval(
                N=model,
                x=x_batch,
                x_adv=None,
                y_target=None,
                logic=self.logic,
                reduction="mean",
            )
            losses.append(loss_i)
            sats.append(sat_i)

        total_loss = torch.stack(losses).mean()
        total_sat = torch.stack(sats).mean()

        stats = {}
        for i, (loss_i, sat_i) in enumerate(zip(losses, sats)):
            stats[f"{self.constraint_names[i]}_loss"] = float(loss_i.item())
            stats[f"{self.constraint_names[i]}_sat"] = float(sat_i.item())
            stats[f"{self.constraint_names[i]}_active_frac"] = float(
                self.constraints[i].postcondition.get_active_frac(x_batch)
            )

        return total_loss, total_sat, stats


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


def target_dominates(probs: torch.Tensor, target_idx: int) -> torch.Tensor:
    p_target = probs[:, target_idx]
    other_indices = [i for i in range(probs.shape[1]) if i != target_idx]

    if len(other_indices) == 0:
        return torch.ones_like(p_target, dtype=torch.bool)

    max_other = probs[:, other_indices].max(dim=1).values
    return p_target >= max_other


def finite_input(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(feat(x)).all(dim=1)


# ============================================================
# DOS_HTTP_FLOOD
# ============================================================

class DosHttpFloodRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        min_duration: float,
        max_duration: float,
        max_valid_pkt_rate: float,
        max_time_elapsed: float,
        min_flood_rate: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_valid_pkt_rate = max_valid_pkt_rate
        self.max_time_elapsed = max_time_elapsed
        self.min_flood_rate = min_flood_rate

    def get_active_frac(self, x):
        valid_input = finite_input(x)
        valid_tcp = col(x, self.feat_idx, "valid_tcp_handshake_feature") > 0
        valid_http = col(x, self.feat_idx, "service") >= 0
        valid_duration = (
            (col(x, self.feat_idx, "duration") >= self.min_duration)
            & (col(x, self.feat_idx, "duration") <= self.max_duration)
        )
        valid_packet_size = (
            (col(x, self.feat_idx, "orig_bytes") >= 0)
            & (col(x, self.feat_idx, "resp_bytes") >= 0)
            & (col(x, self.feat_idx, "orig_pkts") > 0)
        )
        valid_iat = col(x, self.feat_idx, "orig_pkt_rate") <= self.max_valid_pkt_rate

        mal_time_elapsed = col(x, self.feat_idx, "time_elapsed") <= self.max_time_elapsed
        mal_flood_rate = col(x, self.feat_idx, "flood_rate") >= self.min_flood_rate

        antecedent = (
            valid_input
            & valid_tcp
            & valid_http
            & valid_duration
            & valid_packet_size
            & valid_iat
            & (mal_time_elapsed | mal_flood_rate)
        )
        return antecedent.float().mean().item()

    def get_postcondition(self, N, x):
        probs = torch.softmax(N(x), dim=1)

        valid_input = finite_input(x)
        valid_tcp = col(x, self.feat_idx, "valid_tcp_handshake_feature") > 0
        valid_http = col(x, self.feat_idx, "service") >= 0
        valid_duration = (
            (col(x, self.feat_idx, "duration") >= self.min_duration)
            & (col(x, self.feat_idx, "duration") <= self.max_duration)
        )
        valid_packet_size = (
            (col(x, self.feat_idx, "orig_bytes") >= 0)
            & (col(x, self.feat_idx, "resp_bytes") >= 0)
            & (col(x, self.feat_idx, "orig_pkts") > 0)
        )
        valid_iat = col(x, self.feat_idx, "orig_pkt_rate") <= self.max_valid_pkt_rate

        mal_time_elapsed = col(x, self.feat_idx, "time_elapsed") <= self.max_time_elapsed
        mal_flood_rate = col(x, self.feat_idx, "flood_rate") >= self.min_flood_rate

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            antecedent = logic.AND(
                valid_input,
                valid_tcp,
                valid_http,
                valid_duration,
                valid_packet_size,
                valid_iat,
                logic.OR(mal_time_elapsed, mal_flood_rate),
            )
            return logic.IMPL(antecedent, dominates)

        return formula


# ============================================================
# PORTSCAN
# ============================================================

class PortScanRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        min_ports: float,
        max_pkts_per_port: float,
        max_scan_duration: float,
        min_fail_ratio: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.min_ports = min_ports
        self.max_pkts_per_port = max_pkts_per_port
        self.max_scan_duration = max_scan_duration
        self.min_fail_ratio = min_fail_ratio

    def get_active_frac(self, x):
        many_ports = col(x, self.feat_idx, "uniq_dst_ports") >= self.min_ports
        few_pkts = col(x, self.feat_idx, "pkts_per_port") <= self.max_pkts_per_port
        short_scan = col(x, self.feat_idx, "scan_duration") <= self.max_scan_duration
        high_fail = col(x, self.feat_idx, "fail_ratio") >= self.min_fail_ratio

        antecedent = many_ports | few_pkts | short_scan | high_fail
        return antecedent.float().mean().item()

    def get_postcondition(self, N, x):
        probs = torch.softmax(N(x), dim=1)

        many_ports = col(x, self.feat_idx, "uniq_dst_ports") >= self.min_ports
        few_pkts = col(x, self.feat_idx, "pkts_per_port") <= self.max_pkts_per_port
        short_scan = col(x, self.feat_idx, "scan_duration") <= self.max_scan_duration
        high_fail = col(x, self.feat_idx, "fail_ratio") >= self.min_fail_ratio

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            antecedent = logic.OR(
                many_ports,
                few_pkts,
                short_scan,
                high_fail,
            )
            return logic.IMPL(antecedent, dominates)

        return formula


# ============================================================
# UDP FLOOD
# ============================================================

class DdosUdpFloodRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        max_udp_duration: float,
        min_udp_conn_count: float,
        min_udp_packets: float,
        min_udp_rate: float,
        min_unique_src_ips: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.max_udp_duration = max_udp_duration
        self.min_udp_conn_count = min_udp_conn_count
        self.min_udp_packets = min_udp_packets
        self.min_udp_rate = min_udp_rate
        self.min_unique_src_ips = min_unique_src_ips

    def get_active_frac(self, x):
        is_udp = col(x, self.feat_idx, "proto") >= 0
        udp_elapsed = (
            (col(x, self.feat_idx, "duration") >= 0)
            & (col(x, self.feat_idx, "duration") <= self.max_udp_duration)
        )
        udp_conn = col(x, self.feat_idx, "udp_conn_count") >= self.min_udp_conn_count
        udp_packets = col(x, self.feat_idx, "udp_packets") >= self.min_udp_packets
        udp_rate = col(x, self.feat_idx, "udp_rate") >= self.min_udp_rate
        multi_source = col(x, self.feat_idx, "unique_src_ips") >= self.min_unique_src_ips

        antecedent = is_udp & udp_elapsed & udp_conn & udp_packets & udp_rate & multi_source
        return antecedent.float().mean().item()

    def get_postcondition(self, N, x):
        probs = torch.softmax(N(x), dim=1)

        is_udp = col(x, self.feat_idx, "proto") >= 0  # practical placeholder after encoding
        udp_elapsed = (
            (col(x, self.feat_idx, "duration") >= 0)
            & (col(x, self.feat_idx, "duration") <= self.max_udp_duration)
        )
        udp_conn = col(x, self.feat_idx, "udp_conn_count") >= self.min_udp_conn_count
        udp_packets = col(x, self.feat_idx, "udp_packets") >= self.min_udp_packets
        udp_rate = col(x, self.feat_idx, "udp_rate") >= self.min_udp_rate
        multi_source = col(x, self.feat_idx, "unique_src_ips") >= self.min_unique_src_ips

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            antecedent = logic.AND(
                is_udp,
                udp_conn,
                udp_elapsed,
                udp_packets,
                udp_rate,
                multi_source,
            )
            return logic.IMPL(antecedent, dominates)

        return formula


# ============================================================
# DDOS_SYN_FLOOD
# ============================================================

class DdosSynFloodRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        max_syn_duration: float,
        min_syn_conn_count: float,
        min_syn_count: float,
        min_syn_rate: float,
        min_half_open_count: float,
        min_source_ip_count: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.max_syn_duration = max_syn_duration
        self.min_syn_conn_count = min_syn_conn_count
        self.min_syn_count = min_syn_count
        self.min_syn_rate = min_syn_rate
        self.min_half_open_count = min_half_open_count
        self.min_source_ip_count = min_source_ip_count

    def get_active_frac(self, x):
        syn_elapsed = (
            (col(x, self.feat_idx, "syn_duration") >= 0)
            & (col(x, self.feat_idx, "syn_duration") <= self.max_syn_duration)
        )
        syn_conn = col(x, self.feat_idx, "syn_conn_count") >= self.min_syn_conn_count
        syn_count = col(x, self.feat_idx, "syn_count") >= self.min_syn_count
        syn_rate = col(x, self.feat_idx, "syn_rate") >= self.min_syn_rate
        half_open = col(x, self.feat_idx, "half_open_count") >= self.min_half_open_count
        multi_source = col(x, self.feat_idx, "source_ip_count") >= self.min_source_ip_count

        antecedent = multi_source & syn_elapsed & syn_conn & syn_count & syn_rate & half_open
        return antecedent.float().mean().item()

    def get_postcondition(self, N, x):
        probs = torch.softmax(N(x), dim=1)

        syn_elapsed = (
            (col(x, self.feat_idx, "syn_duration") >= 0)
            & (col(x, self.feat_idx, "syn_duration") <= self.max_syn_duration)
        )
        syn_conn = col(x, self.feat_idx, "syn_conn_count") >= self.min_syn_conn_count
        syn_count = col(x, self.feat_idx, "syn_count") >= self.min_syn_count
        syn_rate = col(x, self.feat_idx, "syn_rate") >= self.min_syn_rate
        half_open = col(x, self.feat_idx, "half_open_count") >= self.min_half_open_count
        multi_source = col(x, self.feat_idx, "source_ip_count") >= self.min_source_ip_count

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            antecedent = logic.AND(
                multi_source,
                syn_elapsed,
                syn_conn,
                syn_count,
                syn_rate,
                half_open,
            )
            return logic.IMPL(antecedent, dominates)

        return formula


# ============================================================
# BUILDER
# ============================================================

def build_properties(
    device: torch.device,
    scaler,
    feature_names: list[str],
    label_encoder
):
    feat_idx = get_feature_index_map(feature_names)

    def s(group: str, threshold_key: str, feature_name: str) -> float:
        return scaled_threshold(
            ATTACK_SPECS[group][threshold_key],
            feature_name,
            scaler,
            feature_names,
        )

    constraints = []
    constraint_names = []

    if "DOS_HTTP_FLOOD" in label_encoder.classes_:
        target_idx = int(label_encoder.transform(["DOS_HTTP_FLOOD"])[0])
        constraints.append(
            SimpleConstraint(
                device,
                DosHttpFloodRule(
                    feat_idx=feat_idx,
                    target_idx=target_idx,
                    min_duration=s("dos_http_flood", "min_duration", "duration"),
                    max_duration=s("dos_http_flood", "max_duration", "duration"),
                    max_valid_pkt_rate=s("dos_http_flood", "max_valid_pkt_rate", "orig_pkt_rate"),
                    max_time_elapsed=s("dos_http_flood", "max_time_elapsed", "time_elapsed"),
                    min_flood_rate=s("dos_http_flood", "min_flood_rate", "flood_rate"),
                ),
            )
        )
        constraint_names.append("DOS_HTTP_FLOOD")

    if "PORTSCAN" in label_encoder.classes_:
        target_idx = int(label_encoder.transform(["PORTSCAN"])[0])
        constraints.append(
            SimpleConstraint(
                device,
                PortScanRule(
                    feat_idx=feat_idx,
                    target_idx=target_idx,
                    min_ports=s("portscan", "min_ports", "uniq_dst_ports"),
                    max_pkts_per_port=s("portscan", "max_pkts_per_port", "pkts_per_port"),
                    max_scan_duration=s("portscan", "max_scan_duration", "scan_duration"),
                    min_fail_ratio=s("portscan", "min_fail_ratio", "fail_ratio"),
                ),
            )
        )
        constraint_names.append("PORTSCAN")

    if "DDOS_UDP_FLOOD" in label_encoder.classes_ and all(
        f in feature_names for f in ["duration", "udp_conn_count", "udp_packets", "udp_rate", "unique_src_ips"]
    ):
        target_idx = int(label_encoder.transform(["DDOS_UDP_FLOOD"])[0])
        constraints.append(
            SimpleConstraint(
                device,
                DdosUdpFloodRule(
                    feat_idx=feat_idx,
                    target_idx=target_idx,
                    max_udp_duration=s("ddos_udp_flood", "max_udp_duration", "duration"),
                    min_udp_conn_count=s("ddos_udp_flood", "min_udp_conn_count", "udp_conn_count"),
                    min_udp_packets=s("ddos_udp_flood", "min_udp_packets", "udp_packets"),
                    min_udp_rate=s("ddos_udp_flood", "min_udp_rate", "udp_rate"),
                    min_unique_src_ips=s("ddos_udp_flood", "min_unique_src_ips", "unique_src_ips"),
                ),
            )
        )
        constraint_names.append("DDOS_UDP_FLOOD")

    if "DDOS_SYN_FLOOD" in label_encoder.classes_ and all(
        f in feature_names for f in [
            "syn_duration",
            "syn_conn_count",
            "syn_count",
            "syn_rate",
            "half_open_count",
            "source_ip_count",
        ]
    ):
        target_idx = int(label_encoder.transform(["DDOS_SYN_FLOOD"])[0])
        constraints.append(
            SimpleConstraint(
                device,
                DdosSynFloodRule(
                    feat_idx=feat_idx,
                    target_idx=target_idx,
                    max_syn_duration=s("ddos_syn_flood", "max_syn_duration", "syn_duration"),
                    min_syn_conn_count=s("ddos_syn_flood", "min_syn_conn_count", "syn_conn_count"),
                    min_syn_count=s("ddos_syn_flood", "min_syn_count", "syn_count"),
                    min_syn_rate=s("ddos_syn_flood", "min_syn_rate", "syn_rate"),
                    min_half_open_count=s("ddos_syn_flood", "min_half_open_count", "half_open_count"),
                    min_source_ip_count=s("ddos_syn_flood", "min_source_ip_count", "source_ip_count"),
                ),
            )
        )
        constraint_names.append("DDOS_SYN_FLOOD")

    return GoedelPropertyCollection(constraints, constraint_names)