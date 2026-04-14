from __future__ import annotations

import torch
import property_driven_ml.logics as pml_logics
from property_driven_ml.constraints import Constraint
from property_driven_ml.constraints.preconditions import Precondition
from property_driven_ml.constraints.postconditions import Postcondition


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


class PropertyCollection:
    def __init__(self, constraints, logic=pml_logics.DL2()):
        self.constraints = constraints
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

        stats = {}
        for i, (loss_i, sat_i) in enumerate(zip(losses, sats)):
            stats[f"constraint_{i}_loss"] = float(loss_i.item())
            stats[f"constraint_{i}_sat"] = float(sat_i.item())

        return total_loss, stats


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

class DosUdpFloodRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        max_udp_duration: float,
        min_udp_conn_count: float,
        min_udp_packets: float,
        min_udp_rate: float,
        max_unique_src_ips: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.max_udp_duration = max_udp_duration
        self.min_udp_conn_count = min_udp_conn_count
        self.min_udp_packets = min_udp_packets
        self.min_udp_rate = min_udp_rate
        self.max_unique_src_ips = max_unique_src_ips

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
        single_source = col(x, self.feat_idx, "unique_src_ips") <= self.max_unique_src_ips

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            antecedent = logic.AND(
                is_udp,
                udp_conn,
                udp_elapsed,
                udp_packets,
                udp_rate,
                single_source,
            )
            return logic.IMPL(antecedent, dominates)

        return formula


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
# BUILDER
# ============================================================

def build_properties(
    device: torch.device,
    scaler,
    feature_names: list[str],
    label_encoder,
    logic: pml_logics.Logic = pml_logics.DL2(),
):
    feat_idx = get_feature_index_map(feature_names)

    thresholds = {
        "dos_http_flood": {
            "min_duration": 0.0,
            "max_duration": 60.0,
            "max_valid_pkt_rate": 10000.0,
            "max_time_elapsed": 1.0,
            "min_flood_rate": 500.0,
        },
        "portscan": {
            "min_ports": 10.0,
            "max_pkts_per_port": 3.0,
            "max_scan_duration": 2.0,
            "min_fail_ratio": 0.5,
        },
        "dos_udp_flood": {
            "max_udp_duration": 2.0,
            "min_udp_conn_count": 20.0,
            "min_udp_packets": 200.0,
            "min_udp_rate": 100.0,
            "max_unique_src_ips": 1.0,
        },
        "ddos_udp_flood": {
            "max_udp_duration": 2.0,
            "min_udp_conn_count": 50.0,
            "min_udp_packets": 500.0,
            "min_udp_rate": 200.0,
            "min_unique_src_ips": 5.0,
        },
    }

    def s(group: str, threshold_key: str, feature_name: str) -> float:
        return scaled_threshold(
            thresholds[group][threshold_key],
            feature_name,
            scaler,
            feature_names,
        )

    constraints = []

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

    if "DOS_UDP_FLOOD" in label_encoder.classes_:
        target_idx = int(label_encoder.transform(["DOS_UDP_FLOOD"])[0])
        constraints.append(
            SimpleConstraint(
                device,
                DosUdpFloodRule(
                    feat_idx=feat_idx,
                    target_idx=target_idx,
                    max_udp_duration=s("dos_udp_flood", "max_udp_duration", "duration"),
                    min_udp_conn_count=s("dos_udp_flood", "min_udp_conn_count", "udp_conn_count"),
                    min_udp_packets=s("dos_udp_flood", "min_udp_packets", "udp_packets"),
                    min_udp_rate=s("dos_udp_flood", "min_udp_rate", "udp_rate"),
                    max_unique_src_ips=s("dos_udp_flood", "max_unique_src_ips", "unique_src_ips"),
                ),
            )
        )

    if "DDOS_UDP_FLOOD" in label_encoder.classes_:
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

    return PropertyCollection(constraints, logic=logic)