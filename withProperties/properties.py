from __future__ import annotations

import torch
import property_driven_ml.logics as pml_logics
from property_driven_ml.constraints import Constraint
from property_driven_ml.constraints.preconditions import Precondition
from property_driven_ml.constraints.postconditions import Postcondition


# ============================================================
# PRECONDITION
# ============================================================

class IdentityPrecondition(Precondition):
    def get_precondition(self, x: torch.Tensor):
        return x, x


class SimpleConstraint(Constraint):
    def __init__(self, device: torch.device, postcondition: Postcondition):
        super().__init__(device)
        self.precondition = IdentityPrecondition()
        self.postcondition = postcondition


# ============================================================
# HELPERS
# ============================================================

def get_feature_index_map(feature_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def scaled_threshold(theta_raw: float, feature_name: str, scaler, feature_names: list[str]) -> float:
    idx = feature_names.index(feature_name)
    return (theta_raw - scaler.mean_[idx]) / scaler.scale_[idx]


def target_dominates(probs: torch.Tensor, target_idx: int):
    """
    Returns:
        p_target >= max(other_probs)
    """
    p_target = probs[:, target_idx]
    other_indices = [i for i in range(probs.shape[1]) if i != target_idx]

    if len(other_indices) == 0:
        return torch.ones_like(p_target, dtype=torch.bool)

    max_other = probs[:, other_indices].max(dim=1).values
    return p_target >= max_other


# ============================================================
# VALIDITY / DOS_HTTP_FLOOD MASKS
# ============================================================

def valid_input_mask(x: torch.Tensor):
    feat = x.squeeze(1)
    return torch.isfinite(feat).all(dim=1)


def valid_tcp_handshake_mask(x: torch.Tensor, feat_idx: dict[str, int]):
    feat = x.squeeze(1)
    cond = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    if "orig_pkts" in feat_idx:
        cond = cond & (feat[:, feat_idx["orig_pkts"]] > 0)

    if "resp_pkts" in feat_idx:
        cond = cond & (feat[:, feat_idx["resp_pkts"]] > 0)

    return cond


def valid_http_connection_mask(x: torch.Tensor, feat_idx: dict[str, int]):
    if "service" not in feat_idx:
        return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    feat = x.squeeze(1)
    return feat[:, feat_idx["service"]] >= 0


def valid_duration_mask(x: torch.Tensor, feat_idx: dict[str, int], min_dur: float, max_dur: float):
    feat = x.squeeze(1)
    duration = feat[:, feat_idx["duration"]]
    return (duration >= min_dur) & (duration <= max_dur)


def valid_packet_size_mask(x: torch.Tensor, feat_idx: dict[str, int]):
    feat = x.squeeze(1)
    cond = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    if "orig_bytes" in feat_idx:
        cond = cond & (feat[:, feat_idx["orig_bytes"]] >= 0)
    if "resp_bytes" in feat_idx:
        cond = cond & (feat[:, feat_idx["resp_bytes"]] >= 0)
    if "orig_pkts" in feat_idx:
        cond = cond & (feat[:, feat_idx["orig_pkts"]] > 0)

    return cond


def valid_iat_mask(x: torch.Tensor, feat_idx: dict[str, int], max_pkt_rate: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["orig_pkt_rate"]] <= max_pkt_rate


def mal_time_elapsed_mask(x: torch.Tensor, feat_idx: dict[str, int], max_time_elapsed: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["time_elapsed"]] <= max_time_elapsed


def mal_flood_rate_mask(x: torch.Tensor, feat_idx: dict[str, int], min_flood_rate: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["flood_rate"]] >= min_flood_rate


# ============================================================
# PORTSCAN MASKS
# ============================================================

def many_ports_mask(x: torch.Tensor, feat_idx: dict[str, int], thr_ports: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["uniqDstPorts"]] >= thr_ports


def few_pkts_per_port_mask(x: torch.Tensor, feat_idx: dict[str, int], thr_pkts_per_port: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["pktsPerPort"]] <= thr_pkts_per_port


def scan_elapsed_mask(x: torch.Tensor, feat_idx: dict[str, int], thr_scan_duration: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["scanDuration"]] <= thr_scan_duration


def single_source_mask(x: torch.Tensor, feat_idx: dict[str, int], thr_src_ips: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["uniqSrcIPs"]] <= thr_src_ips


def scan_fail_mask(x: torch.Tensor, feat_idx: dict[str, int], thr_fail_ratio: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["failRatio"]] >= thr_fail_ratio


# ============================================================
# DOS_HTTP_FLOOD PROPERTY
# ============================================================

class DOSHTTPFlood_MainRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        thr_min_duration: float,
        thr_max_duration: float,
        thr_max_pkt_rate_valid_iat: float,
        thr_max_time_elapsed: float,
        thr_min_flood_rate: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.thr_min_duration = thr_min_duration
        self.thr_max_duration = thr_max_duration
        self.thr_max_pkt_rate_valid_iat = thr_max_pkt_rate_valid_iat
        self.thr_max_time_elapsed = thr_max_time_elapsed
        self.thr_min_flood_rate = thr_min_flood_rate

    def get_postcondition(self, N, x):
        logits = N(x)
        probs = torch.softmax(logits, dim=1)

        valid_input = valid_input_mask(x)
        valid_tcp = valid_tcp_handshake_mask(x, self.feat_idx)
        valid_http = valid_http_connection_mask(x, self.feat_idx)
        valid_duration = valid_duration_mask(
            x, self.feat_idx, self.thr_min_duration, self.thr_max_duration
        )
        valid_packet_size = valid_packet_size_mask(x, self.feat_idx)
        valid_iat = valid_iat_mask(
            x, self.feat_idx, self.thr_max_pkt_rate_valid_iat
        )

        mal_time_elapsed = mal_time_elapsed_mask(
            x, self.feat_idx, self.thr_max_time_elapsed
        )
        mal_flood_rate = mal_flood_rate_mask(
            x, self.feat_idx, self.thr_min_flood_rate
        )

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            validity = logic.AND(
                valid_input,
                valid_tcp,
                valid_http,
                valid_duration,
                valid_packet_size,
                valid_iat,
            )
            malicious_behavior = logic.OR(
                mal_time_elapsed,
                mal_flood_rate,
            )
            antecedent = logic.AND(validity, malicious_behavior)
            consequent = dominates
            return logic.IMPL(antecedent, consequent)

        return formula


# ============================================================
# PORTSCAN PROPERTY
# ============================================================

class PortScan_MainRule(Postcondition):
    """
    Your initial proposed OR-rule:
    (manyPorts OR fewPktsPerPort OR scanElapsed OR singleSource OR scanFail)
    => PortScan
    """
    def __init__(
        self,
        feat_idx: dict[str, int],
        target_idx: int,
        thr_ports: float,
        thr_pkts_per_port: float,
        thr_scan_duration: float,
        thr_src_ips: float,
        thr_fail_ratio: float,
    ):
        self.feat_idx = feat_idx
        self.target_idx = target_idx
        self.thr_ports = thr_ports
        self.thr_pkts_per_port = thr_pkts_per_port
        self.thr_scan_duration = thr_scan_duration
        self.thr_src_ips = thr_src_ips
        self.thr_fail_ratio = thr_fail_ratio

    def get_postcondition(self, N, x):
        logits = N(x)
        probs = torch.softmax(logits, dim=1)

        many_ports = many_ports_mask(x, self.feat_idx, self.thr_ports)
        few_pkts = few_pkts_per_port_mask(x, self.feat_idx, self.thr_pkts_per_port)
        short_scan = scan_elapsed_mask(x, self.feat_idx, self.thr_scan_duration)
        single_src = single_source_mask(x, self.feat_idx, self.thr_src_ips)
        high_fail = scan_fail_mask(x, self.feat_idx, self.thr_fail_ratio)

        dominates = target_dominates(probs, self.target_idx)

        def formula(logic):
            antecedent = logic.OR(
                many_ports,
                few_pkts,
                short_scan,
                single_src,
                high_fail,
            )
            consequent = dominates
            return logic.IMPL(antecedent, consequent)

        return formula


# ============================================================
# COLLECTION
# ============================================================

class PropertyCollection:
    def __init__(self, constraints, logic=None):
        self.constraints = constraints
        self.logic = logic or pml_logics.GoedelFuzzyLogic()

    def compute_loss(self, model, x_batch):
        losses = []
        sats = []

        for i, c in enumerate(self.constraints):
            loss_i, sat_i = c.eval(
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
        for i, (l, s) in enumerate(zip(losses, sats)):
            stats[f"constraint_{i}_loss"] = float(l.item())
            stats[f"constraint_{i}_sat"] = float(s.item())

        return total_loss, stats


# ============================================================
# BUILDER
# ============================================================

def build_properties(
    device: torch.device,
    scaler,
    feature_names: list[str],
    label_encoder,
):
    feat_idx = get_feature_index_map(feature_names)

    dos_idx = int(label_encoder.transform(["DOS_HTTP_FLOOD"])[0])
    portscan_idx = int(label_encoder.transform(["PORTSCAN"])[0])

    # DOS_HTTP_FLOOD thresholds
    THR_MIN_DURATION = 0.0
    THR_MAX_DURATION = 60.0
    THR_MAX_VALID_PKT_RATE = 10000.0
    THR_MAX_TIME_ELAPSED = 1.0
    THR_MIN_FLOOD_RATE = 500.0

    dos_thr_min_duration = scaled_threshold(THR_MIN_DURATION, "duration", scaler, feature_names)
    dos_thr_max_duration = scaled_threshold(THR_MAX_DURATION, "duration", scaler, feature_names)
    dos_thr_max_valid_pkt_rate = scaled_threshold(THR_MAX_VALID_PKT_RATE, "orig_pkt_rate", scaler, feature_names)
    dos_thr_max_time_elapsed = scaled_threshold(THR_MAX_TIME_ELAPSED, "time_elapsed", scaler, feature_names)
    dos_thr_min_flood_rate = scaled_threshold(THR_MIN_FLOOD_RATE, "flood_rate", scaler, feature_names)

    # PORTSCAN thresholds
    THR_PORTS = 10.0
    THR_PKTS_PER_PORT = 3.0
    THR_SCAN_DURATION = 2.0
    THR_SRC_IPS = 1.0
    THR_FAIL_RATIO = 0.5

    scan_thr_ports = scaled_threshold(THR_PORTS, "uniqDstPorts", scaler, feature_names)
    scan_thr_pkts_per_port = scaled_threshold(THR_PKTS_PER_PORT, "pktsPerPort", scaler, feature_names)
    scan_thr_scan_duration = scaled_threshold(THR_SCAN_DURATION, "scanDuration", scaler, feature_names)
    scan_thr_src_ips = scaled_threshold(THR_SRC_IPS, "uniqSrcIPs", scaler, feature_names)
    scan_thr_fail_ratio = scaled_threshold(THR_FAIL_RATIO, "failRatio", scaler, feature_names)

    constraints = [
        SimpleConstraint(
            device,
            DOSHTTPFlood_MainRule(
                feat_idx=feat_idx,
                target_idx=dos_idx,
                thr_min_duration=dos_thr_min_duration,
                thr_max_duration=dos_thr_max_duration,
                thr_max_pkt_rate_valid_iat=dos_thr_max_valid_pkt_rate,
                thr_max_time_elapsed=dos_thr_max_time_elapsed,
                thr_min_flood_rate=dos_thr_min_flood_rate,
            ),
        ),
        SimpleConstraint(
            device,
            PortScan_MainRule(
                feat_idx=feat_idx,
                target_idx=portscan_idx,
                thr_ports=scan_thr_ports,
                thr_pkts_per_port=scan_thr_pkts_per_port,
                thr_scan_duration=scan_thr_scan_duration,
                thr_src_ips=scan_thr_src_ips,
                thr_fail_ratio=scan_thr_fail_ratio,
            ),
        ),
    ]

    return PropertyCollection(constraints)