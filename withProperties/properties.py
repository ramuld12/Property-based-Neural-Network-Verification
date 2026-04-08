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


# ============================================================
# VALIDITY MASKS
# ============================================================

def valid_input_mask(x: torch.Tensor):
    feat = x.squeeze(1)
    return torch.isfinite(feat).all(dim=1)


def valid_tcp_handshake_mask(x: torch.Tensor, feat_idx: dict[str, int]):
    feat = x.squeeze(1)

    proto_ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if "proto" in feat_idx:
        # After ordinal encoding, TCP should be a known non-negative value.
        proto_ok = feat[:, feat_idx["proto"]] >= 0

    history_ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if "history" in feat_idx:
        history_ok = feat[:, feat_idx["history"]] >= 0

    conn_state_ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if "conn_state" in feat_idx:
        conn_state_ok = feat[:, feat_idx["conn_state"]] >= 0

    orig_pkts_ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if "orig_pkts" in feat_idx:
        orig_pkts_ok = feat[:, feat_idx["orig_pkts"]] > 0

    resp_pkts_ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if "resp_pkts" in feat_idx:
        resp_pkts_ok = feat[:, feat_idx["resp_pkts"]] > 0

    return proto_ok & history_ok & conn_state_ok & orig_pkts_ok & resp_pkts_ok


def valid_http_connection_mask(x: torch.Tensor, feat_idx: dict[str, int]):
    feat = x.squeeze(1)

    if "service" not in feat_idx:
        return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    # service has been ordinal encoded; known values are >= 0
    return feat[:, feat_idx["service"]] >= 0


def valid_duration_mask(x: torch.Tensor, feat_idx: dict[str, int], min_dur: float, max_dur: float):
    feat = x.squeeze(1)
    duration = feat[:, feat_idx["duration"]]
    return (duration >= min_dur) & (duration <= max_dur)


def valid_packet_size_mask(x: torch.Tensor, feat_idx: dict[str, int], min_orig_bytes: float = None):
    feat = x.squeeze(1)

    cond = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    if "orig_bytes" in feat_idx:
        cond = cond & (feat[:, feat_idx["orig_bytes"]] >= 0)

    if "resp_bytes" in feat_idx:
        cond = cond & (feat[:, feat_idx["resp_bytes"]] >= 0)

    if "orig_pkts" in feat_idx:
        cond = cond & (feat[:, feat_idx["orig_pkts"]] > 0)

    if min_orig_bytes is not None and "orig_bytes" in feat_idx:
        cond = cond & (feat[:, feat_idx["orig_bytes"]] >= min_orig_bytes)

    return cond


def valid_iat_mask(x: torch.Tensor, feat_idx: dict[str, int], max_pkt_rate: float):
    """
    Proxy for IAT validity:
    if packet rate is absurdly large, IAT is implausibly small.
    """
    feat = x.squeeze(1)
    return feat[:, feat_idx["orig_pkt_rate"]] <= max_pkt_rate


def mal_time_elapsed_mask(x: torch.Tensor, feat_idx: dict[str, int], max_time_elapsed: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["time_elapsed"]] <= max_time_elapsed


def mal_flood_rate_mask(x: torch.Tensor, feat_idx: dict[str, int], min_flood_rate: float):
    feat = x.squeeze(1)
    return feat[:, feat_idx["flood_rate"]] >= min_flood_rate


# ============================================================
# MAIN PROPERTY:
# (ValidInput ∧ ValidTCPHandshake ∧ ValidHTTPConnection
#  ∧ ValidDuration ∧ ValidPacketSize ∧ ValidIAT)
# ∧ (malTimeElapsed ∨ malFloodRate)
# => DOS_HTTP_FLOOD
# ============================================================

class DOSHTTPFlood_MainRule(Postcondition):
    def __init__(
        self,
        feat_idx: dict[str, int],
        thr_min_duration: float,
        thr_max_duration: float,
        thr_max_pkt_rate_valid_iat: float,
        thr_max_time_elapsed: float,
        thr_min_flood_rate: float,
        thr_min_orig_bytes_valid: float | None = None,
    ):
        self.feat_idx = feat_idx
        self.thr_min_duration = thr_min_duration
        self.thr_max_duration = thr_max_duration
        self.thr_max_pkt_rate_valid_iat = thr_max_pkt_rate_valid_iat
        self.thr_max_time_elapsed = thr_max_time_elapsed
        self.thr_min_flood_rate = thr_min_flood_rate
        self.thr_min_orig_bytes_valid = thr_min_orig_bytes_valid

    def get_postcondition(self, N, x):
        logits = N(x).view(-1)
        p_dos = torch.sigmoid(logits)

        valid_input = valid_input_mask(x)
        valid_tcp = valid_tcp_handshake_mask(x, self.feat_idx)
        valid_http = valid_http_connection_mask(x, self.feat_idx)
        valid_duration = valid_duration_mask(
            x, self.feat_idx, self.thr_min_duration, self.thr_max_duration
        )
        valid_packet_size = valid_packet_size_mask(
            x, self.feat_idx, self.thr_min_orig_bytes_valid
        )
        valid_iat = valid_iat_mask(
            x, self.feat_idx, self.thr_max_pkt_rate_valid_iat
        )

        mal_time_elapsed = mal_time_elapsed_mask(
            x, self.feat_idx, self.thr_max_time_elapsed
        )
        mal_flood_rate = mal_flood_rate_mask(
            x, self.feat_idx, self.thr_min_flood_rate
        )

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
            consequent = logic.GEQ(p_dos, torch.full_like(p_dos, 0.5))
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
):
    feat_idx = get_feature_index_map(feature_names)

    # Raw thresholds
    THR_MIN_DURATION = 0.0
    THR_MAX_DURATION = 60.0
    THR_MAX_VALID_PKT_RATE = 10000.0
    THR_MAX_TIME_ELAPSED = 1.0
    THR_MIN_FLOOD_RATE = 500.0
    THR_MIN_ORIG_BYTES_VALID = 0.0

    # Scale only thresholds tied to scaled features
    thr_min_duration = scaled_threshold(THR_MIN_DURATION, "duration", scaler, feature_names)
    thr_max_duration = scaled_threshold(THR_MAX_DURATION, "duration", scaler, feature_names)
    thr_max_valid_pkt_rate = scaled_threshold(THR_MAX_VALID_PKT_RATE, "orig_pkt_rate", scaler, feature_names)
    thr_max_time_elapsed = scaled_threshold(THR_MAX_TIME_ELAPSED, "time_elapsed", scaler, feature_names)
    thr_min_flood_rate = scaled_threshold(THR_MIN_FLOOD_RATE, "flood_rate", scaler, feature_names)
    thr_min_orig_bytes_valid = scaled_threshold(THR_MIN_ORIG_BYTES_VALID, "orig_bytes", scaler, feature_names)

    constraints = [
        SimpleConstraint(
            device,
            DOSHTTPFlood_MainRule(
                feat_idx=feat_idx,
                thr_min_duration=thr_min_duration,
                thr_max_duration=thr_max_duration,
                thr_max_pkt_rate_valid_iat=thr_max_valid_pkt_rate,
                thr_max_time_elapsed=thr_max_time_elapsed,
                thr_min_flood_rate=thr_min_flood_rate,
                thr_min_orig_bytes_valid=thr_min_orig_bytes_valid,
            ),
        ),
    ]

    return PropertyCollection(constraints)