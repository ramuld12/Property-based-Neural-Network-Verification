from __future__ import annotations

import torch
from dataclasses import dataclass
import property_driven_ml.logics as pml_logics
from property_driven_ml.constraints import Constraint
from property_driven_ml.constraints.preconditions import Precondition
from property_driven_ml.constraints.postconditions import Postcondition


# ============================================================
# PRECONDITION (identity)
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
# HELPER: feature indexing + scaling
# ============================================================

def get_feature_index_map(feature_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def scaled_threshold(theta_raw: float, feature_name: str, scaler, feature_names: list[str]) -> float:
    idx = feature_names.index(feature_name)
    return (theta_raw - scaler.mean_[idx]) / scaler.scale_[idx]


# ============================================================
# ================= VALIDITY PREDICATES ======================
# ============================================================

def valid_input_mask(x: torch.Tensor):
    feat = x.squeeze(1)
    return torch.isfinite(feat).all(dim=1)


def valid_tcp_handshake_mask(x: torch.Tensor, feat_idx):
    # Approximation: TCP + some packet exchange
    feat = x.squeeze(1)

    # If you have proto encoded, adjust here
    # fallback: assume all flows are TCP (or refine later)
    has_packets = feat[:, feat_idx["orig_pkts"]] > 0

    return has_packets


def valid_http_connection_mask(x: torch.Tensor, feat_idx):
    # Approximation using service (must be encoded beforehand)
    # If no service column, return True
    if "service" not in feat_idx:
        return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    feat = x.squeeze(1)
    return feat[:, feat_idx["service"]] >= 0  # placeholder (refine later)


def valid_duration_mask(x: torch.Tensor, feat_idx, min_dur, max_dur):
    feat = x.squeeze(1)
    duration = feat[:, feat_idx["duration"]]
    return (duration >= min_dur) & (duration <= max_dur)


def valid_packet_size_mask(x: torch.Tensor, feat_idx):
    feat = x.squeeze(1)
    return (
        (feat[:, feat_idx["orig_bytes"]] >= 0)
        & (feat[:, feat_idx["resp_bytes"]] >= 0)
        & (feat[:, feat_idx["orig_pkts"]] > 0)
    )


def valid_iat_mask(x: torch.Tensor, feat_idx, max_rate):
    feat = x.squeeze(1)
    rate = feat[:, feat_idx["orig_pkt_rate"]]
    return rate <= max_rate


# ============================================================
# ================= DOS_HTTP_FLOOD PROPERTIES =================
# ============================================================

class DOSHTTPFlood_HighRequestPressure(Postcondition):
    """
    Valid TCP + HTTP + high packets + high rate => DOS
    """

    def __init__(
        self,
        feat_idx,
        thr_pkts,
        thr_rate,
    ):
        self.feat_idx = feat_idx
        self.thr_pkts = thr_pkts
        self.thr_rate = thr_rate

    def get_postcondition(self, N, x):
        logits = N(x).view(-1)
        p_dos = torch.sigmoid(logits)

        feat = x.squeeze(1)

        high_pkts = feat[:, self.feat_idx["orig_pkts"]] >= self.thr_pkts
        high_rate = feat[:, self.feat_idx["orig_pkt_rate"]] >= self.thr_rate

        def formula(logic):
            antecedent = logic.AND(high_pkts, high_rate)
            consequent = logic.GEQ(p_dos, torch.full_like(p_dos, 0.5))
            return logic.IMPL(antecedent, consequent)

        return formula


class DOSHTTPFlood_ShortBurst(Postcondition):
    """
    Valid TCP + HTTP + high rate + short duration => DOS
    """

    def __init__(
        self,
        feat_idx,
        thr_rate,
        thr_duration,
    ):
        self.feat_idx = feat_idx
        self.thr_rate = thr_rate
        self.thr_duration = thr_duration

    def get_postcondition(self, N, x):
        logits = N(x).view(-1)
        p_dos = torch.sigmoid(logits)

        feat = x.squeeze(1)

        high_rate = feat[:, self.feat_idx["orig_pkt_rate"]] >= self.thr_rate
        short_dur = feat[:, self.feat_idx["duration"]] <= self.thr_duration

        def formula(logic):
            antecedent = logic.AND(high_rate, short_dur)
            consequent = logic.GEQ(p_dos, torch.full_like(p_dos, 0.5))
            return logic.IMPL(antecedent, consequent)

        return formula


# ============================================================
# ================= PROPERTY COLLECTION ======================
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
# ================= BUILDER FUNCTION =========================
# ============================================================

def build_properties(
    device: torch.device,
    scaler,
    feature_names: list[str],
):
    """
    Central place to define ALL properties
    """

    feat_idx = get_feature_index_map(feature_names)

    # ===== thresholds (raw) =====
    THR_PKTS = 25.0
    THR_RATE = 40.0
    THR_DURATION = 2.0

    # ===== scaled thresholds =====
    thr_pkts = scaled_threshold(THR_PKTS, "orig_pkts", scaler, feature_names)
    thr_rate = scaled_threshold(THR_RATE, "orig_pkt_rate", scaler, feature_names)
    thr_duration = scaled_threshold(THR_DURATION, "duration", scaler, feature_names)

    constraints = [
        SimpleConstraint(
            device,
            DOSHTTPFlood_HighRequestPressure(
                feat_idx,
                thr_pkts,
                thr_rate,
            ),
        ),
        SimpleConstraint(
            device,
            DOSHTTPFlood_ShortBurst(
                feat_idx,
                thr_rate,
                thr_duration,
            ),
        ),
    ]

    return PropertyCollection(constraints)