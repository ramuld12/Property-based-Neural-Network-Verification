from __future__ import annotations

import torch
import property_driven_ml.logics as pml_logics

# from specs import ATTACK_SPECS


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


def active_margin_loss(margin: torch.Tensor, active: torch.Tensor, required_margin: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    loss: penalize negative margin for active samples
    sat: fraction of active samples with nonnegative margin
    active_frac: fraction of batch where the rule antecedent is active
    """
    active = active.bool()
    active_frac = active.float().mean()

    if active.sum() == 0:
        zero = torch.zeros((), device=margin.device)
        one = torch.ones((), device=margin.device)
        return zero, one, active_frac

    active_margin = margin[active]
    loss = torch.relu(required_margin - active_margin).mean()
    sat = (active_margin >= required_margin).float().mean()
    return loss, sat, active_frac


# ============================================================
# COLLECTION
# ============================================================

class Dl2PropertyCollection:
    def __init__(self, rules, rule_names, model_feature_indices=None):
        self.rules = rules
        self.rule_names = rule_names
        self.logic = pml_logics.DL2()
        self.model_feature_indices = model_feature_indices

    def get_model_input(self, x_property):
        return x_property[:, :, self.model_feature_indices]

    def compute_loss(self, model, x_property):
        x_model = self.get_model_input(x_property)
        logits = model(x_model)

        losses = []
        sats = []
        stats = {}

        for i, rule in enumerate(self.rules):
            out = rule(logits, x_property)

            if len(out) == 3:
                loss_i, sat_i, active_frac_i = out
                extra = {}
            else:
                loss_i, sat_i, active_frac_i, extra = out

            losses.append(loss_i)
            sats.append(sat_i)

            stats[f"{self.rule_names[i]}_loss"] = float(loss_i.item())
            stats[f"{self.rule_names[i]}_sat"] = float(sat_i.item())
            stats[f"{self.rule_names[i]}_active_frac"] = float(active_frac_i.item())

            for k, v in extra.items():
                stats[f"{self.rule_names[i]}_{k}"] = v

        total_loss = torch.stack(losses).mean() if losses else torch.zeros((), device=logits.device)
        total_sat = torch.stack(sats).mean() if sats else torch.ones((), device=logits.device)

        return total_loss, total_sat, stats

# ============================================================
# RULE BUILDERS
# ============================================================

def build_dos_http_rule(feat_idx, target_idx):
    def rule(logits, x):
        valid_input = col(x, feat_idx, "valid_input") == 1
        valid_tcp_handshake = col(x, feat_idx, "valid_tcp_handshake") == 1
        valid_http = col(x, feat_idx, "valid_http_conn") == 1
        valid_duration = col(x, feat_idx, "valid_duration") == 1
        valid_packet_size = col(x, feat_idx, "valid_packet_size") == 1
        valid_iat = col(x, feat_idx, "valid_iat") == 1
        mal_time_elapsed = col(x, feat_idx, "dos_http_mal_time_elapsed") == 1
        mal_flood_rate = col(x, feat_idx, "dos_http_mal_flood_rate") == 1

        active = (
            valid_input
            & valid_tcp_handshake
            & valid_http
            & valid_duration
            & valid_packet_size
            & valid_iat
            & (mal_time_elapsed | mal_flood_rate)
        )

        margin = class_margin(logits, target_idx)
        loss, sat, active_frac = active_margin_loss(margin, active)

        extra = {
            "valid_input_frac": valid_input.float().mean().item(),
            "valid_tcp_handshake_frac": valid_tcp_handshake.float().mean().item(),
            "valid_http_frac": valid_http.float().mean().item(),
            "valid_duration_frac": valid_duration.float().mean().item(),
            "valid_packet_size_frac": valid_packet_size.float().mean().item(),
            "valid_iat_frac": valid_iat.float().mean().item(),
            "mal_time_elapsed_frac": mal_time_elapsed.float().mean().item(),
            "mal_flood_rate_frac": mal_flood_rate.float().mean().item(),
        }

        return loss, sat, active_frac, extra

    return rule


def build_portscan_rule(feat_idx, target_idx):
    def rule(logits, x):
        many_ports = col(x, feat_idx, "portscan_many_ports") > 0.5
        few_pkts = col(x, feat_idx, "portscan_few_pkts_per_port") > 0.5
        short_scan = col(x, feat_idx, "portscan_short_duration") > 0.5
        high_fail = col(x, feat_idx, "portscan_high_fail_ratio") > 0.5

        active = many_ports | few_pkts | short_scan | high_fail

        margin = class_margin(logits, target_idx)
        loss, sat, active_frac = active_margin_loss(margin, active)

        extra = {
            "many_ports_frac": many_ports.float().mean().item(),
            "few_pkts_frac": few_pkts.float().mean().item(),
            "short_scan_frac": short_scan.float().mean().item(),
            "high_fail_frac": high_fail.float().mean().item(),
        }

        return loss, sat, active_frac, extra

    return rule


# def build_ddos_udp_rule(feat_idx, target_idx, scaler, feature_names):
#     min_udp_duration = scaled_threshold(
#         0.0,
#         "duration",
#         scaler,
#         feature_names,
#     )
#     max_udp_duration = scaled_threshold(
#         ATTACK_SPECS["ddos_udp_flood"]["max_udp_duration"],
#         "duration",
#         scaler,
#         feature_names,
#     )
#     min_udp_conn_count = scaled_threshold(
#         ATTACK_SPECS["ddos_udp_flood"]["min_udp_conn_count"],
#         "udp_conn_count",
#         scaler,
#         feature_names,
#     )
#     min_udp_packets = scaled_threshold(
#         ATTACK_SPECS["ddos_udp_flood"]["min_udp_packets"],
#         "udp_packets",
#         scaler,
#         feature_names,
#     )
#     min_udp_rate = scaled_threshold(
#         ATTACK_SPECS["ddos_udp_flood"]["min_udp_rate"],
#         "udp_rate",
#         scaler,
#         feature_names,
#     )
#     min_unique_src_ips = scaled_threshold(
#         ATTACK_SPECS["ddos_udp_flood"]["min_unique_src_ips"],
#         "unique_src_ips",
#         scaler,
#         feature_names,
#     )

#     def rule(logits, x):
#         is_udp = col(x, feat_idx, "is_udp") >= 0.5
#         udp_elapsed = (
#             (col(x, feat_idx, "duration") >= min_udp_duration)
#             & (col(x, feat_idx, "duration") <= max_udp_duration)
#         )
#         udp_conn = col(x, feat_idx, "udp_conn_count") >= min_udp_conn_count
#         udp_packets = col(x, feat_idx, "udp_packets") >= min_udp_packets
#         udp_rate = col(x, feat_idx, "udp_rate") >= min_udp_rate
#         multi_source = col(x, feat_idx, "unique_src_ips") >= min_unique_src_ips

#         active = is_udp & udp_elapsed & udp_conn & udp_packets & udp_rate & multi_source
#         margin = class_margin(logits, target_idx)
#         loss, sat, active_frac = active_margin_loss(margin, active)

#         extra = {
#             "is_udp_frac": float(is_udp.float().mean().item()),
#             "udp_elapsed_frac": float(udp_elapsed.float().mean().item()),
#             "udp_conn_frac": float(udp_conn.float().mean().item()),
#             "udp_packets_frac": float(udp_packets.float().mean().item()),
#             "udp_rate_frac": float(udp_rate.float().mean().item()),
#             "multi_source_frac": float(multi_source.float().mean().item()),
#         }

#         return loss, sat, active_frac, extra

#     return rule


# def build_ddos_syn_rule(feat_idx, target_idx, scaler, feature_names):
#     min_syn_duration = scaled_threshold(
#         0.0, 
#         "syn_duration", 
#         scaler, 
#         feature_names
#     )
#     max_syn_duration = scaled_threshold(
#         ATTACK_SPECS["ddos_syn_flood"]["max_syn_duration"],
#         "syn_duration",
#         scaler,
#         feature_names,
#     )
#     min_syn_conn_count = scaled_threshold(
#         ATTACK_SPECS["ddos_syn_flood"]["min_syn_conn_count"],
#         "syn_conn_count",
#         scaler,
#         feature_names,
#     )
#     min_syn_count = scaled_threshold(
#         ATTACK_SPECS["ddos_syn_flood"]["min_syn_count"],
#         "syn_count",
#         scaler,
#         feature_names,
#     )
#     min_syn_rate = scaled_threshold(
#         ATTACK_SPECS["ddos_syn_flood"]["min_syn_rate"],
#         "syn_rate",
#         scaler,
#         feature_names,
#     )
#     min_half_open_count = scaled_threshold(
#         ATTACK_SPECS["ddos_syn_flood"]["min_half_open_count"],
#         "half_open_count",
#         scaler,
#         feature_names,
#     )
#     min_source_ip_count = scaled_threshold(
#         ATTACK_SPECS["ddos_syn_flood"]["min_source_ip_count"],
#         "source_ip_count",
#         scaler,
#         feature_names,
#     )

    # def rule(logits, x):
    #     syn_elapsed = (
    #         (col(x, feat_idx, "syn_duration") >= min_syn_duration)
    #         & (col(x, feat_idx, "syn_duration") <= max_syn_duration)
    #     )
    #     syn_conn = col(x, feat_idx, "syn_conn_count") >= min_syn_conn_count
    #     syn_count = col(x, feat_idx, "syn_count") >= min_syn_count
    #     syn_rate = col(x, feat_idx, "syn_rate") >= min_syn_rate
    #     half_open = col(x, feat_idx, "half_open_count") >= min_half_open_count
    #     multi_source = col(x, feat_idx, "source_ip_count") >= min_source_ip_count

    #     active = syn_elapsed & syn_conn & syn_count & syn_rate & half_open & multi_source
    #     margin = class_margin(logits, target_idx)
    #     loss, sat, active_frac = active_margin_loss(margin, active)

    #     extra = {
    #         "syn_elapsed_frac": float(syn_elapsed.float().mean().item()),
    #         "syn_conn_frac": float(syn_conn.float().mean().item()),
    #         "syn_count_frac": float(syn_count.float().mean().item()),
    #         "syn_rate_frac": float(syn_rate.float().mean().item()),
    #         "half_open_frac": float(half_open.float().mean().item()),
    #         "multi_source_frac": float(multi_source.float().mean().item()),
    #     }

    #     return loss, sat, active_frac, extra

    return rule


# ============================================================
# BUILDER
# ============================================================

def build_properties(
    device: torch.device,
    scaler,
    feature_names: list[str],
    label_encoder,
    logic=None,
    model_feature_names: list[str] | None = None,
):
    feat_idx = get_feature_index_map(feature_names)

    model_feature_indices = [
        feature_names.index(name)
        for name in model_feature_names
    ]

    rules = []
    rule_names = []

    classes = set(label_encoder.classes_)

    # ============================================================
    # Binary case: BENIGN vs ATTACK
    # ============================================================
    if "ATTACK" in classes:
        attack_idx = int(label_encoder.transform(["ATTACK"])[0])

        rules.append(build_dos_http_rule(feat_idx, attack_idx))
        rule_names.append("DOS_HTTP_ATTACK")

        rules.append(build_portscan_rule(feat_idx, attack_idx))
        rule_names.append("PORTSCAN_ATTACK")

    # ============================================================
    # Multiclass case
    # ============================================================
    if "DOS_HTTP_FLOOD" in classes:
        target_idx = int(label_encoder.transform(["DOS_HTTP_FLOOD"])[0])
        rules.append(build_dos_http_rule(feat_idx, target_idx))
        rule_names.append("DOS_HTTP_FLOOD")

    if "PORTSCAN" in classes:
        target_idx = int(label_encoder.transform(["PORTSCAN"])[0])
        rules.append(build_portscan_rule(feat_idx, target_idx))
        rule_names.append("PORTSCAN")

    # if "DDOS_UDP_FLOOD" in label_encoder.classes_ and all(
    #     f in feature_names for f in ["duration", "udp_conn_count", "udp_packets", "udp_rate", "unique_src_ips"]
    # ):
    #     target_idx = int(label_encoder.transform(["DDOS_UDP_FLOOD"])[0])
    #     rules.append(build_ddos_udp_rule(feat_idx, target_idx, scaler, feature_names))
    #     rule_names.append("DDOS_UDP_FLOOD")

    # if "DDOS_SYN_FLOOD" in label_encoder.classes_ and all(
    #     f in feature_names for f in [
    #         "syn_duration",
    #         "syn_conn_count",
    #         "syn_count",
    #         "syn_rate",
    #         "half_open_count",
    #         "source_ip_count",
    #     ]
    # ):
    #     target_idx = int(label_encoder.transform(["DDOS_SYN_FLOOD"])[0])
    #     rules.append(build_ddos_syn_rule(feat_idx, target_idx, scaler, feature_names))
    #     rule_names.append("DDOS_SYN_FLOOD")

    return Dl2PropertyCollection(
        rules=rules,
        rule_names=rule_names,
        model_feature_indices=model_feature_indices,
    )