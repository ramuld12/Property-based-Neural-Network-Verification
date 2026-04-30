import torch
import torch.nn.functional as F

from property_driven_ml.constraints.constraints import Constraint
from property_driven_ml.constraints.postconditions import Postcondition
from property_driven_ml.constraints.preconditions import GlobalBounds


class TabularRuleConstraint(Constraint):
    def __init__(self, device, precondition, postcondition, lower_bound, upper_bound):
        super().__init__(device)
        self.precondition = precondition
        self.postcondition = postcondition


class DoSHttpFloodPostcondition(Postcondition):
    def __init__(
        self,
        idx,
        class_idx,
        dos_http_flood_specs,
        min_prob=0.80
    ):
        self.idx = idx
        self.class_idx = class_idx
        self.min_prob = min_prob
        self.dos_http_flood_specs = dos_http_flood_specs

    def get_postcondition(self, N, x, x_adv):
        valid_tcp_handshake = x_adv[:, self.idx["valid_tcp_handshake"]]
        valid_http_conn = x_adv[:, self.idx["valid_http_conn"]]

        time_elapsed = x_adv[:, self.idx["time_elapsed"]]

        orig_bytes = x_adv[:, self.idx["orig_bytes"]]
        orig_pkts = x_adv[:, self.idx["orig_pkts"]]

        orig_pkt_rate = x_adv[:, self.idx["orig_pkt_rate"]]
        orig_byte_rate = x_adv[:, self.idx["orig_byte_rate"]]

        p = F.softmax(N(x_adv), dim=1)[:, self.class_idx]

        return lambda logic: logic.IMPL(
            logic.AND(
                # validInput(x)
                logic.AND(
                    logic.GEQ(orig_bytes, torch.zeros_like(orig_bytes)),
                    logic.GEQ(orig_pkts, torch.zeros_like(orig_pkts)),
                ),

                # validTCPHandshake(x)
                logic.EQ(valid_tcp_handshake, torch.ones_like(valid_tcp_handshake)),

                # validHTTPConn(x)
                logic.EQ(valid_http_conn, torch.ones_like(valid_http_conn)),

                # NOT validTimeElapsed(x)
                logic.AND(
                    logic.LEQ(time_elapsed, torch.full_like(time_elapsed, self.dos_http_flood_specs["valid_time_elapsed_min"])),
                    logic.LEQ(time_elapsed, torch.full_like(time_elapsed, self.dos_http_flood_specs["valid_time_elapsed_max"])),
                ),

                # validSizes(x)
                logic.GEQ(orig_bytes, torch.full_like(orig_bytes, self.dos_http_flood_specs["valid_pkt_size_total_min"])),

                # malicious flood signal
                logic.OR(
                    logic.GEQ(orig_byte_rate, torch.full_like(orig_byte_rate, self.dos_http_flood_specs["mal_byte_rate_min"])),
                    logic.GEQ(orig_pkt_rate, torch.full_like(orig_pkt_rate, self.dos_http_flood_specs["mal_pkt_rate_min"])),
                ),
            ),
            logic.GEQ(p, torch.full_like(p, self.min_prob)),
        )


class PortscanPostcondition(Postcondition):
    def __init__(
        self,
        idx,
        class_idx,
        min_prob,
        portscan_specs
    ):
        self.idx = idx
        self.class_idx = class_idx
        self.min_prob = min_prob
        self.min_uniq_dst_ports = portscan_specs["min_uniq_dst_ports"]
        self.max_pkts_per_port = portscan_specs["max_pkts_per_port"]
        self.max_scan_duration = portscan_specs["max_scan_duration"]
        self.min_fail_ratio = portscan_specs["min_fail_ratio"]

    def get_postcondition(self, N, x, x_adv):
        uniq_dst_ports = x_adv[:, self.idx["uniq_dst_ports"]]
        fail_ratio = x_adv[:, self.idx["fail_ratio"]]
        pkts_per_port = x_adv[:, self.idx["pkts_per_port"]]
        short_scan_duration_max = x_adv[:, self.idx["scan_duration"]]
        p = F.softmax(N(x_adv), dim=1)[:, self.class_idx]

        return lambda logic: logic.IMPL(
            logic.AND(
                logic.GEQ(
                    uniq_dst_ports, torch.full_like(uniq_dst_ports, self.min_uniq_dst_ports),
                ),
                logic.OR(
                    logic.GEQ(
                        fail_ratio, torch.full_like(fail_ratio, self.min_fail_ratio),
                    ),
                    logic.LEQ(
                        pkts_per_port, torch.full_like(pkts_per_port, self.max_pkts_per_port),
                    ),
                    logic.LEQ(
                        short_scan_duration_max, torch.full_like(short_scan_duration_max, self.max_scan_duration),
                    ),   
                )
            ),
            logic.GEQ(p, torch.full_like(p, self.min_prob)),
        )