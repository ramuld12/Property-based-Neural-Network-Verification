ATTACK_SPECS = {
    "validity": {
        "valid_packet_size_min_pkts": 1.0,
        "valid_packet_size_min_avg_bytes": 40.0,
        "valid_packet_size_min_total_bytes": 40.0,
    },
    "dos_http_flood": {
        "mal_time_elapsed_min": 0.0,
        "mal_time_elapsed_max": 120.0,
        "valid_pkt_size_total_min": 300,
        "mal_byte_rate_min": 250.0,
        "mal_pkt_rate_min": 50.0,
    },
    "portscan": {
        "min_uniq_dst_ports": 10.0,
        "max_pkts_per_port": 3.0,
        "max_scan_duration": 2.0,
        "min_fail_ratio": 0.5,
    },
    "ddos_udp_flood": {
        "max_udp_duration": 10.0,
        "min_udp_conn_count": 10.0,
        "min_udp_packets": 50.0,
        "min_udp_rate": 10.0,
        "min_unique_src_ips": 1.0,
    },
    "ddos_syn_flood": {
        "max_syn_duration": 10.0,
        "min_syn_conn_count": 5.0,
        "min_syn_count": 5.0,
        "min_syn_rate": 1.0,
        "min_half_open_count": 1.0,
        "min_source_ip_count": 1.0,
    },
}