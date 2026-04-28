ATTACK_SPECS = {
    "validity": {
        "valid_packet_size_min_pkts": 1.0,
        "valid_packet_size_min_avg_bytes": 40.0,
        "valid_packet_size_min_total_bytes": 40.0,
    },
    "dos_http_flood": {
        "valid_duration_min": 0.00000001,
        "valid_duration_max": 120.0,
        "valid_iat_max_pkt_rate": 20000.0,
        "mal_time_elapsed_max": 2.0,
        "mal_flood_rate_min": 250.0,
    },
    "portscan": {
        "many_ports_min": 10.0,
        "few_pkts_per_port_max": 3.0,
        "short_scan_duration_max": 2.0,
        "high_fail_ratio_min": 0.5,
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