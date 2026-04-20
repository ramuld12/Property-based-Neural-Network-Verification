ATTACK_SPECS = {
    "dos_http_flood": {
        "min_duration": 0.0,
        "max_duration": 300.0,
        "max_valid_pkt_rate": 50000.0,
        "max_time_elapsed": 10.0,
        "min_flood_rate": 10.0,
    },
    "portscan": {
        "min_ports": 10.0,
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