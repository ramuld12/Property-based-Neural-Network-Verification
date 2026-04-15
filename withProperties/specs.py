ATTACK_SPECS = {
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
    "ddos_udp_flood": {
        "max_udp_duration": 2.0,
        "min_udp_conn_count": 50.0,
        "min_udp_packets": 500.0,
        "min_udp_rate": 200.0,
        "min_unique_src_ips": 5.0,
    },
}