#!/usr/bin/env bash
set -euo pipefail

PCAP_ROOT="./pcaps/PCAP-03-11"
OUT_ROOT="./zeek_logs"

mkdir -p "$OUT_ROOT"

find "$PCAP_ROOT" -type f \( -name "*.pcap" -o -name "*.pcapng" -o -name "SAT-*" \) | while read -r pcap; do
    rel_path="${pcap#$PCAP_ROOT/}"
    pcap_dirname="$(dirname "$rel_path")"
    pcap_basename="$(basename "$pcap")"

    out_dir="$OUT_ROOT/$pcap_dirname/$pcap_basename"
    mkdir -p "$out_dir"

    echo "Processing: $pcap"
    zeek -C -r "$pcap" Log::default_logdir="$out_dir"

    if [[ -f "$out_dir/conn.log" ]]; then
        zeek-cut -m < "$out_dir/conn.log" > "$out_dir/conn.tsv"
    else
        echo "Warning: no conn.log for $pcap"
    fi
done