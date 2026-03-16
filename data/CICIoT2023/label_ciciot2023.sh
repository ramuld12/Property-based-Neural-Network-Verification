#!/usr/bin/env bash
set -euo pipefail

PCAP_DIR="pcaps"
OUT_DIR="zeek_logs"
FINAL_CSV="ciciot2023_labeled_conn.csv"

mkdir -p "$OUT_DIR"

# Final merged CSV header
echo "ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,service,conn_state,duration,orig_bytes,resp_bytes,orig_pkts,resp_pkts,missed_bytes,label" > "$FINAL_CSV"

for pcap in "$PCAP_DIR"/*.pcap; do
    fname=$(basename "$pcap")
    stem="${fname%.pcap}"

    # Decide label from filename
    if [[ "$stem" == BenignTraffic* ]]; then
        label="BENIGN"
    elif [[ "$stem" == DoS-HTTP_Flood* ]]; then
        label="DoS-HTTP_Flood"
    else
        label="$stem"
    fi

    workdir="$OUT_DIR/$stem"
    mkdir -p "$workdir"

    echo "Processing $fname -> label=$label"

    # Run Zeek inside a separate folder so logs do not overwrite each other
    (
        cd "$workdir"
        zeek -C -r "../../$pcap"
    )

    # Skip if conn.log was not produced
    if [[ ! -f "$workdir/conn.log" ]]; then
        echo "Warning: no conn.log for $fname"
        continue
    fi

    # Extract selected fields and append label + source file
    zeek-cut ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service conn_state duration orig_bytes resp_bytes orig_pkts resp_pkts missed_bytes < "$workdir/conn.log" \
    | while IFS=$'\t' read -r ts uid orig_h orig_p resp_h resp_p proto service conn_state duration orig_bytes resp_bytes orig_pkts resp_pkts missed_bytes; do
        echo "$ts,$uid,$orig_h,$orig_p,$resp_h,$resp_p,$proto,$service,$conn_state,$duration,$orig_bytes,$resp_bytes,$orig_pkts,$resp_pkts,$missed_bytes,$label"
    done >> "$FINAL_CSV"
done

echo "Done. Output saved to $FINAL_CSV"