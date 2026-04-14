#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="zeek_logs"
FINAL_CSV="ciciot2023_labeled_conn.tsv"

mkdir -p "$OUT_DIR"

# Ensure at least one PCAP is provided

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <pcap1> [pcap2 ...]"
    exit 1
fi

header_written=false

for pcap in "$@"; do
    pcap=$(realpath "$pcap")
    fname=$(basename "$pcap")
    stem="${fname%.pcap}"


    # Decide label from filename
    if [[ "$stem" == Recon-PortScan ]]; then
        label="PortScan"
    elif [[ "$stem" == BenignTraffic* ]]; then
        label="BENIGN"
    elif [[ "$stem" == DoS-HTTP_Flood* ]]; then
        label="DoS-HTTP_Flood"
    elif [[ "$stem" == DoS-UDP_Flood* ]]; then
        label="DoS-UDP_Flood"
    elif [[ "$stem" == DDoS-UDP_Flood* ]]; then
        label="DDoS-UDP_Flood"
    else
        label="$stem"
    fi

    workdir="$OUT_DIR/$stem"
    mkdir -p "$workdir"

    echo "Processing $fname -> label=$label"

    (
        cd "$workdir"
        zeek -C -r "$pcap"
    )

    conn="$workdir/conn.log"

    # Write header only if file doesn't exist or is empty
    if [ ! -s "$FINAL_CSV" ]; then
        header=$(grep '^#fields' "$conn" | cut -f2-)
        echo -e "${header}\tlabel" > "$FINAL_CSV"
    fi

    # Extract data rows and append label
    grep -v '^#' "$conn" | awk -v lbl="$label" '{print $0"\t"lbl}' >> "$FINAL_CSV"


done

echo "Done. Output saved to $FINAL_CSV"
