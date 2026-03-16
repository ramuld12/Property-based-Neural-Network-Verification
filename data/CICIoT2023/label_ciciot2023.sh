#!/usr/bin/env bash
set -euo pipefail

PCAP_DIR="pcaps"
OUT_DIR="zeek_logs"
FINAL_CSV="ciciot2023_labeled_conn.tsv"

mkdir -p "$OUT_DIR"

header_written=false

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

    (
        cd "$workdir"
        zeek -C -r "../../$pcap"
    )

    conn="$workdir/conn.log"

    # Write header only once
    if [ "$header_written" = false ]; then
        header=$(grep '^#fields' "$conn" | cut -f2-)
        echo -e "${header}\tlabel" > "$FINAL_CSV"
        header_written=true
    fi

    # Extract data rows and append label
    grep -v '^#' "$conn" | awk -v lbl="$label" '{print $0"\t"lbl}' >> "$FINAL_CSV"

done

echo "Done. Output saved to $FINAL_CSV"