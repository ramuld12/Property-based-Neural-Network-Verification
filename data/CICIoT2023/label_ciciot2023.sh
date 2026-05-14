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
        label="PORTSCAN"
    elif [[ "$stem" == BenignTraffic* ]]; then
        label="BENIGN"
    elif [[ "$stem" == DoS-HTTP_Flood* ]]; then
        label="DOS_HTTP_FLOOD"
    elif [[ "$stem" == DoS-UDP_Flood* ]]; then
        label="DOS_UDP_FLOOD"
    elif [[ "$stem" == DoS-SYN_Flood* ]]; then
        label="DOS_SYN_FLOOD"
    elif [[ "$stem" == XSS* ]]; then
        label="XSS"
    elif [[ "$stem" == SqlInjection* ]]; then
        label="SQL_INJECTION"
    elif [[ "$stem" == DictionaryBruteForce* ]]; then
        label="BRUTE_FORCE"
    else
        label="ATTACK"
    fi

    workdir="$OUT_DIR/$stem"

    # Skip if already processed

    if [ -d "$workdir" ]; then
        echo "Skipping zeek-processing for $fname (directory already exists)"
    else
        mkdir -p "$workdir"

        echo "Processing $fname -> label=$label"

        (
            cd "$workdir"

            echo "  -> Running Zeek (limit: 0.2 GB conn.log)..."

            setsid zeek -C -r "$pcap" &
            zeek_pid=$!

            LIMIT=$((200 * 1024 * 1024))  # 200 MiB

            while kill -0 "$zeek_pid" 2>/dev/null; do
                if [ -f conn.log ]; then
                    size=$(stat -c%s conn.log 2>/dev/null || echo 0)

                    if [ "$size" -ge "$LIMIT" ]; then
                        echo "  -> conn.log reached 200 MiB, stopping Zeek for $fname"

                        # Kill the whole process group
                        kill -TERM -- "-$zeek_pid" 2>/dev/null || true
                        sleep 3
                        kill -KILL -- "-$zeek_pid" 2>/dev/null || true

                        break
                    fi
                fi
                sleep 1
            done

            wait "$zeek_pid" 2>/dev/null || true
        )
    fi


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