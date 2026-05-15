#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="zeek_logs"

mkdir -p "$OUT_DIR"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <pcap1> [pcap2 ...]"
    exit 1
fi

create_mac_ip_mapping() {
    local dhcp_log="$1"
    local output="$2"

    awk -F'\t' '
      /^#fields/ {
        for (i = 2; i <= NF; i++) {
          col = i - 1

          if ($i == "mac") mac_col = col
          if ($i == "assigned_addr") assigned_col = col
          if ($i == "client_addr") client_col = col
          if ($i == "requested_addr") requested_col = col
          if ($i == "host_name") host_col = col
        }
        next
      }

      /^#/ { next }

      {
        mac = tolower($mac_col)

        ip = "-"
        if (assigned_col && $assigned_col != "-") {
          ip = $assigned_col
        } else if (client_col && $client_col != "-") {
          ip = $client_col
        } else if (requested_col && $requested_col != "-") {
          ip = $requested_col
        }

        host = "-"
        if (host_col && $host_col != "-") {
          host = $host_col
        }

        if (mac != "-" && mac != "" && ip != "-" && ip != "0.0.0.0") {
          print mac "\t" ip "\t" host
        }
      }
    ' "$dhcp_log" \
    | sort -u \
    | awk -F'\t' '
      BEGIN {
        OFS = "\t"
        print "mac", "ip_addresses", "host_names"
      }

      {
        mac = $1
        ip = $2
        host = $3

        if (ips[mac] == "") {
          ips[mac] = ip
        } else if (ips[mac] !~ "(^|;)" ip "(;|$)") {
          ips[mac] = ips[mac] ";" ip
        }

        if (host != "-" && host != "") {
          if (hosts[mac] == "") {
            hosts[mac] = host
          } else if (hosts[mac] !~ "(^|;)" host "(;|$)") {
            hosts[mac] = hosts[mac] ";" host
          }
        }
      }

      END {
        for (mac in ips) {
          if (hosts[mac] == "") hosts[mac] = "-"
          print mac, ips[mac], hosts[mac]
        }
      }
    ' > "$output"
}

for pcap in "$@"; do
    pcap=$(realpath "$pcap")
    fname=$(basename "$pcap")
    stem="${fname%.pcap}"

    workdir="$OUT_DIR/$stem"

    if [ -d "$workdir" ]; then
        echo "Skipping Zeek for $fname because $workdir already exists"
    else
        mkdir -p "$workdir"

        echo "Processing $fname"

        (
            cd "$workdir"

            echo "  -> Running Zeek"

            setsid zeek -C -r "$pcap" &
            zeek_pid=$!

            LIMIT=$((200 * 1024 * 1024))

            while kill -0 "$zeek_pid" 2>/dev/null; do
                if [ -f conn.log ]; then
                    size=$(stat -c%s conn.log 2>/dev/null || echo 0)

                    if [ "$size" -ge "$LIMIT" ]; then
                        echo "  -> conn.log reached 200 MiB, stopping Zeek"

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

    conn_log="$workdir/conn.log"
    conn_tsv="$workdir/conn.tsv"
    dhcp_log="$workdir/dhcp.log"
    mapping_tsv="$workdir/mac_ip_mapping.tsv"

    if [ -s "$conn_log" ]; then
        echo "  -> Creating $conn_tsv"
        zeek-cut -m < "$conn_log" > "$conn_tsv"
    else
        echo "Warning: missing conn.log for $fname"
    fi

    if [ -s "$dhcp_log" ]; then
        echo "  -> Creating $mapping_tsv"
        create_mac_ip_mapping "$dhcp_log" "$mapping_tsv"
    else
        echo "Warning: missing dhcp.log for $fname"
        echo -e "mac\tip_addresses\thost_names" > "$mapping_tsv"
    fi
done

echo "Done. Outputs saved under $OUT_DIR/"