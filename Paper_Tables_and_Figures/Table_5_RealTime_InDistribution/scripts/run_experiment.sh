#!/bin/bash
# Usage: bash /tmp/run_experiment.sh <Class> <RunNumber>
# Example: bash /tmp/run_experiment.sh Chat 1

CLASS=$1
RUN=$2
IFACE=$(networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}')
PCAP_DIR="/tmp/live_experiments/${CLASS}/run${RUN}/pcaps"
CSV_DIR="/tmp/live_experiments/${CLASS}/run${RUN}/csvs"

mkdir -p "$PCAP_DIR" "$CSV_DIR"

echo "========================================================"
echo "  CLASS: $CLASS | RUN: $RUN | Interface: $IFACE"
echo "========================================================"

for i in $(seq 1 10); do
    echo -n "  [Capturing window $i/10 - 8s]..."
    tshark -i "$IFACE" -a duration:8 -w "${PCAP_DIR}/pcap_${i}.pcap" 2>/dev/null
    tshark -r "${PCAP_DIR}/pcap_${i}.pcap" \
        -E separator=, -E occurrence=l -E header=y -Tfields \
        -e _ws.col.number -e _ws.col.cls_time \
        -e _ws.col.def_src -e _ws.col.def_dst \
        -e _ws.col.protocol -e _ws.col.packet_length \
        -e tcp.srcport -e tcp.dstport -e tcp.len \
        -e udp.srcport -e udp.dstport \
        -e tcp.payload -e udp.payload \
        > "${CSV_DIR}/pcap_${i}.csv" 2>/dev/null
    python3 /tmp/rt_experiment.py "$CLASS" "$RUN" "$i" 2>/dev/null
done

echo ""
echo "  Run $RUN complete. Results logged to /tmp/live_experiments/results_log.csv"
echo "========================================================"
