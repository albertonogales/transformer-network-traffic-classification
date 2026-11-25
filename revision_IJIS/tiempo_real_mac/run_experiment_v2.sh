#!/bin/bash
# Campaña de captura en vivo para la revisión IJIS (macOS). Misma estructura que
# 'Real-Time Experiments/scripts/run_experiment.sh' (10 ventanas de 8 s, mismos campos de TShark),
# pero clasificando con rt_experiment_v2.py y guardando en una carpeta nueva.
#
# Uso:  bash run_experiment_v2.sh <Clase> <Run>
# Ej.:  bash run_experiment_v2.sh Email 1
#
# Requiere: tshark, python3 con numpy y pandas, y en esta misma carpeta
#           captura_comun.py, rt_experiment_v2.py y pesos.pkl

set -u
CLASS=$1
RUN=$2
DIR="${RT_DIR:-/tmp/live_experiments_v2}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
export RT_DIR="$DIR"
export RT_PESOS="${RT_PESOS:-$AQUI/pesos.pkl}"

IFACE=$(networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}')
PCAP_DIR="${DIR}/${CLASS}/run${RUN}/pcaps"
CSV_DIR="${DIR}/${CLASS}/run${RUN}/csvs"

# En la campaña de 2026 varios runs se repitieron escribiendo sobre las mismas rutas
# 'run<N>/csvs/pcap_<i>.csv', de modo que en disco solo quedó el último intento mientras el log
# conservaba todos. Aquí se rechaza sobrescribir: cada intento tiene que ser un run nuevo, para que
# el archivo y el log coincidan y no haya que elegir intentos después.
if [ -d "$CSV_DIR" ] && [ -n "$(ls -A "$CSV_DIR" 2>/dev/null)" ]; then
    echo "ERROR: ${CSV_DIR} ya tiene datos."
    echo "       Usa un número de run libre; no repitas un run sobre el anterior."
    exit 1
fi
mkdir -p "$PCAP_DIR" "$CSV_DIR"

echo "========================================================"
echo "  CLASE: $CLASS | RUN: $RUN | Interfaz: $IFACE"
echo "  Modelo: $RT_PESOS"
echo "  Salida: $DIR"
echo "========================================================"

for i in $(seq 1 10); do
    echo -n "  [Capturando ventana $i/10 - 8s]..."
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
    echo ""
    python3 "$AQUI/rt_experiment_v2.py" "$CLASS" "$RUN" "$i"
done

echo ""
echo "  Run $RUN completo. Resultados en ${DIR}/results_log.csv"
echo "========================================================"
