"""
Preprocesamiento de capturas y inferencia en NumPy para la revisión IJIS.

Este módulo reproduce EXACTAMENTE el preprocesamiento del entrenamiento
(src/procesamiento_pcap.procesarDF + src/procesamiento_flujos.obtenerFlujos/obtenerSemiflujos)
sobre los CSV de TShark que generan las capturas en vivo, y evalúa el modelo del artículo
(con el número de capas y la codificación posicional que se le indiquen) usando solo NumPy,
para poder ejecutarlo en una máquina sin TensorFlow.

Diferencias respecto a 'Real-Time Experiments/scripts/rt_experiment.py' (verificadas en la
revisión, 14-11-2026); esa versión no se modifica:
  1. Clave de flujo. Entrenamiento: IPsrc-IPdst-Protocol-TCPsrcport-TCPdstport-UDPsrcport-UDPdstport
     (UNIDIRECCIONAL, con protocolo). rt_experiment.py: tuple(sorted([(src,sp),(dst,dp)])), es decir
     BIDIRECCIONAL y sin protocolo, de modo que mezcla los dos sentidos de la conversación.
  2. Puertos UDP. rt_experiment.py usa row.get("tcp.srcport", row.get("udp.srcport","")): como la
     columna 'tcp.srcport' existe y vale NaN en los paquetes UDP, pandas devuelve NaN en vez del
     valor por defecto, así que todas las conversaciones UDP entre dos IP caen en la misma clave.
  3. Segmentación en semi-flujos. Entrenamiento: al superarse el tiempo máximo se DESECHA el
     semi-flujo parcial y se reinicia en ese paquete. rt_experiment.py desliza la ventana un paquete.
  4. Lista de protocolos: la del entrenamiento (incluye SSLv2, TLSv1.1 y las variantes GTP, y no
     incluye el nombre 'QUIC' a secas).
La normalización NO es una diferencia: dividir cada paquete por su máximo equivale a dividir entre
65535, porque todo paquete lleva el token separador FFFF.
"""

import os
import time

import numpy as np
import pandas as pd

# --- Constantes del artículo -----------------------------------------------------------------
CLASES = ["Chat", "Email", "FileTransfer", "Streaming", "VoIP"]
LONGITUD_RAW = 800      # caracteres hexadecimales de payload (400 bytes)
NUM_HEX_TOKEN = 4       # 4 hexadecimales por token -> uint16
N_TOKENS = LONGITUD_RAW // NUM_HEX_TOKEN            # 200
TOKEN_SEPARADOR = 65535                             # np.iinfo(np.uint16).max
NUM_PAQUETES_SEMIFLUJO = 20
MIN_PAQUETES_FLUJO = 20
TIEMPO_MAXIMO_SEMIFLUJO = 30.0

# src/procesamiento_pcap.py:50 (procesarDF)
PROTOCOLOS = ["TCP", "UDP", "SSL", "SSLv2", "TLSv1", "TLSv1.1", "TLSv1.2", "TLSv1.3", "SSHv2",
              "GTP <TCP>", "GTP <UDP>", "GTP <QUIC>", "GTP <SSL>", "GTP <SSLv2>", "GTP <TLSv1>",
              "GTP <TLSv1.1>", "GTP <TLSv1.2>", "GTP <TLSv1.3>"]
# Protocolos cuyo payload está en la columna UDPpayload (src/procesamiento_pcap.py:85-89)
PROTOCOLOS_UDP = {"UDP", "GTP <QUIC>"}

COLUMNAS = {"_ws.col.number": "No.", "_ws.col.cls_time": "Time", "_ws.col.def_src": "IPsrc",
            "_ws.col.def_dst": "IPdst", "_ws.col.protocol": "Protocol",
            "_ws.col.packet_length": "Length", "tcp.srcport": "TCPsrcport",
            "tcp.dstport": "TCPdstport", "tcp.len": "TCPlen", "udp.srcport": "UDPsrcport",
            "udp.dstport": "UDPdstport", "tcp.payload": "TCPpayload", "udp.payload": "UDPpayload"}


# --- Preprocesamiento ------------------------------------------------------------------------

def _texto(v):
    """str(x) con el mismo resultado que df[col].astype(str) de pandas (NaN -> 'nan')."""
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def paquetes_de_csv(ruta):
    """Devuelve [(tiempo, payload_hex, clave_flujo)] aplicando los filtros de procesarDF."""
    df = pd.read_csv(ruta, low_memory=False)
    df = df.rename(columns={k: v for k, v in COLUMNAS.items() if k in df.columns})
    n_paquetes = len(df)
    if n_paquetes == 0:
        return [], 0
    # procesarDF: solo los protocolos admitidos
    df = df[df["Protocol"].astype(str).str.strip().isin(PROTOCOLOS)]
    # procesarDF: fuera los paquetes con TCPlen <= 2 (en UDP, TCPlen es NaN y NaN <= 2 es False)
    if "TCPlen" in df.columns:
        df = df[~(pd.to_numeric(df["TCPlen"], errors="coerce") <= 2)]
    paquetes = []
    for fila in df.itertuples(index=False):
        d = fila._asdict()
        proto = str(d.get("Protocol", "")).strip()
        # procesarDF vacía UDPpayload si no es UDP/GTP <QUIC>, y TCPpayload si lo es
        payload = d.get("UDPpayload") if proto in PROTOCOLOS_UDP else d.get("TCPpayload")
        payload = "" if payload is None or (isinstance(payload, float) and np.isnan(payload)) else str(payload)
        if not payload or payload == "nan":
            continue        # procesarDF elimina los paquetes sin payload
        clave = "-".join((_texto(d.get("IPsrc")), _texto(d.get("IPdst")), proto,
                          _texto(d.get("TCPsrcport")), _texto(d.get("TCPdstport")),
                          _texto(d.get("UDPsrcport")), _texto(d.get("UDPdstport"))))
        paquetes.append((float(d.get("Time", 0) or 0), payload, clave))
    return paquetes, n_paquetes


def tokenizar(payload):
    """procesar_RAW: trunca/rellena a 800 hex, 200 tokens de 4 hex y token separador FFFF."""
    p = payload.replace(":", "").replace(" ", "")
    p = p.ljust(LONGITUD_RAW, "0") if len(p) < LONGITUD_RAW else p[:LONGITUD_RAW]
    v = np.empty(N_TOKENS + 1, dtype=np.uint16)
    for i in range(N_TOKENS):
        try:
            v[i] = int(p[i * NUM_HEX_TOKEN:(i + 1) * NUM_HEX_TOKEN], 16)
        except ValueError:
            v[i] = 0
    v[N_TOKENS] = TOKEN_SEPARADOR
    return v


def semiflujos_de_paquetes(paquetes):
    """obtenerFlujos + obtenerSemiflujos: (n, 20, 201) uint16. Devuelve también el nº de flujos."""
    flujos = {}
    for t, payload, clave in paquetes:
        flujos.setdefault(clave, []).append((t, payload))
    salida = []
    for pkts in flujos.values():
        if len(pkts) < MIN_PAQUETES_FLUJO:      # obtenerFlujos descarta los flujos cortos
            continue
        contador, t_inicio, actual = 0, None, []
        for t, payload in pkts:                 # orden de captura, como en el dataframe original
            if contador == 0 or (t - t_inicio) > TIEMPO_MAXIMO_SEMIFLUJO:
                # semi-flujo nuevo (el parcial, si lo había, se desecha)
                t_inicio, actual, contador = t, [tokenizar(payload)], 1
                continue
            actual.append(tokenizar(payload))
            contador += 1
            if contador == NUM_PAQUETES_SEMIFLUJO:
                salida.append(np.stack(actual))
                contador = 0
    X = np.stack(salida) if salida else np.zeros((0, NUM_PAQUETES_SEMIFLUJO, N_TOKENS + 1), np.uint16)
    return X, len(flujos)


def preprocesar_csv(ruta):
    """CSV de TShark -> (X uint16, n_paquetes, n_flujos, segundos)."""
    t0 = time.perf_counter()
    paquetes, n_paquetes = paquetes_de_csv(ruta)
    X, n_flujos = semiflujos_de_paquetes(paquetes)
    return X, n_paquetes, n_flujos, time.perf_counter() - t0


PROTOS_RT = {"TCP", "UDP", "SSL", "TLSv1", "TLSv1.2", "TLSv1.3", "SSHv2", "QUIC"}


def preprocesar_csv_rt(ruta):
    """Preprocesamiento de 'Real-Time Experiments/scripts/rt_experiment.py', reproducido literalmente
    (clave bidireccional sin protocolo, puertos UDP perdidos en NaN y ventana deslizante), para poder
    comparar las dos variantes sobre la misma ventana. Devuelve (X, n_paquetes, n_flujos, segundos)."""
    t0 = time.perf_counter()
    flujos = {}
    df = pd.read_csv(ruta, low_memory=False).rename(columns={
        "_ws.col.cls_time": "Time", "_ws.col.def_src": "Source",
        "_ws.col.def_dst": "Destination", "_ws.col.protocol": "Protocol"})
    n_paquetes = len(df)
    for _, row in df.iterrows():
        proto = str(row.get("Protocol", "")).strip()
        if proto not in PROTOS_RT:
            continue
        if proto == "TCP":
            try:
                if float(row.get("tcp.len", 0) or 0) <= 2:
                    continue
                pay = str(row.get("tcp.payload", "") or "")
            except Exception:
                continue
        else:
            pay = str(row.get("udp.payload", "") or "")
        if not pay or pay == "nan":
            continue
        sp = str(row.get("tcp.srcport", row.get("udp.srcport", "")) or "")
        dp = str(row.get("tcp.dstport", row.get("udp.dstport", "")) or "")
        src = str(row.get("Source", "") or "")
        dst = str(row.get("Destination", "") or "")
        flujos.setdefault(tuple(sorted([(src, sp), (dst, dp)])), []).append(
            (float(row.get("Time", 0) or 0), pay))
    semi = []
    for pkts in flujos.values():
        pkts.sort(key=lambda x: x[0])
        if len(pkts) < MIN_PAQUETES_FLUJO:
            continue
        i = 0
        while i + NUM_PAQUETES_SEMIFLUJO <= len(pkts):
            trozo = pkts[i:i + NUM_PAQUETES_SEMIFLUJO]
            if trozo[-1][0] - trozo[0][0] > TIEMPO_MAXIMO_SEMIFLUJO:
                i += 1
                continue
            semi.append(np.stack([tokenizar(p) for _, p in trozo]))
            i += NUM_PAQUETES_SEMIFLUJO
    X = np.stack(semi) if semi else np.zeros((0, NUM_PAQUETES_SEMIFLUJO, N_TOKENS + 1), np.uint16)
    return X, n_paquetes, len(flujos), time.perf_counter() - t0


PREPROCESOS = {"articulo": preprocesar_csv, "rt": preprocesar_csv_rt}


def normalizar(X):
    """Normalización del artículo: cada paquete dividido por su máximo (comun.normalizar)."""
    Xf = np.asarray(X, dtype=np.float32)
    m = np.abs(Xf).max(axis=2, keepdims=True)
    m[m == 0] = 1.0
    return Xf / m


# --- Inferencia en NumPy ---------------------------------------------------------------------

def codificacion_posicional(n_pos, d_model):
    """src/modelos.py:345 (PositionalEncoding), escala 1.0."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d_model)[None, :]
    ang = pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
    ang[:, 0::2] = np.sin(ang[:, 0::2])
    ang[:, 1::2] = np.cos(ang[:, 1::2])
    return ang.astype(np.float64)


def _layer_norm(x, g, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return g * (x - m) / np.sqrt(v + eps) + b


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class ModeloNumPy:
    """Modelo del artículo (build_keras_transformer_model) evaluado con NumPy.

    Orden de pesos de Keras, por bloque encoder: Wq,bq,Wk,bk,Wv,bv,Wo,bo (MultiHeadAttention),
    gamma,beta (LayerNormalization), k1,b1 (Conv1D ff_dim), k2,b2 (Conv1D d_model); al final
    d1w,d1b (Dense mlp_units) y d2w,d2b (Dense n_clases). En inferencia no hay dropout.
    """

    def __init__(self, pesos, capas, pe, input_shape):
        esperados = 14 * capas + 4
        if len(pesos) != esperados:
            raise ValueError(f"{len(pesos)} pesos para {capas} capas; se esperaban {esperados}")
        self.capas = capas
        self.pe = codificacion_posicional(*input_shape) if pe else None
        self.bloques = [[np.asarray(w, dtype=np.float64) for w in pesos[14 * c:14 * (c + 1)]]
                        for c in range(capas)]
        self.d1w, self.d1b, self.d2w, self.d2b = [np.asarray(w, dtype=np.float64) for w in pesos[-4:]]
        self.escala_atencion = 1.0 / np.sqrt(self.bloques[0][0].shape[-1])   # 1/sqrt(head_size)

    def _encoder(self, x, w):
        Wq, bq, Wk, bk, Wv, bv, Wo, bo, g, b, k1, b1, k2, b2 = w
        q = np.einsum("bti,ihd->bhtd", x, Wq) + bq[None, :, None, :]
        k = np.einsum("bti,ihd->bhtd", x, Wk) + bk[None, :, None, :]
        v = np.einsum("bti,ihd->bhtd", x, Wv) + bv[None, :, None, :]
        a = _softmax(np.einsum("bhtd,bhsd->bhts", q, k) * self.escala_atencion)
        o = np.einsum("bhts,bhsd->bthd", a, v)
        res = np.einsum("bthd,hdo->bto", o, Wo) + bo + x
        y = _layer_norm(res, g, b)
        y = np.maximum(y @ k1[0] + b1, 0.0)        # Conv1D kernel_size=1 + relu
        ffn = y @ k2[0] + b2
        return np.clip(ffn + res, -1e4, 1e4)

    def predict(self, X, batch=512):
        """X: (n, 20, 201) ya normalizado. Devuelve probabilidades (n, 5)."""
        salidas = []
        for i in range(0, len(X), batch):
            x = np.asarray(X[i:i + batch], dtype=np.float64)
            if self.pe is not None:
                x = x + self.pe
            for w in self.bloques:
                x = self._encoder(x, w)
            pool = x.mean(axis=2)                  # GlobalAveragePooling1D(channels_first)
            h = np.maximum(pool @ self.d1w + self.d1b, 0.0)
            salidas.append(_softmax(h @ self.d2w + self.d2b))
        return np.concatenate(salidas) if salidas else np.zeros((0, len(CLASES)), np.float32)


def cargar_modelo(ruta_pkl):
    """Carga un '<metodo>_capas<N>_PE-<si|no>_pesos.pkl' guardado por 02_exp_desbalance.py."""
    import pickle
    with open(ruta_pkl, "rb") as fp:
        d = pickle.load(fp)
    modelo = ModeloNumPy(d["pesos"], int(d["capas"]), str(d["pe"]) == "si", tuple(d["input_shape"]))
    return modelo, d
