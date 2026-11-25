#!/usr/bin/env python3
"""
Clasificador en vivo (revisión IJIS), versión 2. Sustituto de
'Real-Time Experiments/scripts/rt_experiment.py', que NO se modifica.

Diferencias respecto a la versión original:
  - carga cualquier modelo guardado por 02_exp_desbalance.py (*_pesos.pkl): número arbitrario de
    bloques encoder y codificación posicional opcional, en vez de 1 bloque sin PE fijo;
  - preprocesamiento idéntico al del entrenamiento (captura_comun.py): clave de flujo
    unidireccional con protocolo, puertos UDP correctos y segmentación con reinicio al superar
    los 30 s. La versión original usaba flujos bidireccionales, perdía los puertos UDP y
    deslizaba la ventana un paquete;
  - solo necesita numpy y pandas (ni TensorFlow ni PyTorch).

Uso:  python3 rt_experiment_v2.py <Clase> <Run> <Ventana>
Variables de entorno:
  RT_PESOS    ruta(s) de *_pesos.pkl, separadas por ':' en macOS (por defecto: ./pesos.pkl)
  RT_DIR      carpeta de la campaña            (por defecto: /tmp/live_experiments_v2)
  RT_PREPROC  'articulo', 'rt' o 'ambos'        (por defecto: ambos)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import captura_comun as cc  # noqa: E402

CLASE, RUN, VENTANA = sys.argv[1], sys.argv[2], sys.argv[3]
DIR = os.environ.get("RT_DIR", "/tmp/live_experiments_v2")
PESOS = os.environ.get("RT_PESOS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pesos.pkl"))
CSV_FILE = os.path.join(DIR, CLASE, f"run{RUN}", "csvs", f"pcap_{VENTANA}.csv")
LOG = os.path.join(DIR, "results_log.csv")

# RT_PESOS admite varias rutas separadas por os.pathsep (':' en macOS/Linux) para clasificar la MISMA ventana con varios
# modelos (p. ej. el del artículo y el de la revisión) y poder compararlos sin recapturar.
modelos = [cc.cargar_modelo(r) for r in PESOS.split(os.pathsep) if r]

# RT_PREPROC: 'articulo', 'rt' o 'ambos' (por defecto). Con 'ambos' cada ventana se clasifica con
# los dos preprocesamientos y con todos los modelos, así que la decisión de cuál adoptar en el
# artículo se toma después, con los datos de la campaña, sin tener que volver a capturar.
QUE_PREPROC = os.environ.get("RT_PREPROC", "ambos")
nombres_preproc = ["articulo", "rt"] if QUE_PREPROC == "ambos" else [QUE_PREPROC]
preprocesados = {}
for nombre in nombres_preproc:
    X, n_paquetes, n_flujos, t_pre = cc.PREPROCESOS[nombre](CSV_FILE)
    preprocesados[nombre] = (X, n_paquetes, n_flujos, t_pre,
                             cc.normalizar(X) if len(X) else None)

cabecera = ("timestamp,class,run,window,n_packets,n_flows,n_semiflows,t_preprocess,t_inference,"
            "t_total,semiflows_per_s,packets_per_s,dominant,mean_confidence,"
            + ",".join(f"n_{c}" for c in cc.CLASES) + ",modelo,capas,pe,preproc\n")
nuevo_log = not os.path.exists(LOG)
os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(LOG, "a", encoding="utf-8") as fp:
    if nuevo_log:
        fp.write(cabecera)
    for nombre in nombres_preproc:
        X, n_paquetes, n_flujos, t_pre, Xn = preprocesados[nombre]
        n_semiflujos = len(X)
        for modelo, meta in modelos:
            if n_semiflujos:
                t0 = time.perf_counter()
                probs = modelo.predict(Xn)
                t_inf = time.perf_counter() - t0
                pred = probs.argmax(1)
                cuentas = np.bincount(pred, minlength=len(cc.CLASES))
                dominante = cc.CLASES[int(cuentas.argmax())]
                confianza = float(probs.max(1).mean())
            else:
                t_inf, cuentas, dominante, confianza = 0.0, np.zeros(len(cc.CLASES), dtype=int), "", 0.0
            t_total = t_pre + t_inf
            fp.write(",".join(str(v) for v in [
                datetime.now().isoformat(timespec="seconds"), CLASE, RUN, VENTANA, n_paquetes,
                n_flujos, n_semiflujos, f"{t_pre:.4f}", f"{t_inf:.4f}", f"{t_total:.4f}",
                f"{n_semiflujos / t_total:.2f}" if t_total else "0",
                f"{n_paquetes / t_total:.2f}" if t_total else "0",
                dominante, f"{confianza:.4f}", *cuentas.tolist(),
                meta["metodo"], meta["capas"], meta["pe"], nombre]) + "\n")
            print(f"  {CLASE}/run{RUN}/ventana {VENTANA} [preproc {nombre} | {meta['metodo']} "
                  f"capas={meta['capas']} PE={meta['pe']}]: {n_paquetes} paquetes, {n_flujos} "
                  f"flujos, {n_semiflujos} semi-flujos -> {dominante or 'sin predicción'} "
                  f"(confianza media {confianza:.3f}; preproc {t_pre:.2f} s, "
                  f"inferencia {t_inf:.3f} s)")
