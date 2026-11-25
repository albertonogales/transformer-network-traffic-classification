"""
Reclasifica las capturas en vivo ya guardadas con un modelo de la revisión (R2.2/R2.3).

Lee los CSV de TShark de 'Real-Time Experiments/results' SIN modificar nada de esa carpeta y
escribe los resultados en 'revision_IJIS/resultados/exp_R2.5_tiempo_real'.

Permite separar el efecto del modelo del efecto del preprocesamiento:
  --preproc articulo  preprocesamiento idéntico al del entrenamiento (por defecto)
  --preproc rt        preprocesamiento de 'Real-Time Experiments/scripts/rt_experiment.py'
                      (flujo bidireccional sin protocolo, puertos UDP perdidos, ventana deslizante)

Ejemplo:
  python revision_IJIS/03_clasificar_capturas.py ^
      --pesos revision_IJIS/resultados/exp_R2.3_desbalance/focal_g2_sin_alfa_capas2_PE-si_pesos.pkl ^
      --etiqueta focal_capas2_PE --preproc articulo
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import captura_comun as cc  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURAS = os.path.abspath(os.path.join(RAIZ, "..", "Real-Time Experiments", "results"))

# --- Preprocesamiento de rt_experiment.py, reproducido aquí para poder compararlos -----------
PROTOS_RT = {"TCP", "UDP", "SSL", "TLSv1", "TLSv1.2", "TLSv1.3", "SSHv2", "QUIC"}


def preprocesar_rt(ruta):
    """Réplica literal de rt_experiment.py:preprocess (incluido el fallback de puertos UDP)."""
    t0 = time.perf_counter()
    flujos = defaultdict(list)
    df = pd.read_csv(ruta, low_memory=False).rename(columns={
        "_ws.col.cls_time": "_ws.col.Time", "_ws.col.def_src": "_ws.col.Source",
        "_ws.col.def_dst": "_ws.col.Destination", "_ws.col.protocol": "_ws.col.Protocol"})
    n_paquetes = len(df)
    for _, row in df.iterrows():
        proto = str(row.get("_ws.col.Protocol", "")).strip()
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
        src = str(row.get("_ws.col.Source", "") or "")
        dst = str(row.get("_ws.col.Destination", "") or "")
        flujos[tuple(sorted([(src, sp), (dst, dp)]))].append((float(row.get("_ws.col.Time", 0) or 0), pay))
    semi = []
    for pkts in flujos.values():
        pkts.sort(key=lambda x: x[0])
        if len(pkts) < cc.MIN_PAQUETES_FLUJO:
            continue
        i = 0
        while i + cc.NUM_PAQUETES_SEMIFLUJO <= len(pkts):
            trozo = pkts[i:i + cc.NUM_PAQUETES_SEMIFLUJO]
            if trozo[-1][0] - trozo[0][0] > cc.TIEMPO_MAXIMO_SEMIFLUJO:
                i += 1
                continue
            semi.append(np.stack([cc.tokenizar(p) for _, p in trozo]))
            i += cc.NUM_PAQUETES_SEMIFLUJO
    X = np.stack(semi) if semi else np.zeros((0, cc.NUM_PAQUETES_SEMIFLUJO, cc.N_TOKENS + 1), np.uint16)
    return X, n_paquetes, len(flujos), time.perf_counter() - t0


# --- Recorrido de las capturas ---------------------------------------------------------------

def ventanas(carpeta):
    """[(clase, run, ventana, ruta_csv)] de todas las capturas guardadas."""
    salida = []
    for clase in sorted(os.listdir(carpeta)):
        d = os.path.join(carpeta, clase)
        if not os.path.isdir(d) or clase not in cc.CLASES:
            continue
        for sub in sorted(os.listdir(d)):
            dir_csv = os.path.join(d, sub, "csvs") if sub != "csvs" else os.path.join(d, "csvs")
            if not os.path.isdir(dir_csv):
                continue
            run = sub if sub != "csvs" else "run0"
            for f in sorted(os.listdir(dir_csv), key=lambda s: (len(s), s)):
                if f.lower().endswith(".csv"):
                    salida.append((clase, run, os.path.splitext(f)[0], os.path.join(dir_csv, f)))
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True, help="*_pesos.pkl guardado por 02_exp_desbalance.py")
    ap.add_argument("--etiqueta", required=True, help="nombre corto para los archivos de salida")
    ap.add_argument("--preproc", default="articulo", choices=["articulo", "rt"])
    ap.add_argument("--capturas", default=CAPTURAS)
    ap.add_argument("--resultados", default=os.path.join(RAIZ, "revision_IJIS", "resultados",
                                                         "exp_R2.5_tiempo_real"))
    ap.add_argument("--clases", nargs="*", default=cc.CLASES)
    args = ap.parse_args()

    modelo, meta = cc.cargar_modelo(args.pesos)
    os.makedirs(args.resultados, exist_ok=True)
    base = f"{args.etiqueta}_preproc-{args.preproc}"
    print(f"Modelo: {meta['metodo']} capas={meta['capas']} PE={meta['pe']} "
          f"(mejor fold {meta['mejor_fold']}) | preprocesamiento: {args.preproc}")

    lista = [v for v in ventanas(args.capturas) if v[0] in args.clases]
    print(f"{len(lista)} ventanas de captura en {args.capturas}")

    filas = []
    conf = np.zeros((len(cc.CLASES), len(cc.CLASES)), dtype=np.int64)   # verdad x predicción
    acierto_dominante = defaultdict(lambda: [0, 0])
    t_inicio = time.time()
    for n, (clase, run, ventana, ruta) in enumerate(lista, 1):
        try:
            X, n_paq, n_flu, t_pre = (cc.preprocesar_csv(ruta) if args.preproc == "articulo"
                                      else preprocesar_rt(ruta))
        except Exception as e:                       # CSV vacío o corrupto
            print(f"  [{n}/{len(lista)}] {clase}/{run}/{ventana}: {type(e).__name__}: {e}")
            continue
        t0 = time.perf_counter()
        p = modelo.predict(cc.normalizar(X)) if len(X) else np.zeros((0, len(cc.CLASES)), np.float32)
        t_inf = time.perf_counter() - t0
        pred = p.argmax(1) if len(p) else np.zeros(0, dtype=int)
        iv = cc.CLASES.index(clase)
        for c in pred:
            conf[iv, c] += 1
        dom = cc.CLASES[int(np.bincount(pred, minlength=len(cc.CLASES)).argmax())] if len(pred) else ""
        acierto_dominante[clase][1] += 1
        if dom == clase:
            acierto_dominante[clase][0] += 1
        filas.append(dict(clase=clase, run=run, ventana=ventana, n_paquetes=n_paq, n_flujos=n_flu,
                          n_semiflujos=len(X), t_preproceso=round(t_pre, 4),
                          t_inferencia=round(t_inf, 4),
                          ms_por_semiflujo=round(1000 * t_inf / len(X), 4) if len(X) else "",
                          dominante=dom, confianza_media=round(float(p.max(1).mean()), 4) if len(p) else "",
                          aciertos=int((pred == iv).sum()),
                          **{f"n_{c}": int((pred == i).sum()) for i, c in enumerate(cc.CLASES)}))
        if n % 25 == 0 or n == len(lista):
            print(f"  [{n}/{len(lista)}] {clase}/{run}/{ventana}: {len(X)} semi-flujos, "
                  f"dominante {dom} ({time.time() - t_inicio:.0f} s)")

    ruta_csv = os.path.join(args.resultados, f"{base}_ventanas.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    # --- Resumen ---
    tot = conf.sum()
    resumen = dict(modelo=dict(metodo=meta["metodo"], capas=meta["capas"], pe=meta["pe"],
                               mejor_fold=meta["mejor_fold"]),
                   preprocesamiento=args.preproc, ventanas=len(filas), semiflujos=int(tot),
                   accuracy=float(np.trace(conf) / tot) if tot else 0.0,
                   matriz_confusion=conf.tolist(), clases=cc.CLASES, por_clase={})
    print("\n| Clase | Semi-flujos | Accuracy (semi-flujo) | Ventanas con clase dominante correcta |")
    print("|---|---|---|---|")
    for i, c in enumerate(cc.CLASES):
        n_i = int(conf[i].sum())
        acc = float(conf[i, i] / n_i) if n_i else float("nan")
        ok, nv = acierto_dominante[c]
        resumen["por_clase"][c] = dict(semiflujos=n_i, accuracy=acc, ventanas=nv, ventanas_ok=ok)
        print(f"| {c} | {n_i} | {acc*100:.1f} % | {ok}/{nv} |")
    print(f"| **Total** | {int(tot)} | **{resumen['accuracy']*100:.1f} %** | "
          f"{sum(v[0] for v in acierto_dominante.values())}/{sum(v[1] for v in acierto_dominante.values())} |")
    ms = [f["ms_por_semiflujo"] for f in filas if f["ms_por_semiflujo"] != ""]
    if ms:
        resumen["ms_por_semiflujo_mediana"] = float(np.median(ms))
        print(f"\nLatencia de inferencia (NumPy, esta máquina): mediana "
              f"{np.median(ms):.3f} ms por semi-flujo")
    with open(os.path.join(args.resultados, f"{base}_resumen.json"), "w", encoding="utf-8") as fp:
        json.dump(resumen, fp, indent=1, ensure_ascii=False)
    print(f"\nEscrito: {ruta_csv}")


if __name__ == "__main__":
    main()
