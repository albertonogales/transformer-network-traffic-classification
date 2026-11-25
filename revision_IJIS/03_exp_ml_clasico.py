"""
R2.4 — Baselines de ML clásico sobre estadísticas de flujo (Exp. 4 del plan de respuesta).

Pregunta de R2: ¿aporta la tokenización del payload algo frente a estadísticas clásicas de tamaño y tiempo?
Se entrenan XGBoost y Random Forest sobre los MISMOS semi-flujos y la MISMA partición que el Transformer.

Características por semi-flujo (de meta.npz, alineado con X.npy):
  - 20 longitudes de payload (bytes reales, sin truncar) y 19 tiempos entre llegadas (IAT)
  - agregados: media, desviación, mínimo, máximo y mediana de longitudes e IAT; duración; bytes totales
  (No hay variable de dirección: en este pipeline los flujos son unidireccionales, así que todos los paquetes
  de un semi-flujo van en el mismo sentido. Conviene decirlo en el artículo.)

Protocolo: rejilla pequeña de hiperparámetros evaluada con los mismos 5 folds (media de accuracy de validación,
como en el artículo); el modelo final se reentrena con todo el 90 % de desarrollo y se evalúa una vez en test.
Se prueban versiones sin ponderar y con pesos de clase 'balanced' (comparables con R2.3).

Uso:
  python revision_IJIS/03_exp_ml_clasico.py --datos revision_IJIS/datos
Requisitos: scikit-learn, xgboost (pip install xgboost).
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun  # noqa: E402


def caracteristicas(meta):
    L = meta["longitudes"].astype(np.float32)          # n x 20
    T = meta["tiempos"].astype(np.float64)             # n x 20
    IAT = np.diff(T, axis=1).astype(np.float32)        # n x 19
    agg = []
    for M in (L, IAT):
        agg += [M.mean(1), M.std(1), M.min(1), M.max(1), np.median(M, 1)]
    agg += [(T[:, -1] - T[:, 0]).astype(np.float32), L.sum(1)]
    F = np.concatenate([L, IAT, np.stack(agg, 1)], axis=1)
    nombres = ([f"len_{i}" for i in range(20)] + [f"iat_{i}" for i in range(19)] +
               [f"{p}_{s}" for p in ("len", "iat") for s in ("media", "sd", "min", "max", "mediana")] +
               ["duracion", "bytes_total"])
    return F, nombres


def pesos_muestra(y_tr):
    frec = np.bincount(y_tr, minlength=len(comun.CLASES)).astype(np.float64)
    w = len(y_tr) / (len(comun.CLASES) * np.maximum(frec, 1))
    return w[y_tr]


def construir(tipo, params, semilla):
    if tipo == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=semilla, eval_metric="mlogloss", **params)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_jobs=-1, random_state=semilla, **params)


REJILLAS = {
    "xgboost": {"n_estimators": [200, 500], "max_depth": [6, 10], "learning_rate": [0.1]},
    "random_forest": {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 5]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(os.path.dirname(__file__), "datos"))
    ap.add_argument("--resultados", default=os.path.join(os.path.dirname(__file__), "resultados", "exp_R2.4_ml_clasico"))
    ap.add_argument("--modelos", nargs="+", default=["xgboost", "random_forest"])
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    _, y, meta = comun.cargar_datos(args.datos, mmap=True)
    if meta is None:
        sys.exit("Falta meta.npz: ejecutar 00_preprocesar_y_reproducir.py")
    F, nombres = caracteristicas(meta)
    idx_dev, idx_test, folds = comun.cargar_particion(os.path.join(args.datos, "particion.npz"))
    os.makedirs(args.resultados, exist_ok=True)
    filas = []

    for tipo in args.modelos:
        for ponderado in (False, True):
            nombre = f"{tipo}{'_balanced' if ponderado else ''}"
            print(f"\n######## {nombre} ########")
            rej = REJILLAS[tipo]
            combinaciones = [dict(zip(rej, v)) for v in itertools.product(*rej.values())]
            mejores = None
            busqueda = []
            for params in combinaciones:
                accs, f1s = [], []
                for tr, va in folds:
                    mdl = construir(tipo, params, args.semilla)
                    sw = pesos_muestra(y[tr]) if ponderado else None
                    mdl.fit(F[tr], y[tr], sample_weight=sw)
                    p = mdl.predict(F[va])
                    m = comun.metricas_desde_cm(comun.matriz_confusion(y[va], p))
                    accs.append(m["accuracy"])
                    f1s.append(m["macro_f1"])
                fila_b = dict(params=params, val_accuracy=float(np.mean(accs)), val_accuracy_sd=float(np.std(accs)),
                              val_macro_f1=float(np.mean(f1s)))
                busqueda.append(fila_b)
                print(f"  {params}  val_acc={fila_b['val_accuracy']*100:.2f}±{fila_b['val_accuracy_sd']*100:.2f}"
                      f"  val_macroF1={fila_b['val_macro_f1']*100:.2f}")
                if mejores is None or fila_b["val_accuracy"] > mejores["val_accuracy"]:
                    mejores = fila_b
            # Modelo final con todo el desarrollo
            mdl = construir(tipo, mejores["params"], args.semilla)
            t0 = time.time()
            mdl.fit(F[idx_dev], y[idx_dev], sample_weight=pesos_muestra(y[idx_dev]) if ponderado else None)
            minutos = (time.time() - t0) / 60
            t0 = time.perf_counter()
            y_pred = mdl.predict(F[idx_test])
            lat = (time.perf_counter() - t0) / len(idx_test) * 1000
            m = comun.metricas_desde_cm(comun.matriz_confusion(y[idx_test], y_pred))
            m["bootstrap_IC95"] = comun.bootstrap_ic(y[idx_test], y_pred)
            m.update(modelo=nombre, mejores_params=mejores["params"], busqueda=busqueda,
                     latencia_ms_por_semiflujo=lat, minutos_entrenamiento_final=minutos,
                     caracteristicas=nombres)
            if hasattr(mdl, "feature_importances_"):
                imp = np.asarray(mdl.feature_importances_)
                orden = np.argsort(-imp)[:15]
                m["importancias_top15"] = {nombres[i]: float(imp[i]) for i in orden}
            comun.guardar_json(os.path.join(args.resultados, nombre + ".json"), m)
            print(comun.resumen_texto(nombre, m))
            pc = m["por_clase"]
            filas.append(dict(modelo=nombre, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                              balanced_accuracy=m["balanced_accuracy"], kappa=m["kappa"], mcc=m["mcc"],
                              **{f"recall_{k}": v["sensibilidad"] for k, v in pc.items()},
                              latencia_ms_por_semiflujo=lat, params=json.dumps(mejores["params"])))

    ref = os.path.join(os.path.dirname(args.resultados), "modelo_articulo.json")
    if os.path.exists(ref):
        with open(ref, encoding="utf-8") as f:
            m = json.load(f)
        filas.append(dict(modelo="Transformer (artículo)", accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                          balanced_accuracy=m["balanced_accuracy"], kappa=m["kappa"], mcc=m["mcc"],
                          **{f"recall_{k}": v["sensibilidad"] for k, v in m["por_clase"].items()},
                          latencia_ms_por_semiflujo=m.get("latencia_ms_por_semiflujo"), params=""))
    ruta = os.path.join(args.resultados, "tabla_R2.4.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for r in filas:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nTabla comparativa: {ruta}")


if __name__ == "__main__":
    main()
