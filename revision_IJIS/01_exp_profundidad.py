"""
R2.2 — Profundidad del encoder y positional encoding (Exp. 2 del plan de respuesta).

Entrena el Transformer del artículo con 1, 2, 4 y 6 capas de encoder, con y sin positional encoding (ya corregido
en src/modelos.py), usando exactamente el mismo protocolo que el artículo: misma partición (random_state=50),
5 folds estratificados, mismos pesos iniciales en cada fold, mismos hiperparámetros (15 épocas, batch 100,
lr 3e-4, 8 cabezas de tamaño 6, FFN 256, MLP 250, dropout 0,2) y evaluación en test del mejor fold.

Para cada configuración guarda: métricas de folds (media ± desviación), métricas de test completas (accuracy,
macro-F1, balanced accuracy, kappa, MCC, por clase con IC de Wilson), nº de parámetros y latencia por semi-flujo.

Uso:
  python revision_IJIS/01_exp_profundidad.py --datos revision_IJIS/datos
  Opciones: --capas 1 2 4 6  --pe no si  --semillas 0 1 2  --escala-pe 1.0  --epochs 15
Requisito: haber ejecutado antes 00_preprocesar_y_reproducir.py (genera X.npy, y.npy y particion.npz).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(os.path.dirname(__file__), "datos"))
    ap.add_argument("--resultados", default=os.path.join(os.path.dirname(__file__), "resultados", "exp_R2.2_profundidad"))
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--capas", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--pe", nargs="+", default=["no", "si"], choices=["no", "si"])
    ap.add_argument("--escala-pe", type=float, default=1.0)
    ap.add_argument("--semillas", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=comun.HP_ARTICULO["epochs"])
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    sys.path.insert(0, args.repo)
    from src.modelos import build_keras_transformer_model

    X, y, _ = comun.cargar_datos(args.datos)
    idx_dev, idx_test, folds = comun.cargar_particion(os.path.join(args.datos, "particion.npz"))
    hp = comun.HP_ARTICULO
    os.makedirs(args.resultados, exist_ok=True)
    filas = []

    for capas in args.capas:
        for pe in args.pe:
            for semilla in args.semillas:
                nombre = f"capas{capas}_PE-{pe}_semilla{semilla}"
                ruta_json = os.path.join(args.resultados, nombre + ".json")
                if os.path.exists(ruta_json):
                    print(f"[saltando] {nombre} ya existe")
                    m = comun_cargar(ruta_json)
                else:
                    print(f"\n######## {nombre} ########")

                    def construir():
                        model = build_keras_transformer_model(
                            X.shape[1:], len(comun.CLASES), hp["head_size"], hp["num_heads"], hp["ff_dim"],
                            hp["mlp_units"], pe == "si", hp["dropout"], hp["mlp_dropout"], capas,
                            escala_PE=args.escala_pe)
                        model.compile(loss="categorical_crossentropy",
                                      optimizer=keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
                                      metrics=["accuracy"])
                        return model

                    ckpt = comun.ruta_checkpoint("R2.2", nombre)
                    r = comun.entrenar_kfold(construir, X, y, folds, idx_test, args.epochs, hp["batch_size"],
                                             semilla=semilla, verbose=args.verbose, ruta_ckpt=ckpt)
                    y_pred = r["p_test"].argmax(1)
                    m = comun.metricas_desde_cm(comun.matriz_confusion(y[idx_test], y_pred))
                    m.update(config=dict(capas=capas, positional_encoding=pe, escala_pe=args.escala_pe,
                                         semilla=semilla, epochs=args.epochs,
                                         **{k: v for k, v in hp.items()
                                            if k not in ("epochs", "positional_encoding", "num_capas_atencion")}),
                             parametros=int(r["model"].count_params()), folds=r["folds"],
                             resumen_folds=r["resumen_folds"], mejor_fold=r["mejor_fold"],
                             minutos_entrenamiento=r["minutos"],
                             latencia_ms_por_semiflujo=comun.latencia_ms_por_semiflujo(r["model"], X, idx_test),
                             dispositivo=str(tf.config.list_physical_devices("GPU") or "CPU"))
                    comun.guardar_json(ruta_json, m)
                    np.save(os.path.join(args.resultados, nombre + "_ptest.npy"), r["p_test"])
                    print(comun.resumen_texto(nombre, m))
                    comun.borrar_checkpoint(ckpt)  # el JSON ya está guardado
                    keras.backend.clear_session()
                filas.append(fila(m))

    escribir_tabla(os.path.join(args.resultados, "tabla_R2.2.csv"), filas)
    print(f"\nTabla resumen: {os.path.join(args.resultados, 'tabla_R2.2.csv')}")


def comun_cargar(ruta):
    import json
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def fila(m):
    c = m["config"]
    pc = m["por_clase"]
    return dict(capas=c["capas"], positional_encoding=c["positional_encoding"], semilla=c["semilla"],
                parametros=m["parametros"],
                val_accuracy_media=m["resumen_folds"]["val_accuracy"]["media"],
                val_accuracy_sd=m["resumen_folds"]["val_accuracy"]["desviacion"],
                test_accuracy=m["accuracy"], test_macro_f1=m["macro_f1"],
                test_balanced_accuracy=m["balanced_accuracy"], kappa=m["kappa"], mcc=m["mcc"],
                **{f"recall_{k}": v["sensibilidad"] for k, v in pc.items()},
                latencia_ms_por_semiflujo=m["latencia_ms_por_semiflujo"],
                minutos=m["minutos_entrenamiento"])


def escribir_tabla(ruta, filas):
    if not filas:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for r in filas:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
