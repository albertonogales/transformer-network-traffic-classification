"""
R2.3 — Mitigación del desbalance de clases (Exp. 3 del plan de respuesta).

Compara, con el mismo protocolo del artículo (partición random_state=50, 5 folds, mejor fold -> test):
  base            : cross-entropy sin ponderar (= modelo del artículo, reentrenado para tener referencia propia)
  pesos_inv       : cross-entropy ponderada con pesos inversos a la frecuencia
  pesos_raiz      : cross-entropy ponderada con la raíz cuadrada de los pesos inversos (menos agresiva)
  focal_g1, focal_g2 : focal loss (gamma 1 y 2) con alfa por clase = pesos_raiz
  sobremuestreo   : réplica aleatoria de Email y Chat SOLO en el conjunto de entrenamiento de cada fold
  logit_adj       : ajuste de logits a posteriori sobre el modelo base (tau elegido en validación por macro-F1)

Los pesos se normalizan para que el peso medio por muestra sea 1 (así la tasa de aprendizaje es comparable).
No se usa SMOTE: interpolar secuencias de bytes no tiene sentido.

Uso:
  python revision_IJIS/02_exp_desbalance.py --datos revision_IJIS/datos
  Opciones: --metodos base pesos_raiz focal_g2 ...  --capas 1 (o la mejor de R2.2)  --pe no
            --fraccion-minima 0.02 (fracción objetivo de Email y de Chat tras sobremuestrear)
            --criterio accuracy|macro_f1 (cómo se elige el mejor fold; accuracy = como el artículo)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun  # noqa: E402

METODOS = ["base", "pesos_inv", "pesos_raiz", "focal_g1", "focal_g2", "focal_g2_sin_alfa", "sobremuestreo",
           "logit_adj"]
# focal_g2_sin_alfa: focal loss "pura" (gamma=2, alfa=1 para todas las clases), es decir, solo el escalado dinámico
# según la confianza de la predicción que pide el revisor 2, sin ponderación por clase (que se evalúa aparte en
# pesos_inv). Las variantes focal_g1/focal_g2 usan además alfa = pesos_raiz.


def pesos_clase(frecuencias, potencia):
    f = np.asarray(frecuencias, dtype=np.float64)
    f = f / f.sum()
    w = (1.0 / f) ** potencia
    w = w / (f * w).sum()  # peso medio por muestra = 1
    return w.astype(np.float32)


def perdida_ponderada(w):
    import tensorflow as tf
    wt = tf.constant(w, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        return -tf.reduce_sum(y_true * wt * tf.math.log(y_pred), axis=-1)
    return loss


def perdida_focal(alfa, gamma):
    import tensorflow as tf
    at = tf.constant(alfa, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        return -tf.reduce_sum(y_true * at * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred), axis=-1)
    return loss


def indices_sobremuestreados(idx_tr, y, fraccion_minima, rng):
    """Replica las clases minoritarias (Email, Chat) hasta que cada una sea al menos `fraccion_minima` del total."""
    idx_tr = np.asarray(idx_tr)
    n = len(idx_tr)
    partes = [idx_tr]
    for c in (comun.CLASES.index("Email"), comun.CLASES.index("Chat")):
        ic = idx_tr[y[idx_tr] == c]
        objetivo = int(np.ceil(fraccion_minima * n))
        if len(ic) and len(ic) < objetivo:
            partes.append(rng.choice(ic, size=objetivo - len(ic), replace=True))
    return np.concatenate(partes)


def logit_adjustment(p_val, y_val, p_test, prior):
    lp = np.log(np.clip(prior, 1e-12, 1))
    mejor = (-1, 0.0)
    for tau in np.round(np.arange(0.0, 2.01, 0.1), 2):
        pred = (np.log(np.clip(p_val, 1e-12, 1)) - tau * lp).argmax(1)
        f = comun.f1_macro(y_val, pred)
        if f > mejor[0]:
            mejor = (f, tau)
    tau = mejor[1]
    return (np.log(np.clip(p_test, 1e-12, 1)) - tau * lp).argmax(1), tau, mejor[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(os.path.dirname(__file__), "datos"))
    ap.add_argument("--resultados", default=os.path.join(os.path.dirname(__file__), "resultados", "exp_R2.3_desbalance"))
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--metodos", nargs="+", default=METODOS, choices=METODOS)
    ap.add_argument("--capas", type=int, default=1)
    ap.add_argument("--pe", default="no", choices=["no", "si"])
    ap.add_argument("--fraccion-minima", type=float, default=0.02)
    ap.add_argument("--criterio", default="accuracy", choices=["accuracy", "macro_f1"])
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=comun.HP_ARTICULO["epochs"])
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args()

    from tensorflow import keras

    sys.path.insert(0, args.repo)
    from src.modelos import build_keras_transformer_model

    X, y, _ = comun.cargar_datos(args.datos)
    idx_dev, idx_test, folds = comun.cargar_particion(os.path.join(args.datos, "particion.npz"))
    hp = comun.HP_ARTICULO
    frec = np.bincount(y[idx_dev], minlength=len(comun.CLASES))
    prior = frec / frec.sum()
    w_inv, w_raiz = pesos_clase(frec, 1.0), pesos_clase(frec, 0.5)
    print("Pesos inversos:", dict(zip(comun.CLASES, np.round(w_inv, 3))))
    print("Pesos raíz:   ", dict(zip(comun.CLASES, np.round(w_raiz, 3))))
    os.makedirs(args.resultados, exist_ok=True)

    def construir_con(perdida):
        def construir():
            model = build_keras_transformer_model(
                X.shape[1:], len(comun.CLASES), hp["head_size"], hp["num_heads"], hp["ff_dim"], hp["mlp_units"],
                args.pe == "si", hp["dropout"], hp["mlp_dropout"], args.capas)
            model.compile(loss=perdida, optimizer=keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
                          metrics=["accuracy"])
            return model
        return construir

    perdidas = {"base": "categorical_crossentropy", "pesos_inv": perdida_ponderada(w_inv),
                "pesos_raiz": perdida_ponderada(w_raiz), "focal_g1": perdida_focal(w_raiz, 1.0),
                "focal_g2": perdida_focal(w_raiz, 2.0),
                "focal_g2_sin_alfa": perdida_focal(np.ones(len(comun.CLASES), dtype=np.float32), 2.0),
                "sobremuestreo": "categorical_crossentropy"}

    filas = []
    base_res = None
    for metodo in args.metodos:
        ruta_json = os.path.join(args.resultados, f"{metodo}.json")
        # 'base' se puede saltar si existen también sus probabilidades (logit_adj las carga de disco)
        base_completo = all(os.path.exists(os.path.join(args.resultados, f)) for f in
                            ("base_ptest.npy", "base_pval.npy", "base_idxval.npy"))
        if os.path.exists(ruta_json) and (metodo != "base" or base_completo):
            with open(ruta_json, encoding="utf-8") as f:
                m = json.load(f)
            filas.append(fila(metodo, m))
            print(f"[saltando] {metodo} ya existe")
            continue
        if metodo == "logit_adj":
            if base_res is None:
                ruta_b = os.path.join(args.resultados, "base_ptest.npy")
                if not os.path.exists(ruta_b):
                    print("logit_adj necesita ejecutar antes 'base' en la misma llamada o en una anterior")
                    continue
                base_res = dict(p_test=np.load(ruta_b),
                                p_val_mejor=np.load(os.path.join(args.resultados, "base_pval.npy")),
                                idx_val_mejor=np.load(os.path.join(args.resultados, "base_idxval.npy")))
            y_pred, tau, f_val = logit_adjustment(base_res["p_val_mejor"], y[base_res["idx_val_mejor"]],
                                                  base_res["p_test"], prior)
            m = comun.metricas_desde_cm(comun.matriz_confusion(y[idx_test], y_pred))
            m.update(metodo=metodo, tau=tau, val_macro_f1=f_val)
        else:
            print(f"\n######## {metodo} ########")
            seq_train = None
            if metodo == "sobremuestreo":
                def seq_train(tr, k, epoch_inicial=0):
                    rng = np.random.default_rng(args.semilla + k)
                    idx = indices_sobremuestreados(tr, y, args.fraccion_minima, rng)
                    return comun.crear_secuencia(X, y, idx, hp["batch_size"], True, semilla=args.semilla + k,
                                                 epoch_inicial=epoch_inicial)
            ckpt = comun.ruta_checkpoint("R2.3", f"{metodo}_capas{args.capas}_PE-{args.pe}_{args.criterio}")
            r = comun.entrenar_kfold(construir_con(perdidas[metodo]), X, y, folds, idx_test, args.epochs,
                                     hp["batch_size"], criterio=args.criterio, construir_secuencia_train=seq_train,
                                     semilla=args.semilla, verbose=args.verbose, ruta_ckpt=ckpt)
            y_pred = r["p_test"].argmax(1)
            m = comun.metricas_desde_cm(comun.matriz_confusion(y[idx_test], y_pred))
            m.update(metodo=metodo, folds=r["folds"], resumen_folds=r["resumen_folds"], mejor_fold=r["mejor_fold"],
                     minutos_entrenamiento=r["minutos"], capas=args.capas, pe=args.pe, criterio=args.criterio,
                     fraccion_minima=args.fraccion_minima if metodo == "sobremuestreo" else None)
            np.save(os.path.join(args.resultados, f"{metodo}_ptest.npy"), r["p_test"])
            # Pesos del mejor fold (solo salida; no altera el protocolo) para reutilizarlos con las capturas en vivo
            import pickle
            with open(os.path.join(args.resultados, f"{metodo}_capas{args.capas}_PE-{args.pe}_pesos.pkl"), "wb") as fp:
                pickle.dump(dict(pesos=r["model"].get_weights(), capas=args.capas, pe=args.pe, metodo=metodo,
                                 mejor_fold=r["mejor_fold"], hp=hp, input_shape=X.shape[1:]), fp)
            if metodo == "base":
                base_res = r
                np.save(os.path.join(args.resultados, "base_pval.npy"), r["p_val_mejor"])
                np.save(os.path.join(args.resultados, "base_idxval.npy"), r["idx_val_mejor"])
            keras.backend.clear_session()
        m["bootstrap_IC95"] = comun.bootstrap_ic(y[idx_test], y_pred)
        comun.guardar_json(ruta_json, m)
        if metodo != "logit_adj":
            comun.borrar_checkpoint(ckpt)  # el JSON ya está guardado
        print(comun.resumen_texto(metodo, m))
        filas.append(fila(metodo, m))

    ruta = os.path.join(args.resultados, "tabla_R2.3.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for r in filas:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nTabla resumen: {ruta}")
    print("Recordatorio: la prueba de Email en tiempo real con el mejor método se hace aparte (ver README).")


def fila(metodo, m):
    pc = m["por_clase"]
    return dict(metodo=metodo, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                balanced_accuracy=m["balanced_accuracy"], kappa=m["kappa"], mcc=m["mcc"],
                email_recall=pc["Email"]["sensibilidad"], email_precision=pc["Email"]["precision"],
                chat_recall=pc["Chat"]["sensibilidad"], chat_precision=pc["Chat"]["precision"],
                streaming_recall=pc["Streaming"]["sensibilidad"], filetransfer_recall=pc["FileTransfer"]["sensibilidad"],
                voip_recall=pc["VoIP"]["sensibilidad"], tau=m.get("tau"))


if __name__ == "__main__":
    main()
