"""
Utilidades comunes para los experimentos de la revisión de IJIS ("From Packets to Tokens").

Todo lo que aquí se hace replica el protocolo del artículo (cuaderno "Transformer entrenamiento.ipynb"):
  - partición 90/10 estratificada con train_test_split(random_state=50) usando las etiquetas one-hot,
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=50) sobre el 90 %,
  - normalización de cada paquete por su máximo (equivale a dividir entre 65535 por el token FFFF),
  - el modelo evaluado en test es el del fold con mayor accuracy de validación en la última época.

Los datos se leen de la carpeta generada por 00_preprocesar_y_reproducir.py:
  X.npy (uint16, N x 20 x 201), y.npy (int8, N), meta.npz (longitudes de payload y tiempos por paquete).
"""
import json
import math
import os
import time

import numpy as np

CLASES = ["Chat", "Email", "FileTransfer", "Streaming", "VoIP"]
SEMILLA_PARTICION = 50
N_FOLDS = 5

# Hiperparámetros del modelo del artículo (grid_XFMR.csv, HS_d86bd8c1..., HM_1c66f54d..., 22-06-2023)
HP_ARTICULO = dict(epochs=15, batch_size=100, learning_rate=3e-4, num_heads=8, head_size=6, ff_dim=256,
                   mlp_units=250, dropout=0.2, mlp_dropout=0.2, num_capas_atencion=1, positional_encoding=False)


# ----------------------------------------------------------------------------------------------------------------
# Datos y partición
# ----------------------------------------------------------------------------------------------------------------

def cargar_datos(carpeta, mmap=False):
    X = np.load(os.path.join(carpeta, "X.npy"), mmap_mode="r" if mmap else None)
    y = np.load(os.path.join(carpeta, "y.npy")).astype(np.int64)
    meta_path = os.path.join(carpeta, "meta.npz")
    meta = dict(np.load(meta_path)) if os.path.exists(meta_path) else None
    return X, y, meta


def one_hot(y, n_clases=len(CLASES)):
    # Igual que keras.utils.to_categorical (float32), que es lo que se usó para estratificar en el artículo.
    oh = np.zeros((len(y), n_clases), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def particion_articulo(y):
    """Devuelve (idx_desarrollo, idx_test, lista_folds) con índices sobre el array completo.

    Reproduce exactamente las llamadas del cuaderno original:
      train_test_split(array_semiflujos, array_etiquetas, test_size=0.1, stratify=array_etiquetas,
                       shuffle=True, random_state=50)
      StratifiedKFold(n_splits=5, shuffle=True, random_state=50).split(X_train_bruto, argmax(y_train_bruto))
    La estratificación se hace con las etiquetas one-hot (no con enteros), porque el orden de clases que usa
    scikit-learn cambia la permutación resultante.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    oh = one_hot(y)
    idx = np.arange(len(y))
    idx_dev, idx_test, oh_dev, _ = train_test_split(idx, oh, test_size=0.1, stratify=oh, shuffle=True,
                                                    random_state=SEMILLA_PARTICION)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA_PARTICION)
    folds = []
    for tr, va in skf.split(idx_dev, np.argmax(oh_dev, axis=1)):
        folds.append((idx_dev[tr], idx_dev[va]))
    return idx_dev, idx_test, folds


def guardar_particion(ruta, idx_dev, idx_test, folds):
    d = {"idx_dev": idx_dev, "idx_test": idx_test}
    for k, (tr, va) in enumerate(folds):
        d[f"fold{k}_train"] = tr
        d[f"fold{k}_val"] = va
    np.savez_compressed(ruta, **d)


def cargar_particion(ruta):
    d = np.load(ruta)
    folds = [(d[f"fold{k}_train"], d[f"fold{k}_val"]) for k in range(N_FOLDS)]
    return d["idx_dev"], d["idx_test"], folds


def normalizar(Xb):
    """Normalización del artículo: cada paquete dividido por su máximo (sklearn normalize(norm='max'))."""
    Xb = Xb.astype(np.float32)
    m = np.abs(Xb).max(axis=2, keepdims=True)
    m[m == 0] = 1.0
    return Xb / m


# ----------------------------------------------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def matriz_confusion(y_true, y_pred, n_clases=len(CLASES)):
    cm = np.zeros((n_clases, n_clases), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def metricas_desde_cm(cm):
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    sop = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    rec = np.divide(tp, sop, out=np.zeros_like(tp), where=sop > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    esp = np.array([(n - sop[i] - pred[i] + tp[i]) / (n - sop[i]) for i in range(len(tp))])
    acc = tp.sum() / n
    pe = (sop * pred).sum() / n ** 2
    kappa = (acc - pe) / (1 - pe)
    num = tp.sum() * n - (sop * pred).sum()
    den = math.sqrt(float(n * n - (pred ** 2).sum())) * math.sqrt(float(n * n - (sop ** 2).sum()))
    mcc = num / den if den > 0 else 0.0
    w = sop / n
    por_clase = {}
    for i, c in enumerate(CLASES[:len(tp)]):
        lo, hi = wilson(int(tp[i]), int(sop[i]))
        por_clase[c] = dict(n=int(sop[i]), precision=prec[i], sensibilidad=rec[i], sensibilidad_IC95=[lo, hi],
                            especificidad=esp[i], f1=f1[i])
    return dict(
        n=int(n), accuracy=acc, accuracy_IC95=list(wilson(int(tp.sum()), int(n))),
        balanced_accuracy=rec.mean(), macro_precision=prec.mean(), macro_f1=f1.mean(),
        weighted_precision=(prec * w).sum(), weighted_f1=(f1 * w).sum(), weighted_especificidad=(esp * w).sum(),
        kappa=kappa, mcc=mcc, por_clase=por_clase, matriz_confusion=cm.astype(int).tolist(),
    )


def bootstrap_ic(y_true, y_pred, n_rep=1000, semilla=0):
    """IC 95 % por bootstrap estratificado (por clase) para accuracy, macro-F1, kappa y MCC."""
    rng = np.random.default_rng(semilla)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    idx_por_clase = [np.where(y_true == c)[0] for c in range(len(CLASES))]
    vals = {k: [] for k in ["accuracy", "macro_f1", "balanced_accuracy", "kappa", "mcc"]}
    for _ in range(n_rep):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_por_clase if len(ix)])
        m = metricas_desde_cm(matriz_confusion(y_true[idx], y_pred[idx]))
        for k in vals:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def f1_macro(y_true, y_pred):
    return metricas_desde_cm(matriz_confusion(y_true, y_pred))["macro_f1"]


def a_json(obj):
    if isinstance(obj, dict):
        return {k: a_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return a_json(obj.tolist())
    return obj


def guardar_json(ruta, obj):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(a_json(obj), f, indent=2, ensure_ascii=False)


def resumen_texto(nombre, m):
    lin = [f"== {nombre} ==",
           f"accuracy={m['accuracy']*100:.2f}%  macro-F1={m['macro_f1']*100:.2f}%  balanced-acc={m['balanced_accuracy']*100:.2f}%"
           f"  kappa={m['kappa']:.3f}  MCC={m['mcc']:.3f}"]
    for c, d in m["por_clase"].items():
        lo, hi = d["sensibilidad_IC95"]
        lin.append(f"  {c:13s} n={d['n']:6d}  P={d['precision']*100:6.2f}  R={d['sensibilidad']*100:6.2f} "
                   f"[{lo*100:.1f}-{hi*100:.1f}]  F1={d['f1']*100:6.2f}  Esp={d['especificidad']*100:6.2f}")
    return "\n".join(lin)


# ----------------------------------------------------------------------------------------------------------------
# Keras: generador por lotes (evita cargar 11 GB en float32) y entrenamiento con el protocolo del artículo
# ----------------------------------------------------------------------------------------------------------------

def crear_secuencia(X, y, indices, batch_size, barajar, n_clases=len(CLASES), semilla=0, epoch_inicial=0):
    """Secuencia por lotes para UNA época. Si barajar=True, el orden depende solo de (semilla, epoch_inicial) y no
    cambia durante la vida del objeto: entrenar_kfold crea una secuencia nueva por época. Así un entrenamiento
    reanudado desde un checkpoint ve exactamente el mismo orden de datos, sin depender de cuándo llama Keras 3 a
    on_epoch_end (lo hace también al reiniciar su iterador)."""
    from tensorflow import keras

    class Secuencia(keras.utils.Sequence):
        def __init__(self):
            try:
                super().__init__()
            except TypeError:
                pass
            self.base = np.array(indices)
            self.epoch = epoch_inicial
            self._barajar()

        def _barajar(self):
            self.idx = self.base.copy()
            if barajar:
                np.random.default_rng([semilla, self.epoch]).shuffle(self.idx)

        def __len__(self):
            return int(math.ceil(len(self.idx) / batch_size))

        def __getitem__(self, i):
            b = np.sort(self.idx[i * batch_size:(i + 1) * batch_size])
            xb = normalizar(X[b])
            if y is None:
                return xb
            return xb, one_hot(y[b], n_clases)

        def on_epoch_end(self):
            pass  # orden fijo por objeto; ver docstring

    return Secuencia()


def predecir(model, X, indices, batch_size=1000):
    probs = []
    for i in range(0, len(indices), batch_size):
        b = indices[i:i + batch_size]
        probs.append(model.predict(normalizar(X[b]), verbose=0))
    return np.concatenate(probs, axis=0)


def latencia_ms_por_semiflujo(model, X, indices, n=2000, batch_size=1000, repeticiones=3):
    sel = np.array(indices[:n])
    xb = normalizar(X[np.sort(sel)])
    model.predict(xb[:batch_size], verbose=0)  # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        model.predict(xb, batch_size=batch_size, verbose=0)
        tiempos.append((time.perf_counter() - t0) / len(xb) * 1000)
    return float(np.median(tiempos))


def _guardar_atomico(ruta, obj):
    import pickle
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    tmp = ruta + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ruta)  # un corte durante la escritura nunca deja un checkpoint a medias


def ruta_checkpoint(experimento, nombre):
    """Checkpoints fuera de Dropbox (se reescriben en cada época): %LOCALAPPDATA%/ijis_checkpoints/..."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ijis_checkpoints", experimento, nombre + ".pkl")


def borrar_checkpoint(ruta):
    if ruta and os.path.exists(ruta):
        os.remove(ruta)


def entrenar_kfold(construir_modelo, X, y, folds, idx_test, epochs, batch_size, criterio="accuracy",
                   construir_secuencia_train=None, semilla=0, verbose=1, ruta_ckpt=None):
    """Replica entrenar_transformer_kfold: mismos pesos iniciales en cada fold, se queda con el mejor fold.

    construir_modelo() -> modelo compilado.
    construir_secuencia_train(idx_train, k, epoch_inicial) permite cambiar el muestreo (sobremuestreo en Exp. 3).
    criterio: 'accuracy' (como el artículo) o 'macro_f1' para elegir el mejor fold.
    ruta_ckpt: si se indica, al final de cada época se guarda el estado completo (variables del modelo, incluido
      el estado aleatorio del dropout, variables del optimizador, folds terminados y mejor fold). Si el proceso se
      interrumpe, al relanzar se continúa desde la última época terminada en lugar de empezar de cero.
      Como en el original, el optimizador no se reinicia entre folds; el checkpoint conserva también ese estado.
    Devuelve dict con el modelo del mejor fold, métricas por fold y probabilidades en test.
    """
    import pickle

    import tensorflow as tf

    tf.keras.utils.set_random_seed(semilla)
    model = construir_modelo()
    pesos_iniciales = model.get_weights()
    res_folds = []
    mejor = None
    fold_ini, epoch_ini, logs_ult, minutos_previos = 0, 0, None, 0.0
    estado = None
    if ruta_ckpt and os.path.exists(ruta_ckpt):
        with open(ruta_ckpt, "rb") as f:
            estado = pickle.load(f)
        pesos_iniciales = estado["pesos_iniciales"]
        res_folds, mejor = estado["res_folds"], estado["mejor"]
        fold_ini, epoch_ini, logs_ult = estado["fold"], estado["epoch"], estado["logs_ult"]
        minutos_previos = estado["minutos"]
        print(f"[checkpoint] reanudando en fold {fold_ini + 1}/{len(folds)}, época {epoch_ini}/{epochs}", flush=True)

    t0 = time.time()

    def restaurar_variables(est):
        # El optimizador crea sus variables al construirse; se construye antes de asignarlas.
        if not model.optimizer.built:
            model.optimizer.build(model.trainable_variables)
        for v, val in zip(model.variables, est["vars_modelo"]):
            v.assign(val)
        for v, val in zip(model.optimizer.variables, est["vars_opt"]):
            v.assign(val)

    def guardar(fold, epoch, logs):
        if not ruta_ckpt:
            return
        _guardar_atomico(ruta_ckpt, dict(
            pesos_iniciales=pesos_iniciales, res_folds=res_folds, mejor=mejor, fold=fold, epoch=epoch,
            logs_ult=logs, minutos=minutos_previos + (time.time() - t0) / 60,
            vars_modelo=[v.numpy() for v in model.variables],
            vars_opt=[v.numpy() for v in model.optimizer.variables] if model.optimizer.built else []))

    for k, (tr, va) in enumerate(folds):
        if k < fold_ini:
            continue
        e0 = epoch_ini if k == fold_ini else 0
        if estado is not None and k == fold_ini:
            # Estado exacto guardado (a mitad de fold, o al inicio de un fold tras terminar el anterior)
            restaurar_variables(estado)
            if e0 == 0:
                model.set_weights(pesos_iniciales)
        else:
            model.set_weights(pesos_iniciales)
        seq_va = crear_secuencia(X, y, va, batch_size, False)
        ult = logs_ult if e0 >= epochs else None  # corte tras la última época, antes de cerrar el fold
        for e in range(e0, epochs):
            # Una llamada a fit por época, con una secuencia cuyo orden depende solo de (semilla, fold, época)
            if construir_secuencia_train is None:
                seq_tr = crear_secuencia(X, y, tr, batch_size, True, semilla=semilla + k, epoch_inicial=e)
            else:
                seq_tr = construir_secuencia_train(tr, k, e)
            if verbose:
                print(f"Epoch {e + 1}/{epochs}", flush=True)
            t_ep = time.time()
            # shuffle=False: la secuencia ya baraja de forma determinista; con True, Keras reordenaría los lotes
            # con el módulo global `random` de Python, cuyo estado no se puede recuperar tras un corte.
            hist = model.fit(seq_tr, validation_data=seq_va, epochs=1, verbose=0, shuffle=False)
            ult = {kk: float(v[-1]) for kk, v in hist.history.items()}
            if verbose:
                print(f"{len(seq_tr)}/{len(seq_tr)} - {time.time() - t_ep:.0f}s - "
                      + " - ".join(f"{kk}: {v:.4f}" for kk, v in ult.items()), flush=True)
            guardar(k, e + 1, ult)
        p_va = predecir(model, X, va)
        y_va_pred = p_va.argmax(1)
        m_va = metricas_desde_cm(matriz_confusion(y[va], y_va_pred))
        r = dict(fold=k, train_accuracy=float(ult["accuracy"]), train_loss=float(ult["loss"]),
                 val_accuracy=float(ult["val_accuracy"]), val_loss=float(ult["val_loss"]),
                 val_macro_f1=m_va["macro_f1"])
        res_folds.append(r)
        valor = r["val_accuracy"] if criterio == "accuracy" else r["val_macro_f1"]
        if mejor is None or valor > mejor[0]:
            mejor = (valor, k, model.get_weights(), p_va)
        guardar(k + 1, 0, None)  # fold cerrado: al reanudar se empieza el siguiente
    minutos = minutos_previos + (time.time() - t0) / 60
    model.set_weights(mejor[2])
    p_test = predecir(model, X, idx_test)
    resumen = {}
    for clave in ["train_accuracy", "train_loss", "val_accuracy", "val_loss", "val_macro_f1"]:
        v = np.array([r[clave] for r in res_folds])
        resumen[clave] = dict(media=float(v.mean()), desviacion=float(v.std()))
    return dict(model=model, folds=res_folds, resumen_folds=resumen, mejor_fold=mejor[1],
                p_val_mejor=mejor[3], idx_val_mejor=folds[mejor[1]][1], p_test=p_test, minutos=minutos)
