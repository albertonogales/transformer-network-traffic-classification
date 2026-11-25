"""
Paso 0 de la revisión de IJIS: regenerar los semi-flujos del artículo, comprobar que son los mismos y reproducir
el 93,11 % con los pesos guardados. Deja preparados los datos y la partición para los experimentos de R2.2-R2.4.

Qué hace
  1. Recorre los PCAP de ISCXVPN2016 (non-VPN) por clase y llama a TShark con los mismos campos que el código
     original (procesamiento_pcap.generar_CSV_Tshark), en streaming y sin escribir CSV gigantes.
  2. Aplica exactamente la lógica de procesarDF / obtenerFlujos / obtenerSemiflujos / procesar_RAW /
     desechar_semiflujos_repetidos (configuración d86bd8c1: min_paquetes_flujo=20, 20 paquetes por semi-flujo,
     30 s, 800 caracteres hex, tokens de 4, separador FFFF). Además guarda, alineados con cada semi-flujo,
     la longitud real del payload y el instante de cada paquete (necesarios para el baseline de R2.4).
  3. Comprueba los recuentos por clase contra los del artículo (Chat 1325, Email 168, FileTransfer 266577,
     Streaming 207405, VoIP 214180).
  4. Reproduce la partición (random_state=50), carga los pesos del artículo, evalúa en test y comprueba el 93,11 %.
  5. Guarda X.npy, y.npy, meta.npz, particion.npz y las métricas completas del modelo del artículo.

Etiquetado de las videollamadas
  La Tabla 1 del artículo (Streaming: 30 PCAP / 7 aplicaciones; VoIP: 23 PCAP / 4 aplicaciones) solo cuadra si las
  11 capturas de videollamada (facebook_video*, hangouts_video*, skype_video*) se etiquetaron como Streaming.
  En la carpeta actual de Dropbox están dentro de VoIP. Por defecto el script las etiqueta como Streaming
  (--videollamadas Streaming); si los recuentos no cuadran, probar --videollamadas VoIP.

Uso (desde la raíz del repositorio, en Windows con Wireshark instalado):
  python revision_IJIS/00_preprocesar_y_reproducir.py ^
      --pcaps "C:/Users/Alberto Nogales/Dropbox/01_UAH/Tesis Activas/Arancha Ciprian/Papers/Enviados/43 Traffic network NLP/ISCXVPN2016_PCAPs" ^
      --salida revision_IJIS/datos ^
      --tshark "C:/Program Files/Wireshark/tshark.exe"

  Opciones útiles: --solo-reproducir (si ya existen X.npy/y.npy), --procesos 4 (PCAP en paralelo).
Requisitos: numpy, pandas, scikit-learn, tensorflow (para el paso 4) y TShark.
Tiempo y memoria: los PCAP suman ~20 GB; cuente con varias horas y ~16 GB de RAM (el array final ocupa ~5,5 GB en uint16).
"""
import argparse
import csv
import hashlib
import io
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun  # noqa: E402

# ---- Configuración del artículo (caracteristicas_semiflujos.csv, hash d86bd8c1f06c6599b5d3ce9b434345d8) ----
MIN_PAQUETES_FLUJO = 20
NUM_PAQUETES_SEMIFLUJO = 20
TIEMPO_MAXIMO_SEMIFLUJO = 30.0
LONGITUD_RAW = 800
NUM_HEX_TOKEN = 4
TOKEN_SEPARADOR = 65535
N_TOKENS = LONGITUD_RAW // NUM_HEX_TOKEN + 1  # 201
RECUENTOS_ARTICULO = {"Chat": 1325, "Email": 168, "FileTransfer": 266577, "Streaming": 207405, "VoIP": 214180}
ACC_TEST_ARTICULO = 0.9311

# procesamiento_pcap.procesarDF: protocolos que se conservan (columna Protocol de Wireshark)
PROTOCOLOS = {"TCP", "UDP", "SSL", "SSLv2", "TLSv1", "TLSv1.1", "TLSv1.2", "TLSv1.3", "SSHv2", "GTP <TCP>",
              "GTP <UDP>", "GTP <QUIC>", "GTP <SSL>", "GTP <SSLv2>", "GTP <TLSv1>", "GTP <TLSv1.1>",
              "GTP <TLSv1.2>", "GTP <TLSv1.3>"}
PROTOCOLOS_UDP = {"UDP", "GTP <QUIC>"}

EXTENSIONES = (".pcap", ".pcapng")


def campos_tshark(tshark):
    """Wireshark >= 4.2 renombró los campos de columna (_ws.col.No. -> _ws.col.number, etc.)."""
    nuevos = ["_ws.col.number", "frame.time_relative", "_ws.col.def_src", "_ws.col.def_dst", "_ws.col.protocol",
              "tcp.srcport", "tcp.dstport", "tcp.len", "udp.srcport", "udp.dstport", "tcp.payload", "udp.payload"]
    antiguos = ["_ws.col.No.", "frame.time_relative", "_ws.col.Source", "_ws.col.Destination", "_ws.col.Protocol",
                "tcp.srcport", "tcp.dstport", "tcp.len", "udp.srcport", "udp.dstport", "tcp.payload", "udp.payload"]
    try:
        out = subprocess.run([tshark, "-G", "fields"], capture_output=True, text=True, errors="ignore").stdout
        return nuevos if "\t_ws.col.protocol\t" in out or "_ws.col.protocol\t" in out else antiguos
    except Exception:
        return antiguos


def clase_de_pcap(carpeta_clase, nombre, videollamadas):
    if "video" in nombre.lower() and carpeta_clase in ("VoIP", "Streaming"):
        return videollamadas
    return carpeta_clase


def procesar_pcap(args):
    """Devuelve (tokens uint16 [n,20,201], longitudes uint16 [n,20], tiempos float64 [n,20], n_paquetes)."""
    ruta, tshark, campos = args
    cmd = [tshark, "-r", ruta, "-E", "separator=,", "-E", "occurrence=l", "-E", "header=n", "-E", "quote=n",
           "-T", "fields"]
    for c in campos:
        cmd += ["-e", c]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    lector = csv.reader(io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="ignore"))

    estado = {}  # clave de flujo -> [contador, t_inicio, lista_tokens, lista_long, lista_t]
    sal_tok, sal_len, sal_t = [], [], []
    n_paq = 0
    for fila in lector:
        n_paq += 1
        if len(fila) < 12:
            continue
        _, t, src, dst, proto, tsp, tdp, tlen, usp, udp_, tpay, upay = fila[:12]
        proto = proto.strip()
        if proto not in PROTOCOLOS:
            continue
        # df.drop(df[df.TCPlen <= 2].index): solo afecta a filas con tcp.len numérico
        if tlen != "":
            try:
                if float(tlen) <= 2:
                    continue
            except ValueError:
                pass
        if proto in PROTOCOLOS_UDP:
            pay = upay
        else:
            pay = tpay
        pay = pay.replace(":", "").strip()
        if pay == "":
            continue
        longitud = len(pay) // 2
        # Etiqueta del flujo: IPsrc-IPdst-Protocol-TCPsrcport-TCPdstport-UDPsrcport-UDPdstport (unidireccional)
        clave = (src, dst, proto, tsp, tdp, usp, udp_)
        t = float(t)
        # procesar_RAW: padding con '0' o truncado a 800 caracteres, tokens de 4 hex, separador FFFF
        h = pay[:LONGITUD_RAW].ljust(LONGITUD_RAW, "0")
        tok = np.frombuffer(bytes.fromhex(h), dtype=">u2").astype(np.uint16)
        tok = np.append(tok, np.uint16(TOKEN_SEPARADOR))
        e = estado.get(clave)
        if e is None or e[0] == 0 or (t - e[1]) > TIEMPO_MAXIMO_SEMIFLUJO:
            # obtenerSemiflujos: empieza un semi-flujo nuevo con este paquete (descarta el incompleto)
            estado[clave] = [1, t, [tok], [longitud], [t]]
            continue
        e[0] += 1
        e[2].append(tok)
        e[3].append(longitud)
        e[4].append(t)
        if e[0] == NUM_PAQUETES_SEMIFLUJO:
            sal_tok.append(np.stack(e[2]))
            sal_len.append(e[3])
            sal_t.append(e[4])
            estado[clave] = [0, 0.0, [], [], []]
    proc.wait()
    # Nota: como num_paquetes_semiflujo (20) == min_paquetes_flujo (20), el filtro de flujos cortos de
    # obtenerFlujos no cambia nada: un flujo con menos de 20 paquetes nunca completa un semi-flujo.
    if not sal_tok:
        return (np.zeros((0, NUM_PAQUETES_SEMIFLUJO, N_TOKENS), np.uint16),
                np.zeros((0, NUM_PAQUETES_SEMIFLUJO), np.uint16), np.zeros((0, NUM_PAQUETES_SEMIFLUJO)), n_paq)
    return (np.stack(sal_tok), np.minimum(np.array(sal_len), 65535).astype(np.uint16),
            np.array(sal_t, dtype=np.float64), n_paq)


def generar(args):
    campos = campos_tshark(args.tshark)
    print(f"Campos TShark: {campos}")
    X_cl, L_cl, T_cl, F_cl, recuentos = {}, {}, {}, {}, {}
    trabajos = []
    for carpeta in sorted(os.listdir(args.pcaps)):
        ruta_c = os.path.join(args.pcaps, carpeta)
        if not os.path.isdir(ruta_c) or carpeta not in comun.CLASES:
            continue
        for f in sorted(os.listdir(ruta_c)):
            if f.lower().endswith(EXTENSIONES):
                trabajos.append((clase_de_pcap(carpeta, f, args.videollamadas), f, os.path.join(ruta_c, f)))
    print(f"{len(trabajos)} PCAP encontrados")
    for c in comun.CLASES:
        print(f"  {c}: {sum(1 for t in trabajos if t[0] == c)} PCAP")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.procesos) as ex:
        resultados = list(ex.map(procesar_pcap, [(r, args.tshark, campos) for _, _, r in trabajos]))
    print(f"TShark + semi-flujos: {(time.time()-t0)/60:.1f} min")

    for (clase, nombre, _), (tok, lon, tie, npq) in zip(trabajos, resultados):
        print(f"  {clase:13s} {nombre:28s} paquetes={npq:9d} semiflujos={len(tok):7d}")
        X_cl.setdefault(clase, []).append(tok)
        L_cl.setdefault(clase, []).append(lon)
        T_cl.setdefault(clase, []).append(tie)
        F_cl.setdefault(clase, []).extend([nombre] * len(tok))

    # desechar_semiflujos_repetidos: np.unique por clase (ordena), en el orden de clases del artículo
    Xs, ys, Ls, Ts, Fs = [], [], [], [], []
    for k, c in enumerate(comun.CLASES):
        if c not in X_cl:
            recuentos[c] = 0
            continue
        Xc = np.concatenate(X_cl[c])
        Lc = np.concatenate(L_cl[c])
        Tc = np.concatenate(T_cl[c])
        Fc = np.array(F_cl[c])
        u, idx = np.unique(Xc, axis=0, return_index=True)
        print(f"{c}: {len(Xc)} semi-flujos, {len(Xc)-len(u)} repetidos, {len(u)} únicos")
        Xs.append(u)
        Ls.append(Lc[idx])
        Ts.append(Tc[idx])
        Fs.append(Fc[idx])
        ys.append(np.full(len(u), k, np.int8))
        recuentos[c] = len(u)
        del Xc
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    os.makedirs(args.salida, exist_ok=True)
    np.save(os.path.join(args.salida, "X.npy"), X)
    np.save(os.path.join(args.salida, "y.npy"), y)
    np.savez_compressed(os.path.join(args.salida, "meta.npz"), longitudes=np.concatenate(Ls),
                        tiempos=np.concatenate(Ts), pcap=np.concatenate(Fs))
    comun.guardar_json(os.path.join(args.salida, "recuentos.json"),
                       dict(recuentos=recuentos, articulo=RECUENTOS_ARTICULO, videollamadas=args.videollamadas,
                            campos_tshark=campos, version_tshark=version_tshark(args.tshark)))
    if args.formato_original:
        # Mismo formato que tratamiento_objetos.guardar_semiflujos, para usar los cuadernos originales.
        d = os.path.join(args.salida, "d86bd8c1f06c6599b5d3ce9b434345d8")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "array_tridimensional.obj"), "wb") as f:
            pickle.dump(X, f)
        with open(os.path.join(d, "array_etiquetas.obj"), "wb") as f:
            pickle.dump(comun.one_hot(y), f)
        with open(os.path.join(d, "diccionario_semiflujos.obj"), "wb") as f:
            pickle.dump(recuentos, f)
    return recuentos


def version_tshark(tshark):
    try:
        return subprocess.run([tshark, "-v"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "desconocida"


def reproducir(args):
    import tensorflow as tf  # noqa: F401

    sys.path.insert(0, args.repo)
    from src.modelos import build_keras_transformer_model

    X, y, _ = comun.cargar_datos(args.salida)
    idx_dev, idx_test, folds = comun.particion_articulo(y)
    comun.guardar_particion(os.path.join(args.salida, "particion.npz"), idx_dev, idx_test, folds)
    print(f"Partición: desarrollo={len(idx_dev)} test={len(idx_test)}")
    print("Test por clase:", {c: int((y[idx_test] == k).sum()) for k, c in enumerate(comun.CLASES)})

    hp = comun.HP_ARTICULO
    model = build_keras_transformer_model(X.shape[1:], len(comun.CLASES), hp["head_size"], hp["num_heads"],
                                          hp["ff_dim"], hp["mlp_units"], False, hp["dropout"], hp["mlp_dropout"],
                                          hp["num_capas_atencion"])
    with open(args.pesos, "rb") as f:
        dic = pickle.load(f)
    model.set_weights(dic["pesos"])
    print(f"Parámetros entrenables: {model.count_params()}")
    p = comun.predecir(model, X, idx_test)
    y_pred = p.argmax(1)
    m = comun.metricas_desde_cm(comun.matriz_confusion(y[idx_test], y_pred))
    m["bootstrap_IC95"] = comun.bootstrap_ic(y[idx_test], y_pred)
    m["latencia_ms_por_semiflujo"] = comun.latencia_ms_por_semiflujo(model, X, idx_test)
    m["md5_pesos"] = hashlib.md5(open(args.pesos, "rb").read()).hexdigest()
    print(comun.resumen_texto("Modelo del artículo (pesos guardados)", m))
    ok = bool(abs(m["accuracy"] - ACC_TEST_ARTICULO) < 0.0006)
    print("\nREPRODUCCIÓN:", "OK, coincide con el 93,11 % del artículo" if ok else
          f"NO coincide ({m['accuracy']*100:.2f} %). Revisar recuentos, versión de TShark y etiquetado de videollamadas.")
    m["reproduccion_ok"] = ok
    comun.guardar_json(os.path.join(args.salida, "..", "resultados", "modelo_articulo.json"), m)
    np.save(os.path.join(args.salida, "..", "resultados", "modelo_articulo_ptest.npy"), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcaps", help="Carpeta con subcarpetas Chat, Email, FileTransfer, Streaming, VoIP")
    ap.add_argument("--salida", default=os.path.join(os.path.dirname(__file__), "datos"))
    ap.add_argument("--tshark", default="tshark")
    ap.add_argument("--videollamadas", default="Streaming", choices=["Streaming", "VoIP"])
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--pesos", default=None)
    ap.add_argument("--solo-reproducir", action="store_true")
    ap.add_argument("--formato-original", action="store_true",
                    help="Guarda también los .obj del formato original (duplica ~5,5 GB)")
    args = ap.parse_args()
    if args.pesos is None:
        args.pesos = os.path.join(args.repo, "Objetos preprocesados", "Pesos_TransformerSimple",
                                  "HS_d86bd8c1f06c6599b5d3ce9b434345d8_HM_1c66f54d4dadd2ff14455a5a6b4a46fe_PesosXFMR.obj")
    os.makedirs(os.path.join(args.salida, "..", "resultados"), exist_ok=True)
    if not args.solo_reproducir:
        if not args.pcaps:
            sys.exit("Falta --pcaps")
        rec = generar(args)
        print("\nRecuentos obtenidos vs artículo:")
        iguales = True
        for c in comun.CLASES:
            ok = rec.get(c, 0) == RECUENTOS_ARTICULO[c]
            iguales &= ok
            print(f"  {c:13s} {rec.get(c, 0):7d}  artículo {RECUENTOS_ARTICULO[c]:7d}  {'OK' if ok else 'DISTINTO'}")
        if not iguales:
            print("\nAVISO: los recuentos no coinciden. Causas probables: etiquetado de videollamadas "
                  "(probar --videollamadas VoIP), PCAP distintos a los de 2023 o versión de TShark distinta "
                  "(cambia la columna Protocol). La partición ya no será la del artículo.")
    reproducir(args)


if __name__ == "__main__":
    main()
