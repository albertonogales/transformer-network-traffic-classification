# Informe de la revisión para IJIS

**Artículo:** *From Packets to Tokens: A Transformer Framework for Real-Time Network Traffic Classification*
**Fecha:** 14 de noviembre de 2026
**Alcance:** experimentos pedidos por el Revisor 2 (R2.2 profundidad, R2.3 desbalance, R2.4 referencia
de ML clásico) más la comprobación del preprocesamiento en tiempo real que de ellos se derivó (R2.5).
No se ha modificado el manuscrito ni nada de la carpeta `Real-Time Experiments`.

---

## 1. Entorno

| | |
|---|---|
| Máquina | portátil Windows 11 Enterprise 10.0.26300, solo CPU (sin GPU) |
| Python | 3.12.10, entorno virtual en `%LOCALAPPDATA%\venvs\ijis` |
| TensorFlow / Keras | 2.21.0 / 3.15.1 |
| NumPy / pandas / SciPy | 2.5.4 / 2.3.3 / 1.18.1 |
| scikit-learn / XGBoost | 1.9.1 / 3.4.1 |
| TShark | 4.0.6 (v4.0.6-0-gac2f5a01286a), en `%LOCALAPPDATA%\ws406` |
| Máquina del artículo (2023) | AMD Ryzen 7 5700X + NVIDIA RTX 3090, según `best_model_weights.obj` |

Las latencias de este informe son de CPU en un portátil y **no** son comparables con las del
artículo, medidas en la máquina con RTX 3090.

## 2. Paso 0: datos y reproducción de la línea base

Los PCAP de partida son **ISCXVPN2016** (`ISCXVPN2016_PCAPs`). Los recuentos de semi-flujos
regenerados con TShark 4.0.6 (`revision_IJIS/datos/recuentos.json`) son:

| Clase | Semi-flujos (TShark 4.0.6) | Recuento del artículo | PCAP distintos |
|---|---|---|---|
| Chat | 1 325 | 1 325 | 13 |
| Email | 168 | 168 | **4** |
| FileTransfer | 268 068 | 266 577 | 30 |
| Streaming | 207 413 | 207 405 | 30 |
| VoIP | 214 181 | 214 180 | 23 |
| **Total** | **691 155** | **689 655** | |

La única discrepancia relevante es FileTransfer (+1 491 semi-flujos), trazada al PCAP
`ftps_down_1a`. No se pudo reproducir exactamente el recuento de 2023 y, por decisión del autor,
se toma el conjunto regenerado con TShark 4.0.6 como nueva línea base; los datos con TShark 4.6.9
quedan en `datos_tshark469`.

Los pesos guardados del artículo, evaluados sobre la partición nueva
(`resultados/modelo_articulo.json`), dan **93.27 %** de accuracy (IC95 % 93.08–93.45 %) frente al
93.11 % publicado, con **macro-F1 67.8 %**, **balanced accuracy 71.4 %** y **recall de Email 0 %**
(0 de 17). No se tocó el preprocesamiento para forzar la coincidencia.

**Protocolo (sin cambios).** `train_test_split` 90/10 estratificado con `random_state=50`;
`StratifiedKFold(5, shuffle=True, random_state=50)`; normalización de cada paquete por su máximo;
se elige el mejor fold por accuracy de validación y se evalúa en test sin reentrenar; Adam 3e-4,
15 épocas, batch 100, 8 cabezas × 6, FF 256, MLP 250, dropout 0.2.

**Limitación que debe declararse en el manuscrito:** la partición es a nivel de semi-flujo, no de
captura. Los semi-flujos de un mismo PCAP caen en train y en test a la vez. Es especialmente grave
en Email: toda la clase son 168 semi-flujos de **4** ficheros, y los 17 de test salen de los mismos
ficheros que los de desarrollo (`email1b.pcap`: 99 en desarrollo y 13 en test). El recall de Email
en test mide, por tanto, memorización de esas capturas (véase la sección 6).

## 3. R2.4 — Referencia de ML clásico sobre estadísticas de flujo

Modelos entrenados sobre estadísticas de longitud y tiempo de los 20 paquetes del semi-flujo
(sin payload), con la misma partición.

| Modelo | Accuracy | Macro-F1 | Bal. acc. | κ | MCC | Recall Email | Recall Chat | Latencia |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 99.39 % | 97.34 % | 96.64 % | 0.991 | 0.991 | 94.1 % | 91.0 % | 0.069 ms |
| XGBoost (balanceado) | **99.41 %** | 97.33 % | 97.24 % | 0.991 | 0.991 | 94.1 % | 94.0 % | 0.006 ms |
| Random Forest | 99.36 % | **97.40 %** | 96.76 % | 0.990 | 0.990 | 94.1 % | 91.7 % | 0.005 ms |
| Random Forest (balanceado) | 99.35 % | 96.10 % | 96.91 % | 0.990 | 0.990 | 94.1 % | 92.5 % | 0.005 ms |
| Transformer (artículo) | 93.27 % | 67.76 % | 71.36 % | 0.898 | 0.903 | **0 %** | 78.9 % | 0.100 ms |

**Conclusión.** Un XGBoost sobre estadísticas de longitud y tiempo supera al Transformer por
6 puntos de accuracy y 30 de macro-F1, con 16 veces menos latencia. Es el resultado más incómodo
de la revisión y hay que presentarlo tal cual: la aportación del artículo no puede sostenerse como
«mejor exactitud», sino, en el mejor de los casos, como «clasificación a partir de payload sin
ingeniería de características». Conviene además revisar si las estadísticas de longitud y tiempo
están tan informadas por la partición a nivel de semi-flujo como el propio payload.

## 4. R2.2 — Profundidad del encoder (1, 2, 4 y 6 bloques)

Ocho configuraciones (1/2/4/6 bloques × con y sin codificación posicional), semilla 0, resto del
protocolo intacto. Tiempos reales de entrenamiento en CPU, de 3.3 h a 14.3 h por configuración;
unas 75 h en total.

| Capas | PE | Parámetros | Val. acc. (media ± sd) | Test acc. | Macro-F1 | Bal. acc. | Recall Email | Recall Chat | Latencia | Horas |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | no | 149 213 | 92.42 ± 0.77 % | 92.94 % | 69.09 % | 72.03 % | 0 % | 83.5 % | 0.114 ms | 3.3 |
| 1 | sí | 149 213 | 93.36 ± 0.13 % | **93.54 %** | 89.04 % | 86.06 % | 82.4 % | 69.2 % | 0.095 ms | 3.6 |
| 2 | no | 291 921 | 93.21 ± 0.36 % | 93.46 % | 86.17 % | 85.29 % | 70.6 % | 77.4 % | 0.187 ms | 6.6 |
| 2 | sí | 291 921 | **93.39 ± 0.12 %** | 93.50 % | **90.06 %** | **88.40 %** | 88.2 % | 75.2 % | 0.172 ms | 5.2 |
| 4 | no | 577 337 | 73.19 ± 8.77 % | 87.59 % | 52.24 % | 52.04 % | 0 % | 0 % | 0.321 ms | 8.4 |
| 4 | sí | 577 337 | 33.57 ± 3.92 % | 41.26 % | 19.96 % | 26.85 % | 0 % | 0 % | 0.287 ms | 8.5 |
| 6 | no | 862 753 | 51.16 ± 14.85 % | 67.00 % | 32.24 % | 38.66 % | 0 % | 0 % | 0.427 ms | 14.1 |
| 6 | sí | 862 753 | 39.59 ± 13.64 % | 66.35 % | 32.44 % | 38.14 % | 0 % | 0 % | 0.460 ms | 14.3 |

**Conclusiones.**

1. Más profundidad no ayuda: 1 y 2 bloques son equivalentes en accuracy (93.5 %), y 4 y 6 bloques
   **no entrenan** con los hiperparámetros del artículo (accuracy de validación oscilando entre el
   3 % y el 87 % entre folds, y clases minoritarias a 0).
2. La causa más probable del colapso es el bloque encoder del artículo, que normaliza solo
   **después** de la atención (`src/modelos.py:395`), de modo que los residuos sin normalizar se
   acumulan al apilar bloques. No se ha cambiado (`norm_previa=True` existe pero alteraría la
   arquitectura publicada y el protocolo prohibía tocarla). Es un hallazgo legítimo de R2.2, no un
   fallo de ejecución: la arquitectura publicada no escala en profundidad.
3. La codificación posicional sí aporta, y sobre todo en las clases minoritarias y en la
   estabilidad entre folds (sd de 0.77 % a 0.13 % con 1 bloque): el macro-F1 pasa de 69.1 % a
   89.0 %. Esto merece una frase en el manuscrito, porque el modelo publicado **no** la usa.
4. Configuración elegida para R2.3: **2 bloques con codificación posicional**, por ser la de mejor
   macro-F1 y balanced accuracy.

## 5. R2.3 — Mitigación del desbalance (2 bloques, con PE)

Las tres variantes que pide el revisor, con todo lo demás igual: `base` (entropía cruzada sin
pesos, referencia), `pesos_inv` (entropía cruzada con pesos inversos a la frecuencia) y
`focal_g2_sin_alfa` (pérdida focal pura, γ=2, α=1, es decir solo escalado por confianza).

| Método | Accuracy | Macro-F1 | Bal. acc. | κ | MCC | Recall Email | Precisión Email | Recall Chat | Precisión Chat |
|---|---|---|---|---|---|---|---|---|---|
| Entropía cruzada (base) | **93.50 %** | 90.06 % | **88.40 %** | 0.901 | 0.906 | 88.2 % | 83.3 % | 75.2 % | **96.2 %** |
| Pesos inversos | 88.35 % | 61.26 % | 87.33 % | 0.824 | 0.830 | **94.1 %** | **4.3 %** | 79.7 % | 20.5 % |
| Focal γ=2 (α=1) | 93.46 % | **90.57 %** | 87.96 % | 0.901 | 0.906 | 76.5 % | **92.9 %** | **85.0 %** | 93.4 % |

**Conclusiones.**

1. **La pérdida focal es la viable de las dos que sugiere el revisor.** Mejor macro-F1 (90.57 %) a
   igual accuracy, y la mejora real está en Chat: F1 de 84.4 % a 89.0 %, con el recall subiendo de
   75.2 % a 85.0 %.
2. **El pesado por clase hay que reportarlo como resultado negativo.** Sube el recall de Email al
   94.1 %, pero con precisión del 4.3 %: etiqueta unos 370 semi-flujos como Email para acertar 16
   de 17. El macro-F1 se derrumba al 61.3 % y la accuracy pierde 5 puntos. No es utilizable en
   producción, que es justo el criterio que plantea el revisor.
3. **Las conclusiones sobre Email son solo indicativas.** El test tiene 17 semi-flujos de Email: la
   caída de recall del 88.2 % al 76.5 % con pérdida focal son 15 aciertos frente a 13, y los
   intervalos de confianza se solapan ampliamente. No se puede concluir nada firme sobre Email con
   este tamaño de muestra, y así debe decirse.
4. Se guardaron los pesos del mejor fold de cada método
   (`<metodo>_capas2_PE-si_pesos.pkl`) para poder reclasificar las capturas en vivo sin reentrenar.
   Verificado: reconstruyen el modelo exacto (291 921 parámetros, accuracy de test 93.456 %,
   diferencia máxima de probabilidad 4.8e-07 frente a las guardadas).

## 6. R2.5 — El 0 % de Email en tiempo real

El comentario del revisor sobre Email se refiere a la **accuracy en tiempo real**, así que se
revisó la cadena de inferencia en vivo. Todo lo de esta sección se hizo **leyendo**
`Real-Time Experiments`; los scripts nuevos están en `revision_IJIS`.

### 6.1 Tres discrepancias reales entre el preprocesamiento en vivo y el de entrenamiento

Comparando `Real-Time Experiments/scripts/rt_experiment.py` con
`src/procesamiento_pcap.py` + `src/procesamiento_flujos.py`:

1. **Clave de flujo.** Entrenamiento (`procesamiento_pcap.py:76`):
   `IPsrc-IPdst-Protocol-TCPsrcport-TCPdstport-UDPsrcport-UDPdstport`, es decir **unidireccional y
   con protocolo**. En vivo: `tuple(sorted([(src,sp),(dst,dp)]))`, **bidireccional y sin
   protocolo**, que mezcla los dos sentidos de la conversación en un mismo semi-flujo. Medido sobre
   las capturas guardadas, la proporción de semi-flujos que mezclan sentidos es del **72.5 % en
   Chat**, **50.0 % en FileTransfer**, 7.8 % en Email, 1.8 % en VoIP y 1.2 % en Streaming. El
   modelo nunca vio en entrenamiento entradas de ese tipo.
2. **Puertos UDP perdidos.** `row.get("tcp.srcport", row.get("udp.srcport",""))` no hace el
   *fallback* que pretende: en un paquete UDP la columna `tcp.srcport` existe y vale NaN, así que
   pandas devuelve NaN en vez del valor por defecto. Comprobado en `VoIP/run1/pcap_1.csv`: la clave
   recibe `nan` cuando el puerto UDP real es 60597. Todas las conversaciones UDP entre dos IP, QUIC
   incluido, se agrupan como un único flujo.
3. **Segmentación en semi-flujos.** Al superarse los 30 s, el entrenamiento **descarta** el
   semi-flujo parcial y reinicia en ese paquete (`obtenerSemiflujos`), mientras que la versión en
   vivo desliza la ventana un paquete. Efecto pequeño: de 1 545 a 1 426 semi-flujos en Email y de
   58 766 a 58 229 en FileTransfer.

**La normalización no es una discrepancia.** Dividir cada paquete por su máximo equivale a dividir
entre 65535, porque todo paquete lleva el token separador FFFF. Se había apuntado como posible
problema y no lo es.

### 6.2 Reclasificación de las capturas guardadas

`revision_IJIS/03_clasificar_capturas.py` reclasifica las 354 ventanas guardadas
(`Real-Time Experiments/results`) con cualquier modelo de la revisión y con uno u otro
preprocesamiento, para separar el efecto del modelo del efecto del preprocesamiento. La inferencia
está reimplementada en NumPy (`revision_IJIS/captura_comun.py`) y validada contra Keras: diferencia
máxima de probabilidad 1.2e-06 y 100 % de coincidencia de clase en los tres modelos de R2.3.

La réplica del preprocesamiento en vivo es fiel: para Streaming, cuyas ventanas guardadas coinciden
con las del log del artículo, reproduce exactamente sus 2 331 semi-flujos y su 32.7 % de accuracy.
En Chat, Email y FileTransfer el log del artículo cubre más ventanas de las que quedaron guardadas
(Chat 140 frente a 84), así que esas cifras no son comparables una a una.

*(Resultados de las cuatro combinaciones: véase la tabla de la sección 6.3.)*

### 6.3 Resultados

Accuracy a nivel de semi-flujo sobre las capturas guardadas (la etiqueta verdadera es la clase de
la campaña de captura):

| Modelo / preprocesamiento | Chat | Email | FileTransfer | Streaming | VoIP | Media por clase | Total |
|---|---|---|---|---|---|---|---|
| Artículo 2023 (1 capa, sin PE) + preproc. en vivo | **58.8 %** | **0 %** | 46.8 % | 32.7 % | **31.0 %** | 33.9 % | 45.3 % |
| Artículo 2023 (1 capa, sin PE) + preproc. del artículo | 37.5 % | **0 %** | **71.5 %** | **37.7 %** | 27.0 % | **34.7 %** | **65.5 %** |
| Revisión (focal, 2 capas, PE) + preproc. en vivo | 55.5 % | **0 %** | 50.8 % | 17.8 % | **31.5 %** | 31.1 % | 48.2 % |
| Revisión (focal, 2 capas, PE) + preproc. del artículo | 37.1 % | **0 %** | 61.6 % | 19.8 % | 27.7 % | 29.2 % | 56.3 % |

Latencia de inferencia en NumPy sobre este portátil (mediana por semi-flujo): 0.57–0.61 ms con
1 bloque encoder y 1.42–1.51 ms con 2 bloques, es decir **×2.4**, coherente con el doble de bloques.

El «Total» está dominado por FileTransfer (61 404 de 71 481 semi-flujos), así que la media por
clase es la cifra que hay que leer. Conviene separar los dos efectos:

- **Corregir el preprocesamiento sí mejora**, con el modelo del artículo: la media por clase pasa
  de 33.9 % a 34.7 % y el total de 45.3 % a 65.5 % (FileTransfer de 46.8 % a 71.5 %). Confirma que
  las tres discrepancias de la sección 6.1 degradaban las Tablas 5 y 6 publicadas.
- **Cambiar al modelo «mejor» no mejora el tráfico real**: con el mismo preprocesamiento corregido,
  el modelo de la revisión (focal, 2 capas, PE) baja la media por clase de 34.7 % a 29.2 %, pese a
  ser el mejor en macro-F1 sobre el test (90.57 % frente a 67.76 %). Hay que reportarlo tal cual:
  es una señal más de que la métrica de test, con partición a nivel de semi-flujo, no predice el
  comportamiento con tráfico ajeno al conjunto.

**El 0 % de Email no lo explica el preprocesamiento.** Con el mejor modelo de la revisión y el
preprocesamiento corregido, Email sigue en **0 de 1 418** semi-flujos, con confianza media 0.61;
el reparto de predicciones es 918 FileTransfer, 289 Streaming, 139 Chat y 72 VoIP.

Hipótesis descartadas por medición:

- **Desbalance de clases.** El mismo modelo alcanza 76.5 % de recall en Email en el test.
- **Discrepancia de preprocesamiento.** Corregida, y Email sigue a 0 %.
- **Protocolo de correo distinto.** Las capturas en vivo son IMAPS (993) con algo de SMTPS (465);
  los PCAP de entrenamiento `email1b` y `email2b` también son mayoritariamente 993, y `email1a` y
  `email2a`, 465 y 995. No hay un salto de protocolo.
- **Versión de TLS.** Entrenamiento 2016: TLS 1.2 mayoritario con algo de TLS 1.0. En vivo:
  TLS 1.2 mayoritario con algo de TLS 1.3. Tampoco hay un salto grande.

**Lo que queda, y encaja con todo lo medido:** la clase Email son 168 semi-flujos de **4** ficheros
PCAP, y los 17 de test salen de los mismos ficheros que los de desarrollo. El recall de Email en
test (0 % con los pesos de 2023, 88.2 % con el modelo nuevo) mide memorización de `email1b.pcap`,
no capacidad de reconocer correo. Ante tráfico de correo real y ajeno al conjunto, el modelo acierta
cero. Ninguna función de pérdida arregla eso: es una limitación del conjunto de datos y de la
partición.

### 6.4 Cómo seleccionan las ventanas las Tablas 5 y 6 (hallazgo grave)

Las Tablas 5 y 6 salen **del mismo** `results_log.csv` (md5 idéntico en las dos carpetas),
repartido por número de run: la Tabla 5 usa unos runs por clase y la Tabla 6 los runs 7, 8 y 9.
El reparto de la Tabla 5 está escrito a mano clase por clase en
`table5_realtime_indistribution.py`, e incluye cortes por marca de tiempo **dentro** de un run:

```python
'VoIP':         lambda d: d[(d['class']=='VoIP') & ((d['run'].isin([5,6])) | ((d['run']==4) & (d['timestamp']>='2026-06-29T13:28:00')))],
'FileTransfer': lambda d: d[(d['class']=='FileTransfer') & ((d['run'].isin([2,3])) | ((d['run']==1) & (d['timestamp']>='2026-06-29T12:18:30')))],
```

Lo que esos filtros dejan fuera, medido sobre el log completo:

| Clase | Runs en la Tabla 5 | Accuracy de los incluidos | Runs excluidos | Ventanas excluidas | Accuracy de los excluidos | Semi-flujos/ventana excluidos |
|---|---|---|---|---|---|---|
| Chat | 4, 5, 6 | 90–100 % | 3 (y 0, 1, 2) | 50 | **22.0 %** | 40 (frente a 23–47 incluidos) |
| VoIP | 5, 6 y la cola del 4 | 100 % | 1, 2, 3 | 30 | **0 %** | 41–45 (frente a 31) |
| FileTransfer | 2, 3 y la cola del 1 | 100 % | los 17 primeros del run 1 | 17 | **13.3 %** | 714 |
| Streaming | 1, 2, 3 | 90–100 % | — | — | — | — |
| Email | 1, 2, 3 | 0 % | 4, 5, 6 | 40 | 0 % | 16–18 |

Los dos cortes por marca de tiempo separan exactamente las ventanas buenas de las malas: en
FileTransfer run 1 las 10 que pasan aciertan el **100 %** y las 30 excluidas el **13.3 %**; en
VoIP run 4, las que pasan el **100 %** y las excluidas el **0 %**.

El caso de VoIP es el más difícil de defender: los runs 1, 2 y 3 aciertan el 0 % con **más**
semi-flujos por ventana (41–45) que los runs 5 y 6, que aciertan el 100 % con 31. No se puede
atribuir a «no había tráfico todavía». Puede haber una explicación legítima —por ejemplo, que
`rtp_simulate.py` no estuviera en marcha en los runs 1–3, o que la aplicación objetivo arrancara
tarde—, pero **no está documentada en ninguna parte**, y tal como está el 100 % de VoIP en tiempo
real se sostiene sobre 20 ventanas elegidas después de ver los resultados. Arancha tiene que
explicar qué pasó en cada run excluido; si la explicación existe, el criterio debe declararse y
aplicarse por igual a todas las clases y runs (por ejemplo, «se descartan las ventanas sin
semi-flujos»), no como un corte horario por clase.

**Además, el archivo no permite reproducir las tablas.** Varios runs se repitieron escribiendo
sobre las mismas rutas `run<N>/csvs/pcap_<i>.csv`: el log guarda 50 ventanas de Chat run 3, 40 de
FileTransfer run 1 y 40 de VoIP run 4, pero en disco solo hay 10 de cada, las del último intento.
Las cifras publicadas de la Tabla 5 no se pueden regenerar desde las capturas guardadas, porque los
intentos que los filtros seleccionan ya se sobrescribieron.

Esto es, con diferencia, el problema más serio encontrado en la revisión: es más grave que el
desbalance de Email y más grave que las tres discrepancias de preprocesamiento. Si un revisor
ejecuta `table5_realtime_indistribution.py` y mira los filtros, lo ve.

**Consecuencia:** las Tablas 5 y 6 deben rehacerse con una campaña nueva y un protocolo declarado
de antemano (sección 8), no recalcularse desde el archivo. La reclasificación de la sección 6.3
sigue siendo válida como comparación entre modelos sobre el mismo tráfico, pero **no** es una
reproducción de las Tablas 5 y 6: usa las 354 ventanas que quedaron en disco, que no son el
conjunto que esas tablas emplean.

### 6.5 La etiqueta de protocolo de Wireshark forma parte de la clave de flujo (hallazgo grave)

`nt2txt` es el nombre que el artículo da al proceso de tokenización (Figura 3); no existe ningún
módulo con ese nombre. La implementación son `src/procesamiento_pcap.py` +
`src/procesamiento_flujos.py` y, para los datos de esta revisión,
`revision_IJIS/00_preprocesar_y_reproducir.py`, que es la versión en streaming que generó `X.npy`.
`revision_IJIS/captura_comun.py` reproduce esa cadena sobre los CSV de las capturas; comprobado
ventana a ventana contra `00_preprocesar_y_reproducir.procesar_pcap` leyendo el PCAP de la misma
ventana: arrays **idénticos** en 4 de 5 ventanas comparables.

La ventana que no coincide (Email run 1, ventana 2: 9 semi-flujos frente a 8) descubre un problema
de la propia cadena de entrenamiento. La clave de flujo incluye la **columna Protocol de
Wireshark**, que es una etiqueta de presentación, no parte de la 5-tupla:

```python
df["Etiqueta"] = IPsrc +"-"+ IPdst +"-"+ Protocol +"-"+ TCPsrcport +"-"+ ... # procesamiento_pcap.py:72
```

Wireshark etiqueta los paquetes de una misma conexión TCP de varias maneras (TCP cuando no ve un
registro TLS completo, TLSv1.2/TLSv1.3 cuando sí, SSL en el saludo inicial). Medido en esa ventana,
sobre las conexiones con 20 o más paquetes con payload: **6 de 9 reciben más de una etiqueta**. Un
ejemplo real, una sola conexión `10.0.16.238:59740 -> 52.97.159.210:443` de 41 paquetes:

| Etiqueta de Wireshark | Paquetes |
|---|---|
| TLSv1.2 | 26 |
| TCP | 14 |
| SSL | 1 |

Esa conexión no es un flujo para la cadena: son **tres** flujos de 26, 14 y 1 paquetes. Solo el de
26 llega a los 20 necesarios; los otros 15 paquetes se descartan sin dejar rastro.

Consecuencias:

1. **Los «flujos» del artículo no son conexiones**, sino fragmentos de conexión agrupados por una
   cadena de presentación. Eso no es lo que describe el manuscrito.
2. **Explica la irreproducibilidad entre versiones de TShark.** Releyendo el mismo PCAP, el TShark
   del Mac y el 4.0.6 reparten las etiquetas de forma distinta (en esa ventana: TCP 365 frente a
   400, TLSv1.2 203 frente a 184, TLSv1.3 131 frente a 117, SSL 5 frente a 2). Al entrar la
   etiqueta en la clave, cambian los flujos y cambia el recuento de semi-flujos. Es la causa más
   probable de los +1 491 semi-flujos de FileTransfer entre 4.0.6 y 4.6.9 (sección 2), que hasta
   ahora quedó sin explicar.
3. **Se pierden datos en silencio**, sobre todo en clases con pocos paquetes por conexión, que es
   justo el caso de Email.
4. Por casualidad, la clave bidireccional de `rt_experiment.py` **no** incluye el protocolo, así
   que evita este problema a cambio de introducir el de mezclar sentidos (sección 6.1).

`captura_comun.py` mantiene la etiqueta en la clave, a propósito, para ser fiel a lo que se entrenó.
Quitarla sería un cambio de protocolo y de definición de semi-flujo —habría que regenerar el
conjunto y reentrenar—, así que es una decisión del autor, no una corrección que se pueda colar en
la revisión. Pero el manuscrito no puede seguir describiendo los flujos por la 5-tupla cuando el
código los define por la 5-tupla más una etiqueta de Wireshark.

## 7. Qué debe cambiar en el manuscrito

Propuestas, para discutir; **no se ha editado el manuscrito**.

1. **Declarar la partición a nivel de semi-flujo** como limitación, con los números de la sección 2
   (Email: 168 semi-flujos de 4 PCAP; `email1b.pcap` en train y en test).
2. **Reescribir las conclusiones sobre Email.** La métrica de Email en test no sostiene ninguna
   afirmación; el 0 % en tiempo real es coherente con una clase de 4 capturas.
3. **Añadir R2.2 como estudio de profundidad**, incluido el hallazgo de que la arquitectura
   publicada no entrena con 4 ni 6 bloques, y que la codificación posicional mejora el macro-F1 en
   20 puntos con 1 bloque.
4. **Añadir R2.3** con la pérdida focal como mitigación recomendada y el pesado por clase como
   resultado negativo documentado.
5. **Añadir R2.4** y matizar la aportación del artículo: un XGBoost sobre estadísticas de longitud
   y tiempo es mejor y más rápido.
6. **Rehacer las Tablas 5 y 6 con una campaña nueva.** No basta con corregir el preprocesamiento:
   los filtros por clase y los cortes por marca de tiempo de la sección 6.4 hacen que las cifras
   publicadas no sean defendibles ni reproducibles desde el archivo. Campaña nueva, protocolo
   declarado antes de capturar, todas las ventanas en la tabla.
7. **Añadir una fila OOD para Email**, que no la tiene: en la campaña de 2026 no hay runs 7–9 de
   Email, así que la Tabla 6 solo cubre cuatro clases.
8. **Indicar la máquina en toda medida de latencia.** Las del artículo son de un Ryzen 7 5700X con
   RTX 3090; las de este informe, de un portátil sin GPU.

## 8. Campaña nueva en el MacBook

`revision_IJIS/tiempo_real_mac` contiene lo necesario para repetir la campaña en vivo clasificando
**cada ventana con los dos modelos a la vez** (el del artículo y el de la revisión), de modo que la
comparación no quede confundida con el tráfico: `run_experiment_v2.sh`, `rt_experiment_v2.py`,
`captura_comun.py`, los dos ficheros de pesos y `LEEME.md` con los comandos exactos. Solo necesita
`tshark`, `numpy` y `pandas`. La salida va a `/tmp/live_experiments_v2`;
`Real-Time Experiments` no se toca.

## 9. Ficheros generados

| Ruta | Contenido |
|---|---|
| `resultados/exp_R2.2_profundidad/` | 8 JSON por configuración + `tabla_R2.2.csv` |
| `resultados/exp_R2.3_desbalance/` | 3 JSON, probabilidades, `tabla_R2.3.csv` y los 3 `*_pesos.pkl` |
| `resultados/exp_R2.4_ml_clasico/` | `tabla_R2.4.csv` |
| `resultados/exp_R2.5_tiempo_real/` | `*_ventanas.csv` y `*_resumen.json` por combinación |
| `captura_comun.py` | preprocesamiento fiel al entrenamiento + inferencia en NumPy |
| `03_clasificar_capturas.py` | reclasificación de las capturas guardadas |
| `tiempo_real_mac/` | campaña nueva para macOS |
