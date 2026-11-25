# Campaña de captura en vivo en el MacBook (revisión IJIS)

Repite los experimentos en tiempo real clasificando con el modelo de la revisión
(2 bloques encoder + codificación posicional, pérdida focal γ=2) y, en la misma ventana de
captura, con el modelo original del artículo, de modo que los dos sean directamente comparables.

Nada de la carpeta `Real-Time Experiments` se modifica: los resultados nuevos van a
`/tmp/live_experiments_v2` (o a lo que indique `RT_DIR`).

## 1. Copiar esta carpeta al Mac

Copia **toda** la carpeta `tiempo_real_mac` (por ejemplo a `~/ijis_rt`). Contiene:

| Archivo | Qué es |
|---|---|
| `run_experiment_v2.sh` | captura 10 ventanas de 8 s y clasifica cada una |
| `rt_experiment_v2.py` | preprocesamiento + inferencia (solo numpy y pandas) |
| `captura_comun.py` | preprocesamiento idéntico al del entrenamiento |
| `pesos.pkl` | modelo de la revisión: focal γ=2, 2 capas, con PE |
| `pesos_articulo2023.pkl` | modelo original del artículo: 1 capa, sin PE |

## 2. Requisitos

```bash
tshark -v            # Wireshark/TShark instalado
python3 -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
```

No hace falta TensorFlow ni PyTorch. Si TShark pide permisos para capturar, concédelos
(`ChmodBPF`, igual que en la campaña anterior).

## 3. Lanzar la campaña

Para cada clase, con la aplicación correspondiente generando tráfico (igual que la vez anterior):

```bash
cd ~/ijis_rt
export RT_PESOS="$PWD/pesos.pkl:$PWD/pesos_articulo2023.pkl"
bash run_experiment_v2.sh Email 1
```

Clases y runs a repetir, para igualar la campaña original (10 ventanas cada run):

| Clase | Runs |
|---|---|
| Chat | 1, 2, 3 |
| Email | 1, 2, 3 |
| FileTransfer | 1, 2, 3 |
| Streaming | 1, 2, 3 |
| VoIP | 1, 2, 3 |

Son 15 runs × 10 ventanas × 8 s ≈ 20 min de captura, más el tiempo de cambiar de aplicación.
Si quieres más potencia estadística, añade runs 4, 5 y 6 por clase.

Cada ventana escribe **cuatro** filas en `/tmp/live_experiments_v2/results_log.csv`: los dos
modelos × los dos preprocesamientos (`preproc` = `articulo`, el fiel al entrenamiento, y `rt`, el
de la campaña de 2026, con flujos bidireccionales). Cada fila lleva `modelo`, `capas`, `pe`,
`preproc`, el reparto de predicciones por clase, la confianza media y los tiempos de
preprocesamiento e inferencia.

Así no hay que decidir antes de capturar qué preprocesamiento se adopta en el artículo: la campaña
mide los dos sobre el mismo tráfico y la decisión se toma con los datos. Con `RT_PREPROC=articulo`
o `RT_PREPROC=rt` se mide solo uno, si se prefiere.

## 4. Traer los resultados

Copia de vuelta la carpeta completa `/tmp/live_experiments_v2` (incluye los `.pcap` y los `.csv`,
para que la campaña se pueda reanalizar sin volver a capturar).

## 5. Protocolo, declarado de antemano

Esto no es burocracia: la campaña de 2026 se analizó con filtros por clase escritos **después** de
ver los resultados (`Table_5_RealTime_InDistribution/table5_realtime_indistribution.py`), que
excluyen por marca de tiempo justo las ventanas que fallaban. Con la campaña nueva eso no puede
volver a pasar, así que se fija antes de capturar:

1. **Todas las ventanas de todos los runs entran en la tabla.** Ninguna exclusión posterior.
2. **Un intento = un run nuevo.** Si un run sale mal (la aplicación no estaba generando tráfico, el
   simulador no arrancó), no se repite con el mismo número: se usa el siguiente libre y se anota en
   el diario de capturas por qué el anterior no sirve. El script ya se niega a sobrescribir un run.
3. **Si hace falta descartar ventanas, el criterio se define ahora y se aplica igual a todas las
   clases y runs.** Criterio propuesto: descartar la ventana si `n_semiflows == 0` (no hubo
   semi-flujos que clasificar). Nada más.
4. **Diario de capturas.** Para cada run: clase, número de run, aplicación concreta y versión, hora
   de inicio, y qué se estuvo haciendo durante los 80 s. Sin esto no se puede justificar después
   qué tráfico contenía cada ventana.
5. **Las dos campañas, in-distribution y OOD, con el mismo protocolo.** Para la OOD, anotar la
   aplicación usada por clase y añadir una para Email, que en la campaña anterior se quedó sin
   fila OOD. Aplicaciones OOD para esta campaña (revisadas 2026-11-14):

   | Clase | Aplicación OOD | Nota |
   |---|---|---|
   | Chat | WhatsApp Web | Genuinamente OOD (no está en ISCXVPN2016) |
   | Email | (definir, p. ej. ProtonMail o Tutanota) | No hay fila OOD en la campaña anterior |
   | FileTransfer | BitTorrent | Genuinamente OOD (no está en ISCXVPN2016) |
   | Streaming | **Amazon Prime Video** | Netflix NO sirve: netflix1–4.pcap están en ISCXVPN2016 Streaming (datos de entrenamiento) |
   | VoIP | Google Meet | Genuinamente OOD (no está en ISCXVPN2016) |

## 6. Notas para el artículo

- La latencia y el throughput de `results_log.csv` son los de **este** Mac con la inferencia en
  NumPy; no son comparables con los de la Tabla 5 original (RTX 3090 y PyTorch según
  `best_model_weights.obj`). Hay que indicar la máquina en la tabla nueva.
- El modelo de la revisión tiene dos bloques encoder, así que su coste de inferencia es
  aproximadamente el doble que el del artículo. El `results_log.csv` permite comprobarlo
  directamente, porque mide los dos modelos sobre las mismas ventanas.
- Las capturas de 8 s rara vez permiten completar semi-flujos de 20 paquetes en flujos lentos
  (correo, mensajería). El número de semi-flujos por ventana (`n_semiflows`) debe aparecer en la
  tabla: es parte del resultado, no un detalle de implementación.
