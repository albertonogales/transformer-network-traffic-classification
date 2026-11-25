# Revisión IJIS — "From Packets to Tokens": código de los experimentos

Carpeta con todo lo necesario para responder a R2.2, R2.3 y R2.4 con el mismo protocolo que el artículo.
Se ejecuta desde la raíz del repositorio `Network-traffic-classification-CEIEC-master`.

## Orden de ejecución

| Paso | Script | Qué hace | Tiempo orientativo |
|---|---|---|---|
| 0 | `00_preprocesar_y_reproducir.py` | Regenera los semi-flujos del artículo desde los PCAP, comprueba los recuentos, reproduce el 93,11 % con los pesos guardados y guarda la partición | Varias horas (TShark sobre ~20 GB) |
| 1 | `01_exp_profundidad.py` | **R2.2**: 1, 2, 4 y 6 capas × positional encoding sí/no | ~8 configuraciones × (5 folds × 15 épocas) en la 3090 |
| 2 | `02_exp_desbalance.py` | **R2.3**: pesos de clase, focal loss, sobremuestreo, ajuste de logits | ~6 entrenamientos |
| 3 | `03_exp_ml_clasico.py` | **R2.4**: XGBoost y Random Forest sobre longitudes y tiempos de los mismos semi-flujos | CPU, < 1 h |

```bat
REM Windows, con Wireshark instalado
python revision_IJIS\00_preprocesar_y_reproducir.py --pcaps "C:\Users\Alberto Nogales\Dropbox\01_UAH\Tesis Activas\Arancha Ciprian\Papers\Enviados\43 Traffic network NLP\ISCXVPN2016_PCAPs" --tshark "C:\Program Files\Wireshark\tshark.exe"
python revision_IJIS\01_exp_profundidad.py
python revision_IJIS\02_exp_desbalance.py --capas 1
python revision_IJIS\03_exp_ml_clasico.py
```

Requisitos: `numpy`, `scikit-learn`, `tensorflow` (2.10 o posterior), `xgboost`, y TShark. Memoria: el array de
semi-flujos ocupa ~5,5 GB en `uint16`; los scripts normalizan por lotes y no necesitan los 11 GB en `float32`.

Todos los resultados quedan en `revision_IJIS/resultados/` (un JSON por configuración con métricas completas y una
tabla CSV por experimento). Los scripts se pueden interrumpir y relanzar: lo que ya existe se salta.

## Paso 0: por qué hace falta y qué comprobar

Los arrays de semi-flujos del artículo no están en el repositorio (`.gitignore`). El script los regenera con la misma
lógica que `procesarDF` / `obtenerFlujos` / `obtenerSemiflujos` / `procesar_RAW` / `desechar_semiflujos_repetidos`
(se ha verificado con datos sintéticos que produce exactamente los mismos semi-flujos que el código original).
Como la eliminación de repetidos ordena los semi-flujos (`np.unique`) y la partición usa `random_state=50`,
**si los recuentos coinciden la partición es exactamente la del artículo**.

Comprobaciones que imprime:

1. Recuentos por clase: Chat 1325, Email 168, FileTransfer 266 577, Streaming 207 405, VoIP 214 180.
2. Accuracy en test con los pesos del artículo (`HS_d86bd8c1…_HM_1c66f54d…_PesosXFMR.obj`): 93,11 %.

Si no coinciden:
- **Videollamadas.** La Tabla 1 (Streaming 30 PCAP / 7 apps; VoIP 23 / 4) solo cuadra si las 11 capturas
  `*_video*` se etiquetaron como Streaming. En la carpeta de Dropbox están en VoIP; el script las etiqueta como
  Streaming por defecto. Probar `--videollamadas VoIP` si los recuentos de Streaming/VoIP no salen.
- **Versión de TShark.** El filtro de protocolos usa la columna *Protocol* de Wireshark; versiones muy distintas a
  la de 2023 pueden etiquetar paquetes de otra forma. El script guarda la versión en `datos/recuentos.json`.

## Correcciones en `src/modelos.py`

- **Positional encoding.** La clase anterior no sumaba nada (`call` devolvía la entrada) y además copiaba la
  columna 0 en todas las dimensiones. Ahora implementa el encoding sinusoidal estándar sobre la posición del paquete
  (20 posiciones × 201 dimensiones). Con `positional_encoding=False` el modelo es idéntico al del artículo.
  Consecuencia para el artículo: el modelo publicado no usaba PE, y la opción "Yes" de la búsqueda nunca tuvo efecto.
- **Primera LayerNormalization.** Se calculaba pero no se usaba (Keras la elimina del grafo). Se mantiene ese
  comportamiento por defecto para poder cargar los pesos; `norm_previa=True` activa la normalización previa.

## Métricas del modelo del artículo (calculadas de su matriz de confusión)

Matriz guardada en `Resultados/Resultados_TransformerSimple/Matrices de confusion/HS_d86bd8c1…_HM_1c66f54d…`.
Test: 68 966 semi-flujos.

| Clase | n test | Precisión | Sensibilidad [IC 95 % Wilson] | F1 | Especificidad |
|---|---|---|---|---|---|
| Chat | 132 | 43,69 % | 73,48 % [65,4–80,3] | 54,80 % | 99,82 % |
| Email | 17 | 100,00 % | 5,88 % [1,0–27,0] | 11,11 % | 100,00 % |
| FileTransfer | 26 658 | 85,78 % | 99,74 % [99,7–99,8] | 92,24 % | 89,58 % |
| Streaming | 20 741 | 99,22 % | 78,05 % [77,5–78,6] | 87,37 % | 99,74 % |
| VoIP | 21 418 | 99,57 % | 99,63 % [99,5–99,7] | 99,60 % | 99,81 % |

Global: accuracy 93,11 %, **macro-F1 69,02 %**, **balanced accuracy 71,36 %**, macro-precisión 85,65 %,
weighted F1 92,97 %, κ 0,895, MCC 0,90. (Coinciden con todas las cifras publicadas; las macro son nuevas.)

## Cómo se configuró el entrenamiento del artículo (para la sección de métodos)

- Partición 90/10 estratificada (`train_test_split`, `random_state=50`) y `StratifiedKFold(5, shuffle, random_state=50)`.
- El modelo evaluado en test es **el del fold con mayor accuracy de validación** (entrenado con el 72 % de los datos), no uno reentrenado.
- Cross-entropy categórica, Adam (lr 3e-4), 15 épocas, batch 100, sin early stopping.
- Normalización: cada paquete se divide por su máximo, que siempre es el token FFFF (equivale a dividir entre 65 535).
- Semi-flujos repetidos eliminados dentro de cada clase antes de la partición.
- Hardware: AMD Ryzen 7 5700X, 64 GB RAM, NVIDIA RTX 3090. Ejecución del 22-06-2023.

## Experimentos en tiempo real (carpeta `Real-Time Experiments`): problemas detectados

Revisado `scripts/rt_experiment.py`, `results/results_log.csv` y los scripts de `Paper_Tables_and_Figures`.
Las cifras de las Tablas 5 y 6 se reproducen exactamente a partir de `results_log.csv`, pero:

1. **El preprocesado en tiempo real no es el del entrenamiento.** `rt_experiment.py` reimplementa el pipeline y difiere en:
   - Para cualquier protocolo distinto de "TCP" toma `udp.payload`. Los paquetes que Wireshark etiqueta como
     TLSv1.2, TLSv1.3 o SSHv2 (justamente el tráfico cifrado de aplicación) se quedan sin payload y **se descartan**;
     en el entrenamiento se conservaban con `tcp.payload`.
   - Para UDP lee `row.get("tcp.srcport", ...)`; como la columna existe (vacía), los puertos UDP valen "nan" y
     **todo el tráfico UDP entre dos IP se mezcla en un único flujo**.
   - Los flujos son bidireccionales y sin la etiqueta de protocolo; en el entrenamiento eran unidireccionales e
     incluían el protocolo.
   - Acepta QUIC (excluido en el entrenamiento) y no acepta SSL, SSLv2, TLSv1.1 ni GTP.
   - Si se superan 30 s desplaza la ventana un paquete; el original descartaba el semi-flujo y empezaba de nuevo.
   La normalización (/65535) y la reimplementación del modelo en PyTorch sí son coherentes con Keras.
2. **Selección de ejecuciones.** El artículo dice "three independent experimental runs" por clase, pero el log
   tiene más ejecuciones y la Tabla 5 usa un subconjunto (filtros en `table5_realtime_indistribution.py`):
   - Chat: se usan las ejecuciones 4–6 (28/30). Las 0–3 no se usan; la 3 (50 ventanas) da 11/50, con 37 Streaming.
   - VoIP: se usan la 5, la 6 y la parte de la 4 posterior a las 13:28 (30/30). Las 1–3 dan 0/30 (todas Streaming).
   - FileTransfer: se usan la 2, la 3 y la parte de la 1 posterior a las 12:18:30 (29/29).
   - Email: se usan la 1–3 (0/20 válidas); la 4–6 también dan 0.
   Si los cambios corresponden a cambios de protocolo (p. ej. paso de una aplicación VoIP real al RTP sintético),
   hay que explicarlo en el artículo; si no, la selección no es defendible.
3. **Ventanas sin decisión.** Son ventanas en las que ningún flujo llega a 20 paquetes (0 semi-flujos). Se excluyen
   del denominador también en la Tabla 6: Chat/WhatsApp 14 de 30 válidas, Netflix 13 de 30, BitTorrent 38 de 40.
4. **Netflix no es una aplicación OOD**: `netflix1–4.pcap` están en la clase Streaming de entrenamiento.
5. `documentation/experiment_guide.pdf` cita como mejor modelo `fe693df6…`, que en `grid_XFMR.csv` corresponde a otro
   conjunto de datos (86,5 %). El modelo del 93,11 % es `HS_d86bd8c1…_HM_1c66f54d…`.

Como se guardaron los PCAP de todas las ventanas (`results/<Clase>/run*/pcaps`), las Tablas 5 y 6 se pueden recalcular
offline con el preprocesado correcto, sin volver a capturar.
