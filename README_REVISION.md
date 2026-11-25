# Repository — IJIS Revision Notes
# From Packets to Tokens: A Transformer Framework for Real-Time Network Traffic Classification

**Last updated:** 23 November 2026  
**Revision deadline:** 23 November 2026  
**Submission ID:** a7ad189f-64cf-4240-8b6a-00022ee3596b

---

## Final model (manuscript Tables 3–6)

| Property | Value |
|---|---|
| Architecture | 2 Transformer encoder layers + sinusoidal positional encoding |
| Loss function | Cross-entropy (CE baseline — see R2.3 below) |
| Total parameters | **291,921** |
| Training hardware | Apple MacBook Air, Apple M4, 10-core CPU, 16 GB unified memory, macOS 15.6.1 |
| Training time | ~311 minutes |
| Test accuracy | **93.50%** (IC95 93.31–93.68%) |
| Macro-averaged F1 | **90.06%** |
| Balanced accuracy | **88.4%** |
| Cohen's κ | **0.901** |
| MCC | **0.906** |
| Inference latency | 0.172 ms/semi-flow (CPU) |

Weights: `revision_IJIS/resultados/exp_R2.3_desbalance/base_capas2_PE-si_pesos.pkl`  
Also copied to `~/ijis_rt/pesos.pkl` for the live-capture campaign.

---

## Repository structure

```
Network-traffic-classification-CEIEC-master/
├── src/                          Core library (preprocessing, model, metrics, real-time)
├── Transformer entrenamiento.ipynb  Original training notebook (1-layer, no PE)
├── LSTM entrenamiento.ipynb         LSTM baseline notebook
├── Captura en tiempo real.ipynb     Original live-capture notebook
├── Inferencia.ipynb                 Offline inference notebook
├── Preprocesamiento.ipynb           Preprocessing notebook
└── revision_IJIS/                   ALL revision experiments (R2.2–R2.5)
    ├── 00_preprocesar_y_reproducir.py   Regenerate X.npy/y.npy from PCAPs
    ├── 01_exp_profundidad.py            R2.2: depth search (1/2/4/6 layers × PE yes/no)
    ├── 02_exp_desbalance.py             R2.3: imbalance mitigation (CE / focal / inv-weights)
    ├── 03_exp_ml_clasico.py             R2.4: XGBoost and Random Forest baselines
    ├── 03_clasificar_capturas.py        R2.5: reclassify saved captures with any model
    ├── comun.py                         Shared protocol (partition, normalisation, metrics)
    ├── captura_comun.py                 Live-capture preprocessing (faithful to training)
    ├── datos/                           Preprocessed data (X.npy, y.npy, recuentos.json)
    ├── resultados/                      All result JSON/CSV files
    │   ├── modelo_articulo.json             Original 1-layer model re-evaluated on new partition
    │   ├── exp_R2.2_profundidad/            Depth search: 8 configs × JSON + tabla_R2.2.csv
    │   ├── exp_R2.3_desbalance/             CE / focal / inv-weights + tabla_R2.3.csv
    │   ├── exp_R2.4_ml_clasico/             XGBoost, RF results + tabla_R2.4.csv
    │   └── exp_R2.5_tiempo_real/            Reclassification of 2026 captures
    └── tiempo_real_mac/                 Scripts for new live-capture campaign on macOS
        ├── run_experiment_v2.sh
        ├── rt_experiment_v2.py
        ├── captura_comun.py
        ├── pesos.pkl                    → CE baseline (2L+PE, 291,921 params)  ← FINAL MODEL
        ├── pesos_articulo2023.pkl       → Original 1-layer, no PE
        ├── LEEME.md                     Protocol and instructions
        └── DIARIO_CAPTURAS.md           Capture log template

Real-Time Experiments/
├── scripts/                      Original capture scripts (DO NOT MODIFY)
└── Paper_Tables_and_Figures/     Scripts to reproduce each paper table/figure
    ├── Table_1_Dataset_Stats/        ISCXVPN2016 non-VPN subset statistics
    ├── Table_2_Preprocessing_Search/ nt2txt preprocessing parameter search
    ├── Table_3_Hyperparameter_Tuning/ Grid search + R2.2 depth search (REVISED)
    ├── Table_4_PerClass_Performance/  Per-class metrics from JSON (REVISED)
    ├── Table_5_RealTime_InDistribution/ Live-capture results (REVISED — new campaign)
    ├── Table_6_OOD_Generalization/    OOD results (REVISED — apps updated)
    ├── Baseline_Comparison/           Transformer vs LSTM vs XGBoost/RF (REVISED)
    └── Figures_1_2_3_Workflow_Diagrams/

~/ijis_rt/                        Live-capture campaign directory (macOS)
    ├── run_experiment_v2.sh
    ├── rt_experiment_v2.py
    ├── captura_comun.py
    ├── pesos.pkl                  → CE baseline (FINAL MODEL, updated 23 Nov 2026)
    ├── pesos_articulo2023.pkl     → Original model
    ├── LEEME.md
    └── DIARIO_CAPTURAS.md
```

---

## Revision experiments summary

### R2.2 — Depth search (encoder layers)

Script: `revision_IJIS/01_exp_profundidad.py`  
Results: `revision_IJIS/resultados/exp_R2.2_profundidad/tabla_R2.2.csv`

| Layers | PE | Params | Val acc (mean±sd) | Test acc | Macro-F1 | Email recall |
|---|---|---|---|---|---|---|
| 1 | No | 149,213 | 92.42±0.77% | 92.94% | 69.09% | 0% |
| 1 | Yes | 149,213 | 93.36±0.13% | 93.54% | 89.04% | 82.4% |
| **2** | **Yes** | **291,921** | **93.39±0.12%** | **93.50%** | **90.06%** | **88.2%** |
| 2 | No | 291,921 | 93.21±0.36% | 93.46% | 86.17% | 70.6% |
| 4 | No | 577,337 | 73.19±8.77% | 87.59% | 52.24% | 0% |
| 4 | Yes | 577,337 | 33.57±3.92% | 41.26% | 19.96% | 0% |
| 6 | No | 862,753 | 51.16±14.85% | 67.00% | 32.24% | 0% |
| 6 | Yes | 862,753 | 39.59±13.64% | 66.35% | 32.44% | 0% |

**Finding:** 4 and 6 layers fail to train with the published hyperparameters (Post-LN architecture). Best = 2 layers + PE.

### R2.3 — Imbalance mitigation

Script: `revision_IJIS/02_exp_desbalance.py`  
Results: `revision_IJIS/resultados/exp_R2.3_desbalance/tabla_R2.3.csv`

| Method | Accuracy | Macro-F1 | Email recall | Email precision |
|---|---|---|---|---|
| **CE baseline (retained)** | **93.50%** | **90.06%** | **88.2%** | **83.3%** |
| Inverse class weights | 88.35% | 61.26% | 94.1% | 4.3% (unusable) |
| Focal loss γ=2 | 93.46% | 90.57% | 76.5% | 92.9% |

**Decision:** CE baseline retained. Inverse weights: unusable precision. Focal loss: worse Email recall than baseline with no overall gain. Email real-time failure is a data-volume problem (too few packets per 8-second window), not a loss function problem.

### R2.4 — Classical ML baselines (flow statistics)

Script: `revision_IJIS/03_exp_ml_clasico.py`  
Results: `revision_IJIS/resultados/exp_R2.4_ml_clasico/tabla_R2.4.csv`

| Model | Accuracy | Macro-F1 | κ | MCC | Email recall |
|---|---|---|---|---|---|
| XGBoost | 99.39% | 97.34% | 0.991 | 0.991 | 94.1% |
| Random Forest | 99.36% | 97.40% | 0.990 | 0.990 | 94.1% |
| **Transformer 2L+PE (ours)** | **93.50%** | **90.06%** | **0.901** | **0.906** | **88.2%** |
| LSTM (Maitin et al. 2025) | 88.87% | — | 0.83 | 0.84 | — |

**Presentation:** Classical ML outperforms the Transformer on this dataset. Presented as an honest finding. The Transformer's contribution is payload-based classification without manual feature engineering.

### R2.5 — Real-time preprocessing audit

Script: `revision_IJIS/03_clasificar_capturas.py`  
Results: `revision_IJIS/resultados/exp_R2.5_tiempo_real/`

Three discrepancies found between training preprocessing and original live-capture preprocessing:
1. Flow key: training=unidirectional+protocol, live=bidirectional (72.5% Chat semi-flows affected)
2. UDP port: NaN bug in `rt_experiment.py` (affects all UDP traffic including VoIP)
3. Semi-flow segmentation: training discards at timeout, live slides window

The new campaign (`ijis_rt/`, `run_experiment_v2.sh`) uses `captura_comun.py`, which is faithful to training preprocessing. All results for Tables 5 and 6 must come from this new campaign, **not** from the original `Real-Time Experiments/results/` data.

---

## Consistency checklist (verified 23 November 2026)

| Item | Paper value | Code/JSON source | Status |
|---|---|---|---|
| Total parameters | 291,921 | `capas2_PE-si_semilla0.json`.parametros | ✓ |
| Test accuracy | 93.50% | `capas2_PE-si_semilla0.json`.accuracy | ✓ |
| Macro-F1 | 90.06% | `capas2_PE-si_semilla0.json`.macro_f1 | ✓ |
| Balanced accuracy | 88.4% | `capas2_PE-si_semilla0.json`.balanced_accuracy | ✓ |
| κ | 0.90 | `capas2_PE-si_semilla0.json`.kappa | ✓ |
| MCC | 0.91 | `capas2_PE-si_semilla0.json`.mcc | ✓ |
| Chat: P=96.15%, Se=75.19%, F1=84.39% | JSON por_clase.Chat | ✓ |
| Email: P=83.33%, Se=88.24%, F1=85.71% | JSON por_clase.Email | ✓ |
| FileTransfer: P=86.14%, Se=99.76%, F1=92.46% | JSON por_clase.FileTransfer | ✓ |
| Streaming: P=99.41%, Se=78.96%, F1=88.01% | JSON por_clase.Streaming | ✓ |
| VoIP: P=99.59%, Se=99.85%, F1=99.72% | JSON por_clase.VoIP | ✓ |
| XGBoost accuracy | 99.39% | `xgboost.json`.accuracy | ✓ |
| RF macro-F1 | 97.40% | `random_forest.json`.macro_f1 | ✓ |
| OOD Chat app | WhatsApp Web | `table6_ood_generalization.py` | ✓ FIXED |
| OOD FileTransfer app | uTorrent | `table6_ood_generalization.py` | ✓ FIXED |
| OOD Streaming app | Amazon Prime Video | `table6_ood_generalization.py` | ✓ FIXED |
| OOD VoIP app | Zoom | `table6_ood_generalization.py` | ✓ FIXED |
| Table 5 filters | All windows (n_semiflows>0) | `table5_realtime_indistribution.py` | ✓ FIXED |
| pesos.pkl | CE baseline (2L+PE) | `~/ijis_rt/pesos.pkl` | ✓ FIXED |
| baseline_comparison.py | 93.50%, XGBoost, RF | `baseline_comparison.py` | ✓ FIXED |
| table4_perclass_performance.py | Reads from JSON | `table4_perclass_performance.py` | ✓ FIXED |
| LSTM improvement | +4.63 pp (93.50−88.87) | manuscript text | ✓ |

---

## Status: live-capture campaign (November 2026)

The campaign was run on Alberto's MacBook Air. Data is in:
- `Paper_Tables_and_Figures/Table_5_RealTime_InDistribution/data/results_log.csv` (runs 1–3)
- `Paper_Tables_and_Figures/Table_6_OOD_Generalization/data/results_log.csv` (runs 7–9)

**In-distribution apps used (Table 5, runs 1–3) — COMPLETED:**
| Class | Application | Notes |
|---|---|---|
| Chat | Slack | Confirmed from TLS/DNS analysis of captures |
| Email | Thunderbird | IMAP/SMTPS with ≥5 MB attachments |
| FileTransfer | wget (HTTPS) | Large file download |
| Streaming | Spotify | Audio streaming over HTTPS |
| VoIP | Microsoft Teams + G.711 RTP simulator | Video calls + synthetic RTP |

**OOD apps used (Table 6, runs 7–9) — PARTIALLY COMPLETED:**
| Class | Application | Status | Notes |
|---|---|---|---|
| Chat | WhatsApp Web | ✅ Done | 0.0%, classified as Streaming(86%) |
| FileTransfer | uTorrent | ✅ Done | 63.3%, partial generalisation |
| Streaming | Amazon Prime Video | ⚠️ Incomplete | 0.0%, FileTransfer(100%). HTTPS/DASH same as training protocol. Needs true protocol-OOD source (RTSP/RTP, port 554 blocked on capture network) |
| VoIP | Zoom | ⚠️ Incomplete | 0.0%, Streaming(100%). Paper table shows "Mumble" — must fix to "Zoom". Needs true protocol-OOD source |

**Known errors in manuscript to fix before submission:**
1. Table 6, Chat ref accuracy: paper says 100.0% → correct value is 93.3% (same as Table 5)
2. Table 6, VoIP OOD app: paper says "Mumble" → correct value is "Zoom"
3. Limitations section: "10 Email windows producing no prediction" → correct value is 0

**To complete OOD experiments (Streaming and VoIP):**
- **Streaming OOD:** Replace Amazon Prime Video with an RTSP/RTP source. Easiest option: local RTSP server with `ffmpeg -re -i video.mp4 -f rtsp rtsp://localhost:8554/live` + `mpv rtsp://localhost:8554/live`. Requires any .mp4 file and ffmpeg installed.
- **VoIP OOD:** Replace Zoom with a genuinely different VoIP protocol. Options (easiest to hardest): FaceTime (Apple proprietary, needs Apple device), Signal voice call (Signal Protocol), Telegram voice (MTProto).
- After capturing, add rows to `data/results_log.csv` and re-run `table6_ood_generalization.py`.
