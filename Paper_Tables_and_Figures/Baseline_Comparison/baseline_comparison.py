# Baseline Comparison — Table 5 in paper comparing Transformer vs LSTM and classical ML baselines
#
# Sources:
#   data/lstm_grid_results.csv           — LSTM grid search runs (this study)
#   Maitin et al. (2025)                 — external paper (not in CSV)
#   revision_IJIS/resultados/            — R2.2 depth search and R2.4 classical ML results
#
# Best Transformer (R2.2 revision, IJIS round 2):
#   2 encoder layers + sinusoidal PE, 291,921 parameters
#   Cross-entropy loss (CE baseline from R2.3)
#   accuracy=93.50%, macro-F1=90.06%, kappa=0.90, MCC=0.91, balanced-accuracy=88.4%
#   Training: ~311 min on Apple M4 CPU (16 GB unified memory)
#
# Classical ML baselines (R2.4, flow statistics — no payload):
#   XGBoost: accuracy=99.39%, macro-F1=97.34%, kappa=0.991, MCC=0.991
#   Random Forest: accuracy=99.36%, macro-F1=97.40%, kappa=0.990, MCC=0.990
import pandas as pd, json, os

BASE = os.path.dirname(__file__)
RESULTS = os.path.normpath(os.path.join(
    BASE, '..', '..',
    'revision_IJIS', 'resultados'))

lstm  = pd.read_csv(os.path.join(BASE, 'data', 'lstm_grid_results.csv'))
best_lstm = lstm.sort_values('accuracy_test', ascending=False).iloc[0]

with open(os.path.join(RESULTS, 'exp_R2.4_ml_clasico', 'xgboost.json')) as f:
    xgb = json.load(f)
with open(os.path.join(RESULTS, 'exp_R2.4_ml_clasico', 'random_forest.json')) as f:
    rf = json.load(f)
with open(os.path.join(RESULTS, 'exp_R2.2_profundidad', 'capas2_PE-si_semilla0.json')) as f:
    tr = json.load(f)

print("Table 5 — Baseline Comparison")
print(f"{'Model':<35}{'Features':<20}{'Accuracy':>10}{'Macro-F1':>10}{'Kappa':>8}{'MCC':>7}")
print("-"*93)
print(f"{'XGBoost (R2.4)':<35}{'flow statistics':<20}{xgb['accuracy']*100:>9.2f}%{xgb['macro_f1']*100:>9.2f}%{xgb['kappa']:>8.3f}{xgb['mcc']:>7.3f}")
print(f"{'Random Forest (R2.4)':<35}{'flow statistics':<20}{rf['accuracy']*100:>9.2f}%{rf['macro_f1']*100:>9.2f}%{rf['kappa']:>8.3f}{rf['mcc']:>7.3f}")
print(f"{'Transformer 2L+PE (ours, R2.2)':<35}{'nt2txt payload':<20}{tr['accuracy']*100:>9.2f}%{tr['macro_f1']*100:>9.2f}%{tr['kappa']:>8.3f}{tr['mcc']:>7.3f}")
print(f"{'LSTM (Maitin et al. 2025)':<35}{'nt2txt payload':<20}{'88.87%':>10}{'  —   ':>10}{'0.83':>8}{'0.84':>7}")
print(f"{'LSTM (this study)':<35}{'nt2txt payload':<20}{best_lstm['accuracy_test']*100:>9.2f}%{'':>10}{best_lstm['cohen']:>8.3f}{best_lstm['matthews']:>7.3f}")
print()
print("Notes:")
print("  Transformer: 2-layer encoder + sinusoidal PE, 291,921 params, CE loss (best from R2.2/R2.3).")
print("  Classical ML: trained on 23 per-flow statistics (packet length + inter-arrival time);")
print("    same 90/10 stratified split and 5-fold CV as Transformer.")
print("  LSTM (Maitin et al.): uses nt2txt preprocessing, results from published paper.")
print(f"  LSTM (this study) best config: accuracy={best_lstm['accuracy_test']*100:.2f}%,"
      f" epochs={best_lstm['epochs']}, batch={best_lstm['batch_size']}, lr={best_lstm['learning_rate']}")
