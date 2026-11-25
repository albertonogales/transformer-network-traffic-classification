# Table 3 — Hyperparameter tuning for the Transformer model (IJIS revision, November 2026)
#
# Part A — Original grid search (2023, fixed hyperparameters)
#   Source: data/grid_search_results.csv (42 runs)
#   Fixed: 1 encoder layer, no positional encoding
#   Explored: batch size, epochs, learning rate, MLP neurons, dropout, attention heads
#
# Part B — Depth search (R2.2, 2026 revision)
#   Source: revision_IJIS/resultados/exp_R2.2_profundidad/tabla_R2.2.csv
#   Explored: encoder layers {1, 2, 4, 6} × positional encoding {Yes, No}
#   Fixed: batch=100, epochs=15, lr=3e-4, heads=8, head_size=6, ff_dim=256, mlp=250, dropout=0.2
#
# Best overall (manuscript final model):
#   2 encoder layers + sinusoidal PE, 291,921 parameters
#   val-accuracy = 93.39 ± 0.12% (5-fold)
#   test-accuracy = 93.50%, macro-F1 = 90.06%, kappa = 0.901, MCC = 0.906
#   balanced-accuracy = 88.4%, training ≈ 311 min on Apple M4 CPU
import pandas as pd, os

BASE = os.path.dirname(__file__)

# --- Part A: original grid search ---
grid_csv = os.path.join(BASE, 'data', 'grid_search_results.csv')
if os.path.exists(grid_csv):
    df = pd.read_csv(grid_csv)
    print(f"Part A — Original grid search ({len(df)} runs, 1 encoder layer, no PE)")
    print("  Explored: batch size (100/500/1000/2500), epochs (15/50/100/200/500),")
    print("            learning rate (1e-3/5e-4/3e-4/1e-4), MLP neurons (100/150/250),")
    print("            dropout (0.15/0.20/0.25/0.50/0.75), attention heads (4/8)")
    best_hash = 'fe693df6778a35ca5a920dfa73b35b69'
    best_row = df[df['hashModelo'] == best_hash]
    if not best_row.empty:
        r = best_row.iloc[0]
        print(f"  Best (hash={best_hash}): accuracy={r['accuracy_test']*100:.2f}%,"
              f" kappa={r['cohen']:.3f}, MCC={r['matthews']:.3f}")
        print(f"    batch={r['batch_size']}, epochs={r['epochs']}, lr={r['learning_rate']},"
              f" heads={r['num_heads']}, mlp={r['mlp_units']}, dropout={r['dropout']}")

# --- Part B: depth search (R2.2) ---
depth_csv = os.path.join(BASE, '..', '..',
                         'revision_IJIS', 'resultados', 'exp_R2.2_profundidad', 'tabla_R2.2.csv')
depth_csv = os.path.normpath(depth_csv)
print()
if os.path.exists(depth_csv):
    d = pd.read_csv(depth_csv)
    print("Part B — Depth search (R2.2, 8 configurations)")
    print(f"{'Layers':>7}{'PE':>5}{'Params':>10}{'Val acc (mean±sd)':>20}{'Test acc':>10}"
          f"{'Macro-F1':>10}{'Bal acc':>9}{'Recall Email':>14}")
    print("-" * 90)
    for _, r in d.iterrows():
        mark = " ← BEST" if (r['capas'] == 2 and r['positional_encoding'] == 'si') else ""
        print(f"{int(r['capas']):>7}{'Yes' if r['positional_encoding']=='si' else 'No':>5}"
              f"{int(r['parametros']):>10}"
              f"  {r['val_accuracy_media']*100:5.2f}±{r['val_accuracy_sd']*100:4.2f}%"
              f"{r['test_accuracy']*100:>10.2f}%"
              f"{r['test_macro_f1']*100:>10.2f}%"
              f"{r['test_balanced_accuracy']*100:>9.2f}%"
              f"{r['recall_Email']*100:>14.1f}%{mark}")
else:
    print("Part B — Depth search table not found at expected path.")
    print("  Expected:", depth_csv)

import json as _json, os as _os
_best_json = _os.path.normpath(_os.path.join(BASE, '..', '..',
    'revision_IJIS', 'resultados', 'exp_R2.2_profundidad',
    'capas2_PE-si_semilla0.json'))
with open(_best_json) as _f:
    _m = _json.load(_f)
print()
print(f"Final model (manuscript): 2 layers + sinusoidal PE, {_m['parametros']:,} params,")
print("  CE loss (cross-entropy baseline from R2.3).")
print(f"  test-accuracy={_m['accuracy']*100:.2f}%, macro-F1={_m['macro_f1']*100:.2f}%,"
      f" kappa={_m['kappa']:.3f}, MCC={_m['mcc']:.3f},"
      f" balanced-acc={_m['balanced_accuracy']*100:.1f}%.")
