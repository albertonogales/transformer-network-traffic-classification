# Table 4 — Per-class performance of the five-class Transformer model
# Figure 4 — Confusion matrix for the five-class network traffic classification
#
# Source: revision_IJIS/resultados/exp_R2.2_profundidad/capas2_PE-si_semilla0.json
#         (best model: 2 encoder layers + sinusoidal PE, CE loss, 291,921 parameters)
#
# All values derived from the JSON result file (not manually transcribed from image).
# Confusion matrix (rows=true, cols=predicted):
#   Chat:         [100,   0,  15,  17,   1]
#   Email:        [  0,  15,   1,   0,   1]
#   FileTransfer: [  0,   2, 26744, 55,  6]
#   Streaming:    [  3,   1, 4280, 16377, 80]
#   VoIP:         [  1,   0,   6,  25, 21386]

import os, json, math

# Load from the authoritative JSON result file
RESULT_JSON = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'revision_IJIS', 'resultados', 'exp_R2.2_profundidad',
    'capas2_PE-si_semilla0.json'))

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, c-h)*100, min(1.0, c+h)*100

if os.path.exists(RESULT_JSON):
    with open(RESULT_JSON) as f:
        d = json.load(f)
    pc = d['por_clase']
    print("Table 4 — Per-class performance (2-layer Transformer + PE, CE loss)")
    print(f"{'Class':<14}{'n':>6}{'Precision':>11}{'Recall (Se)':>13}{'95% CI':>16}{'F1':>8}{'Specificity':>13}")
    print("-"*83)
    for cls in ['Chat', 'Email', 'FileTransfer', 'Streaming', 'VoIP']:
        r = pc[cls]
        lo, hi = r['sensibilidad_IC95']
        print(f"{cls:<14}{r['n']:>6}{r['precision']*100:>10.2f}%{r['sensibilidad']*100:>12.2f}%"
              f"  [{lo*100:5.1f}–{hi*100:5.1f}%]{r['f1']*100:>7.2f}%{r['especificidad']*100:>12.2f}%")
    print(f"\nOverall: Accuracy={d['accuracy']*100:.2f}%  macro-F1={d['macro_f1']*100:.2f}%"
          f"  balanced-acc={d['balanced_accuracy']*100:.1f}%  κ={d['kappa']:.3f}  MCC={d['mcc']:.3f}")
    print(f"Parameters: {d['parametros']:,}  |  Training: {d['minutos_entrenamiento']:.0f} min (Apple M4 CPU)")
else:
    print(f"Result JSON not found at:\n  {RESULT_JSON}")
    print("Run revision_IJIS/01_exp_profundidad.py to generate it.")

img = os.path.join(os.path.dirname(__file__), 'data', 'confusion_matrix_best_model.png')
if os.path.exists(img):
    print(f"\nFigure 4 image: {img}")
