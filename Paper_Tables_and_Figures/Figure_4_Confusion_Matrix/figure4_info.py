# Figure 4 — Confusion matrix for the five-class network traffic classification
# Source image: ../Table_4_PerClass_Performance/data/confusion_matrix_best_model.png
#
# Table 4 and Figure 4 share the same source; see Table_4_PerClass_Performance/ for:
#   - data/confusion_matrix_best_model.png
#   - table4_perclass_performance.py
import os
img = os.path.join(os.path.dirname(__file__),
                   '..', 'Table_4_PerClass_Performance',
                   'data', 'confusion_matrix_best_model.png')
img = os.path.abspath(img)
if os.path.exists(img):
    print(f"Figure 4 image: {img}")
    print("Open the image to view the confusion matrix.")
else:
    print(f"Image not found at: {img}")
