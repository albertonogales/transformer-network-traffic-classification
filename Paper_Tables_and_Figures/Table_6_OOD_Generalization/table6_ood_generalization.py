# Table 6 — Cross-application generalization results (OOD)
# Source: data/results_log.csv (runs 7, 8, 9 per class)
# Capture scripts: ../Table_5_RealTime_InDistribution/scripts/run_experiment.sh
#
# OOD applications used (IJIS revision, November 2026):
#   Chat         → WhatsApp Web      (reference: Slack / Gmail Chat)
#   FileTransfer → uTorrent          (reference: HTTPS wget downloads)
#   Streaming    → Amazon Prime Video (reference: Spotify — Netflix excluded: in ISCXVPN2016 training data)
#   VoIP         → Zoom              (reference: Microsoft Teams + G.711 RTP simulator)
#
# Current results (November 2026 campaign):
#   Chat:         0.0%  [0.0–11.4%]  — predicted as Streaming(86%) / FileTransfer(6%)
#   FileTransfer: 63.3% [45.5–78.1%] — partial generalisation; Streaming(36%) misclassified
#   Streaming:    0.0%  [0.0–11.4%]  — predicted as FileTransfer(100%)  *** INCOMPLETE — see note ***
#   VoIP:         0.0%  [0.0–11.4%]  — predicted as Streaming(100%)     *** INCOMPLETE — see note ***
#
# *** INCOMPLETE OOD EXPERIMENTS ***
# Streaming OOD: Amazon Prime Video captured (runs 7-9). Uses HTTPS/DASH, same protocol family
#   as training (Netflix/YouTube/Vimeo). Not strictly protocol-OOD. Needs replacement with
#   RTSP/RTP source (port 554 blocked on capture network) or justification added to paper.
#   Suggested alternatives: local RTSP server via ffmpeg, or network with open port 554.
#
# VoIP OOD: Zoom captured (runs 7-9). Uses SRTP/UDP, same protocol family as training
#   (Skype/Hangouts). Paper currently shows "Mumble" in table — must be corrected to "Zoom".
#   For genuinely protocol-OOD VoIP, suggested alternatives: FaceTime (Apple proprietary),
#   Signal voice call (Signal Protocol), Telegram voice (MTProto).
import pandas as pd, math, os
from collections import defaultdict

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, 'data', 'results_log.csv'))
df = df[df['dominant'].notna() & (df['dominant'] != 'None')]
df_ood = df[df['run'].isin([7, 8, 9])]

OOD_APPS = {
    'Chat': 'WhatsApp Web',
    'FileTransfer': 'uTorrent',
    'Streaming': 'Amazon Prime Video',
    'VoIP': 'Zoom',
}

def wilson_ci(acc, n, z=1.96):
    if n == 0: return 0, 0
    c = (acc + z**2/(2*n)) / (1 + z**2/n)
    m = z * math.sqrt(acc*(1-acc)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0, c-m), min(1, c+m)

print("Table 6 — Cross-application generalization results (OOD)")
print(f"{'Class':<14}{'OOD App':<18}{'Accuracy':>10}{'95% CI':>18}{'Predicted as (majority)':>30}")
print("-"*95)

for cls, app in OOD_APPS.items():
    sub = df_ood[df_ood['class'] == cls]
    n = len(sub)
    correct = (sub['dominant'] == cls).sum()
    acc = correct / n if n > 0 else 0
    lo, hi = wilson_ci(acc, n)
    dist = defaultdict(int)
    for _, row in sub.iterrows():
        if row['dominant'] != cls:
            dist[row['dominant']] += 1
    wrong_total = sum(dist.values())
    top = sorted(dist.items(), key=lambda x: -x[1])
    pred_str = ' / '.join([f"{c}({v*100//n:.0f}%)" for c,v in top[:2]]) if top else '—'
    print(f"{cls:<14}{app:<18}{acc*100:>9.1f}%  [{lo*100:.1f}–{hi*100:.1f}%]{pred_str:>30}")
