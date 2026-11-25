# Table 5 — Per-class real-time classification results (In-Distribution)
# Source: data/results_log.csv (new campaign, November 2026)
# Capture scripts: scripts/run_experiment.sh, scripts/rt_experiment.py
# VoIP simulator: scripts/rtp_simulate.py
#
# In-distribution applications (November 2026 campaign):
#   Chat         → Slack (confirmed from TLS/DNS analysis of captures)
#   Email        → Thunderbird (IMAP/SMTPS, attachments ≥5 MB)
#   FileTransfer → HTTPS wget downloads (large file transfers)
#   Streaming    → Spotify (audio streaming) — runs 1-3
#   VoIP         → G.711 RTP simulator (rtp_simulate.py) + Microsoft Teams
#
# Protocol (declared before capture):
#   - All windows from all runs enter the table (runs 1–3 per class, 10 windows each = 30 windows).
#   - Only exclusion: n_semiflows == 0 (no semi-flow formed in that window).
#   - No post-hoc timestamp filters.
#
# Usage: bash scripts/run_experiment.sh <Class> <RunNumber>
# Example: bash scripts/run_experiment.sh Chat 1
import pandas as pd, math, os
from scipy import stats

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, 'data', 'results_log.csv'))

# Keep only windows with at least one semi-flow (declared exclusion criterion)
df = df[df['n_semiflows'] > 0]

# In-distribution runs: 1, 2, 3 per class (all included, no timestamp cherry-picking)
FILTERS = {
    'Chat':         lambda d: d[(d['class']=='Chat')         & (d['run'].isin([1,2,3]))],
    'Email':        lambda d: d[(d['class']=='Email')        & (d['run'].isin([1,2,3]))],
    'FileTransfer': lambda d: d[(d['class']=='FileTransfer') & (d['run'].isin([1,2,3]))],
    'Streaming':    lambda d: d[(d['class']=='Streaming')    & (d['run'].isin([1,2,3]))],
    'VoIP':         lambda d: d[(d['class']=='VoIP')         & (d['run'].isin([1,2,3]))],
}

def wilson_ci(acc, n, z=1.96):
    c = (acc + z**2/(2*n)) / (1 + z**2/n)
    m = z * math.sqrt(acc*(1-acc)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0, c-m), min(1, c+m)

print("Table 5 — Per-class real-time classification results")
print(f"{'Class':<14}{'Windows':>8}{'Accuracy':>10}{'95% CI':>18}{'Pre(ms)':>9}{'Inf(ms)':>8}{'Total(ms)':>11}{'RTF':>7}")
print("-"*90)

all_pre, all_inf = [], []
for cls, filt in FILTERS.items():
    sub = filt(df)
    n = len(sub)
    if n == 0: continue
    correct = (sub['dominant'] == cls).sum()
    acc = correct / n
    lo, hi = wilson_ci(acc, n)
    pre = sub['t_preprocess'].mean() * 1000
    inf_ = sub['t_inference'].mean() * 1000
    tot = sub['t_total'].mean() * 1000
    rtf = sub['t_total'].mean() / 8.0
    print(f"{cls:<14}{n:>8}{acc*100:>9.1f}%  [{lo*100:.1f}–{hi*100:.1f}%]{pre:>9.1f}{inf_:>8.1f}{tot:>11.1f}{rtf:>7.3f}")
    all_pre.extend(sub['t_preprocess'].tolist())
    all_inf.extend(sub['t_inference'].tolist())

t_stat, p_val = stats.ttest_rel(all_pre, all_inf)
print(f"\nPaired t-test (preprocessing vs inference): t={t_stat:.2f}, p={p_val:.3e}")
