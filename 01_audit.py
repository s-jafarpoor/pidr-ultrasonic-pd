"""
01_audit.py  -  Step 1: data audit + shortcut audit  (runs in ~1-2 minutes, CPU only)
=====================================================================================
What it answers (all of this becomes Section 3 of the paper):
  A. Basic physics of the recordings: SPL, SNR above the floor, arrival time, spectrum.
  B. Temporal dependence inside each file (is random CV leaky?)
  C. SHORTCUT AUDIT: how far do 3 trivial numbers (loudness, peak, arrival index) go?
     -> if they already give ~100 %, "99 % accuracy" papers on this data measure the lab,
        not the discharge physics. This motivates the whole method.
  D. Random-CV vs blocked-CV optimism gap.
  E. Predicted detection budget vs distance with ISO 9613-1 absorption.

Usage:   python 01_audit.py                (CSV files next to this script or in ./dataset)
         python 01_audit.py --data_dir PATH
Output:  audit_results/audit_report.txt  (+ figures)  -> send audit_report.txt to Claude
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_common as pc

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

data_dir = pc.find_data_dir(args.data_dir)
out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_results")
os.makedirs(out, exist_ok=True)
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


X, y, row = pc.load_data(data_dir)
log(f"data_dir = {data_dir}")
log(f"X shape = {X.shape}, classes = {np.bincount(y).tolist()}")
log(f"NaN = {int(np.isnan(X).sum())}, Inf = {int(np.isinf(X).sum())}")
_, uniq = np.unique(X.round(10), axis=0, return_index=True)
log(f"duplicate rows = {len(X) - len(uniq)}")

# ------------------------------------------------------------------ A. physics summary
P0 = 20e-6
Fe = pc.features(X)
rms = np.sqrt((X ** 2).mean(1))
pk = np.abs(X).max(1)
floor_rms = float(np.median(rms[y == pc.NOISE_CLASS]))
log("\n=== A. Per-class physics summary (median [IQR]) ===")
log(f"lab noise-floor RMS (median of Noise class) = {floor_rms:.3e} Pa = {20*np.log10(floor_rms/P0):.1f} dB SPL")
hdr = f"{'class':8s} {'RMS dB SPL':>16s} {'peak dB SPL':>16s} {'peak SNR dB':>14s} {'peak idx':>12s} {'f_centroid kHz':>16s} {'kurtosis':>12s}"
log(hdr)


def q(v):
    a, b, c = np.percentile(v, [25, 50, 75])
    return f"{b:7.2f} [{a:.1f},{c:.1f}]"


for c in range(pc.N_CLASSES):
    m = y == c
    log(f"{pc.CLASS_NAMES[c]:8s} {q(20*np.log10(rms[m]/P0)):>16s} {q(20*np.log10(pk[m]/P0)):>16s} "
        f"{q(20*np.log10(pk[m]/floor_rms)):>14s} {q(Fe[m,2]*pc.N):>12s} {q(Fe[m,9]):>16s} {q(Fe[m,4]):>12s}")

# ------------------------------------------------------------------ B. temporal dependence
from scipy.stats import spearmanr
log("\n=== B. Temporal dependence along recording order ===")
log("lag-1 autocorr of log_rms / f_centroid ; Spearman(row, log_rms) ; consecutive-vs-random waveform corr")
rng = np.random.default_rng(0)
for c in range(pc.N_CLASSES):
    m = np.where(y == c)[0]
    m = m[np.argsort(row[m])]
    a = Fe[m, 0]
    b = Fe[m, 9]
    ac1 = np.corrcoef(a[:-1], a[1:])[0, 1]
    ac2 = np.corrcoef(b[:-1], b[1:])[0, 1]
    rho, p = spearmanr(np.arange(len(a)), a)
    W = X[m]
    W = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-12)
    cons = np.mean([np.dot(W[i], W[i + 1]) / pc.N for i in range(len(W) - 1)])
    rp = rng.permutation(len(W))
    rnd = np.mean([np.dot(W[rp[i]], W[rp[i + 1]]) / pc.N for i in range(len(W) - 1)])
    log(f"{pc.CLASS_NAMES[c]:8s} ac1(log_rms)={ac1:+.3f} ac1(fc)={ac2:+.3f} drift rho={rho:+.3f} (p={p:.1e}) "
        f"corr consecutive={cons:+.3f} random={rnd:+.3f}")

# ------------------------------------------------------------------ C/D. shortcut audit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

log("\n=== C/D. Shortcut audit: macro-F1 (mean +- sd over 5 folds) ===")
sets = {
    "log_rms only": [0], "peak_idx only (arrival time)": [2], "log_peak only": [1],
    "SHORTCUT probe (rms+peak+idx)": pc.SHORTCUT_IDX,
    "all 18 features": list(range(len(pc.FEATURE_NAMES))),
    "shape-only (no rms/peak/idx)": [i for i in range(len(pc.FEATURE_NAMES)) if i not in pc.SHORTCUT_IDX],
}
folds_r = pc.random_folds(y, 5, 0)
folds_b = pc.blocked_purged_folds(y, row, 5, 10)
log(f"{'feature set':34s} {'model':5s} {'random CV':>14s} {'blocked CV':>14s}")
for name, cols in sets.items():
    for mname in ["LR", "RF"]:
        res = []
        for folds in (folds_r, folds_b):
            s = []
            for tr, te in folds:
                if mname == "LR":
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                else:
                    clf = RandomForestClassifier(300, n_jobs=-1, random_state=0)
                clf.fit(Fe[tr][:, cols], y[tr])
                s.append(f1_score(y[te], clf.predict(Fe[te][:, cols]), average="macro"))
            res.append(f"{np.mean(s):.4f}+-{np.std(s):.4f}")
        log(f"{name:34s} {mname:5s} {res[0]:>14s} {res[1]:>14s}")

# ------------------------------------------------------------------ E. detection budget
log("\n=== E. Median-PD peak SNR above lab floor vs distance (spreading + ISO 9613-1, 20C 50%RH) ===")
f = np.fft.rfftfreq(pc.NFFT, 1 / pc.FS)
log(f"alpha(dB/m) @25k={pc.iso9613_alpha(np.array(25e3),20,50):.2f}  @40k={pc.iso9613_alpha(np.array(40e3),20,50):.2f}  "
    f"@60k={pc.iso9613_alpha(np.array(60e3),20,50):.2f}  @100k={pc.iso9613_alpha(np.array(100e3),20,50):.2f}")
alpha = pc.iso9613_alpha(f, 20.0, 50.0)
dists = [1, 2, 4, 6, 8, 10, 12, 15, 20]
log("class    " + " ".join(f"{d:>6d}m" for d in dists))
budget = {}
for c in range(3):
    m = y == c
    Xc = np.fft.rfft(X[m], n=pc.NFFT)
    row_s = []
    for d in dists:
        H = (1 / d) * 10 ** (-alpha * (d - 1) / 20)
        z = np.fft.irfft(Xc * H[None, :], n=pc.NFFT)[:, :pc.N]
        row_s.append(float(np.median(20 * np.log10(np.abs(z).max(1) / floor_rms))))
    budget[c] = row_s
    log(f"{pc.CLASS_NAMES[c]:8s} " + " ".join(f"{v:7.1f}" for v in row_s))

# ------------------------------------------------------------------ figures
plt = pc.plot_style()
t_us = np.arange(pc.N) / pc.FS * 1e6
fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
for c in range(pc.N_CLASSES):
    m = y == c
    med = np.median(X[m], 0)
    lo, hi = np.percentile(X[m], [25, 75], 0)
    ax = axs.flat[c]
    ax.fill_between(t_us, lo, hi, color=pc.CLASS_COLORS[c], alpha=0.25, linewidth=0)
    ax.plot(t_us, med, color=pc.CLASS_COLORS[c])
    ax.set_title(pc.CLASS_NAMES[c])
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("pressure [Pa]")
fig.suptitle("Median waveform and inter-quartile band per class")
fig.savefig(os.path.join(out, "fig_waveforms.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for c in range(pc.N_CLASSES):
    S = np.abs(np.fft.rfft(X[y == c], n=pc.NFFT)) ** 2
    ax.plot(f / 1e3, 10 * np.log10(S.mean(0) + 1e-30), color=pc.CLASS_COLORS[c], label=pc.CLASS_NAMES[c])
ax.set_xlim(0, 150)
ax.set_xlabel("frequency [kHz]")
ax.set_ylabel("mean power [dB]")
ax.set_title("Average power spectrum per class")
ax.legend()
fig.savefig(os.path.join(out, "fig_spectra.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for i, (T, rh) in enumerate([(0, 20), (20, 50), (35, 90)]):
    ax.plot(f / 1e3, pc.iso9613_alpha(f, T, rh), color=pc.PALETTE[i], label=f"{T} °C, {rh} % RH")
ax.axvspan(25, 100, color=pc.GRID, alpha=0.5, linewidth=0)
ax.set_xlim(0, 150)
ax.set_xlabel("frequency [kHz]")
ax.set_ylabel("absorption [dB/m]")
ax.set_title("ISO 9613-1 atmospheric absorption (shaded: measurement band)")
ax.legend()
fig.savefig(os.path.join(out, "fig_absorption.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for c in range(3):
    ax.plot(dists, budget[c], "o-", color=pc.CLASS_COLORS[c], label=pc.CLASS_NAMES[c], markersize=6)
ax.axhline(0, color=pc.INK2, linewidth=1)
ax.set_xlabel("distance [m]")
ax.set_ylabel("median peak SNR above lab floor [dB]")
ax.set_title("Acoustic budget vs distance (spreading + absorption)")
ax.legend()
fig.savefig(os.path.join(out, "fig_budget.png"))
plt.close(fig)

with open(os.path.join(out, "audit_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
np.save(os.path.join(out, "floor_rms.npy"), np.array(floor_rms))
print(f"\nSaved to {out}  ->  send audit_report.txt")
