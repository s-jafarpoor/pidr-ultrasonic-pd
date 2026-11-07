"""
make_figures.py  -  publication figures for the manuscript (Figure 1 ... Figure 12)
==================================================================================
Put this file in the same folder as the CSV data and the other scripts, then run:

    python make_figures.py

It reads (whatever exists):
    the 4 CSV files                     -> Figures 2, 3, 4
    results/runs.csv                    -> Figures 6, 7          (step 2)
    results_ext/physics_centroid.csv    -> Figure 5              (step 5)
    results_ext/main_runs.csv           -> Figure 8              (step 5)
    results_ext/scores/*.npz            -> Figure 9              (step 5)
    results_ext/hp_runs.csv             -> Figure 10             (step 5)
    results_ext/xai.csv                 -> Figure 11             (step 5)
    results_mw/mw_runs.csv              -> Figure 12             (step 4)
and writes  paper_figures/Figure_N.png (600 dpi) and Figure_N.pdf (vector).
A figure whose input is missing is skipped with a message; the others are still made.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_common as pc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=None)
ap.add_argument("--results", default=os.path.join(here, "results"))
ap.add_argument("--results_mw", default=os.path.join(here, "results_mw"))
ap.add_argument("--results_ext", default=os.path.join(here, "results_ext"))
ap.add_argument("--out", default=os.path.join(here, "paper_figures"))
ap.add_argument("--dpi", type=int, default=600)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------------------- style
INK, INK2, GRID, SURF = "#0b0b0b", "#3d3c39", "#dcdbd6", "#ffffff"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLS_COL = {"Air": C[0], "Void": C[1], "Corona": C[2], "Noise": "#7f7e79"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "font.weight": "bold",
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 16, "axes.labelweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13, "legend.title_fontsize": 13,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.linewidth": 1.4, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.6, "lines.markersize": 8, "legend.frameon": False,
    "figure.constrained_layout.use": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def save(fig, n):
    for a in fig.axes:
        bold_ticks(a)
    fig.savefig(os.path.join(args.out, f"Figure_{n}.png"), dpi=args.dpi)
    fig.savefig(os.path.join(args.out, f"Figure_{n}.pdf"))
    plt.close(fig)
    print(f"  Figure_{n} saved")


def panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=17, fontweight="bold",
            ha="right", va="bottom")


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan, np.nan
    from scipy import stats
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def need(path, n):
    if not os.path.exists(path):
        print(f"  Figure_{n} skipped (missing {path})")
        return False
    return True


# ============================================================================ Figure 1
def figure1():
    fig = plt.figure(figsize=(17, 7.6))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 50)
    ax.axis("off")
    W, H = 21.5, 15.5

    def box(x, y, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    linewidth=2.4, edgecolor=col, facecolor=col + "18"))
        ax.text(x + W / 2, y + H - 2.2, title, ha="center", va="top", fontsize=15.5, fontweight="bold", color=INK)
        ax.text(x + W / 2, y + H - 6.0, body, ha="center", va="top", fontsize=13, fontweight="bold",
                color=INK2, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=26,
                                     linewidth=2.6, color=INK2, shrinkA=0, shrinkB=0))

    xs = [2.5, 27.0, 51.5, 76.0]
    yt, yb = 33.0, 7.5
    top = [("1  Public dataset", "4 classes x 1000 waveforms\nanechoic chamber, 1 m\n125 samples, 500 kS/s", C[0]),
           ("2  Leakage-safe split", "blocked + purged 5-fold\ncross-validation x 3 seeds\ninner validation split", C[0]),
           ("3  Shortcut audit", "loudness, peak and arrival\nprobes; recording drift;\nacoustic budget vs distance", C[7]),
           ("4  Field simulator", "spreading + ISO 9613-1\nabsorption, reverberation,\ntrigger loss, sensor, noise", C[2])]
    bot = [("8  Deployment view", "detection range in metres,\nfalse alarms at a fixed rate,\n~24k-parameter models", C[1]),
           ("7  Statistics + XAI", "Wilcoxon-Holm, Friedman-\nNemenyi, ablation, gain /\nband / time occlusion", C[6]),
           ("6  Stress-test suite", "clean + 25 stress domains\nincl. out-of-distribution;\nmulti-window listening", C[3]),
           ("5  Train 5 model families", "small CNN, ResNet, Inception,\nspectrogram CNN, random forest;\nwith and without simulator", C[2])]
    for x, (t, b, c) in zip(xs, top):
        box(x, yt, t, b, c)
    for x, (t, b, c) in zip(xs, bot):
        box(x, yb, t, b, c)
    for i in range(3):
        arrow(xs[i] + W + 0.6, yt + H / 2, xs[i + 1] - 0.6, yt + H / 2)
        arrow(xs[i + 1] - 0.6, yb + H / 2, xs[i] + W + 0.6, yb + H / 2)
    arrow(xs[3] + W / 2, yt - 0.7, xs[3] + W / 2, yb + H + 0.7)
    fig.savefig(os.path.join(args.out, "Figure_1.png"), dpi=args.dpi)
    fig.savefig(os.path.join(args.out, "Figure_1.pdf"))
    plt.close(fig)
    print("  Figure_1 saved")


# ============================================================================ data-based figures
def load_raw():
    try:
        return pc.load_data(pc.find_data_dir(args.data_dir))
    except FileNotFoundError as e:
        print("  raw data not found:", e)
        return None


def figure2(X, y):
    t = np.arange(pc.N) / pc.FS * 1e6
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    for c, ax in enumerate(axs.flat):
        name = pc.CLASS_NAMES[c]
        m = y == c
        med = np.median(X[m], 0) * 1e3
        lo, hi = np.percentile(X[m] * 1e3, [25, 75], 0)
        ax.fill_between(t, lo, hi, color=CLS_COL[name], alpha=0.28, linewidth=0, label="inter-quartile range")
        ax.plot(t, med, color=CLS_COL[name], label="median waveform")
        ax.set_title({"Air": "Air-gap discharge (12 kV)", "Void": "Void discharge (25 kV)",
                      "Corona": "Corona at needle tip (11 kV)", "Noise": "No discharge (0 kV)"}[name])
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Pressure (mPa)")
        ax.set_xlim(0, 250)
        panel(ax, "abcd"[c])
    axs[0, 0].legend(loc="upper right")
    save(fig, 2)


def figure3(X, y):
    f = np.fft.rfftfreq(pc.NFFT, 1 / pc.FS)
    floor = float(np.median(np.sqrt((X[y == pc.NOISE_CLASS] ** 2).mean(1))))
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2))
    ax = axs[0]
    for c in range(pc.N_CLASSES):
        S = np.abs(np.fft.rfft(X[y == c], n=pc.NFFT)) ** 2
        ax.plot(f / 1e3, 10 * np.log10(S.mean(0) + 1e-30), color=CLS_COL[pc.CLASS_NAMES[c]], label=pc.CLASS_NAMES[c])
    ax.axvspan(25, 100, color=GRID, alpha=0.45, linewidth=0)
    ax.set_xlim(0, 150)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Mean power (dB)")
    ax.set_title("Average spectrum per class")
    ax.legend(loc="upper right")
    panel(ax, "a")
    ax = axs[1]
    for i, (T, rh) in enumerate([(0, 20), (20, 50), (35, 90)]):
        ax.plot(f / 1e3, pc.iso9613_alpha(f, T, rh), color=C[i + 3 if i else 6], label=f"{T} °C, {rh} % RH")
    ax.axvspan(25, 100, color=GRID, alpha=0.45, linewidth=0)
    ax.set_xlim(0, 150)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Absorption (dB/m)")
    ax.set_title("ISO 9613-1 air absorption")
    ax.legend(loc="upper left")
    panel(ax, "b")
    ax = axs[2]
    alpha = pc.iso9613_alpha(f, 20.0, 50.0)
    dists = np.array([1, 2, 3, 4, 6, 8, 10, 12, 15, 20])
    for c in range(3):
        Xc = np.fft.rfft(X[y == c], n=pc.NFFT)
        v = []
        for d in dists:
            z = np.fft.irfft(Xc * ((1 / d) * 10 ** (-alpha * (d - 1) / 20))[None, :], n=pc.NFFT)[:, :pc.N]
            v.append(np.median(20 * np.log10(np.abs(z).max(1) / floor)))
        ax.plot(dists, v, "o-", color=CLS_COL[pc.CLASS_NAMES[c]], label=pc.CLASS_NAMES[c])
    ax.axhline(0, color=INK2, linewidth=1.6, linestyle="--")
    ax.text(20, 2.5, "laboratory noise floor", ha="right", va="bottom", fontsize=13, fontweight="bold", color=INK2)
    ax.set_xlabel("Source-microphone distance (m)")
    ax.set_ylabel("Peak level above floor (dB)")
    ax.set_title("Acoustic budget vs distance")
    ax.legend(loc="upper right")
    panel(ax, "c")
    save(fig, 3)


def figure4(X, y, row):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score
    Fe = pc.features(X)
    sets = [("Arrival index only", [2]), ("Peak level only", [1]), ("RMS level only", [0]),
            ("Level + peak + arrival", pc.SHORTCUT_IDX),
            ("15 shape features (no level)", [i for i in range(len(pc.FEATURE_NAMES)) if i not in pc.SHORTCUT_IDX]),
            ("All 18 features", list(range(len(pc.FEATURE_NAMES))))]
    folds = pc.blocked_purged_folds(y, row, 5, 10)
    M, E = [], []
    for _, cols in sets:
        s = []
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            clf.fit(Fe[tr][:, cols], y[tr])
            s.append(f1_score(y[te], clf.predict(Fe[te][:, cols]), average="macro"))
        M.append(np.mean(s))
        E.append(np.std(s))
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    cols = [C[7] if "only" in n or "+" in n else C[0] for n, _ in sets]
    yy = np.arange(len(sets))
    ax.barh(yy, M, xerr=E, color=cols, height=0.62, error_kw=dict(lw=2, capsize=5, ecolor=INK2))
    for i, (m, e) in enumerate(zip(M, E)):
        ax.text(m + e + 0.015, i, f"{m:.3f}", va="center", ha="left", fontsize=14, fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _ in sets])
    ax.axvline(0.25, color=INK2, linestyle=":", linewidth=1.6)
    ax.text(0.26, -0.55, "chance", fontsize=13, fontweight="bold", color=INK2, va="center")
    ax.set_ylim(-0.9, len(sets) - 0.5)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Macro-F1 (logistic regression, blocked CV)")
    ax.grid(axis="y", visible=False)
    save(fig, 4)


# ============================================================================ result figures
def figure5():
    p = os.path.join(args.results_ext, "physics_centroid.csv")
    if not need(p, 5):
        return
    cc = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ["Air", "Void", "Corona"]:
        s = cc[cc.cls == name].sort_values("d")
        ax.plot(s.d, s.f_centroid_khz, "o-", color=CLS_COL[name], label=name)
    ax.set_xlabel("Simulated distance (m)")
    ax.set_ylabel("Spectral centroid (kHz)")
    ax.legend(loc="upper right")
    save(fig, 5)


def figure6():
    p = os.path.join(args.results, "runs.csv")
    if not need(p, 6):
        return
    df = pd.read_csv(p)
    show = [("ERM_raw", "Standard CNN"), ("GenericAug", "Generic augmentation"),
            ("RF_PIDR", "Random forest + simulator"), ("PIDR_full", "Small CNN + simulator")]
    show = [s for s in show if s[0] in df.config.unique()]
    dd = sorted(d for d in df.domain.unique() if d.startswith("dist_"))
    nd = sorted(d for d in df.domain.unique() if d.startswith("noise_"))
    fig, axs = plt.subplots(1, 2, figsize=(17, 6.4))
    for ax, doms, xs, xl, letter, ttl in [
            (axs[0], dd, [float(d[5:7]) for d in dd], "Source-microphone distance (m)", "a", "Distance"),
            (axs[1], nd, [float(d[6:8]) for d in nd], "Background noise level (dB re floor)", "b", "Industrial noise")]:
        for i, (cfg, lab) in enumerate(show):
            st = [ci(df[(df.config == cfg) & (df.domain == d)].macro_f1) for d in doms]
            m, lo, hi = map(np.array, zip(*st))
            ax.fill_between(xs, lo, hi, color=C[i], alpha=0.2, linewidth=0)
            ax.plot(xs, m, "o-", color=C[i], label=lab)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(xl)
        ax.set_ylabel("Macro-F1 (mean, 95% CI)")
        ax.set_title(ttl)
        panel(ax, letter)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09))
    save(fig, 6)


def figure7():
    p = os.path.join(args.results, "runs.csv")
    if not need(p, 7):
        return
    df = pd.read_csv(p)
    df["run"] = df.fold.astype(str) + "_" + df.seed.astype(str)
    target = [("abl_noShift", "untriggered", "Trigger loss, untriggered test"),
              ("abl_noReverb", "reverb_drr3", "Reverberation, reverberant test"),
              ("abl_noNorm", "sensor_mems", "Input standardization, sensor test"),
              ("abl_noSensor", "sensor_mems", "Sensor response, sensor test"),
              ("abl_noFieldNoise", "noise_15dB", "Field noise, 15 dB noise test"),
              ("abl_noFieldNoise", "impulsive", "Field noise, impulsive test"),
              ("abl_noDistance", "dist_08m", "Distance, 8 m test"),
              ("abl_noAbsorption", "dist_04m", "Absorption only, 4 m test"),
              ("abl_noAbsorption", "dist_08m", "Absorption only, 8 m test"),
              ("abl_noAux", "field_composite", "Distance head, field composite")]
    rows = []
    rng = np.random.default_rng(0)
    for cfg, dom, lab in target:
        W = df[df.domain == dom].pivot_table(index="run", columns="config", values="macro_f1")
        if cfg not in W or "PIDR_full" not in W:
            continue
        d = (W[cfg] - W["PIDR_full"]).dropna().values
        bs = rng.choice(d, (10000, len(d))).mean(1)
        rows.append((lab, d.mean(), *np.percentile(bs, [2.5, 97.5])))
    if not rows:
        print("  Figure_7 skipped (no ablation configs)")
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    yy = np.arange(len(rows))
    for i, (lab, m, lo, hi) in enumerate(rows):
        col = C[7] if hi < 0 else INK2
        ax.hlines(i, lo, hi, color=col, linewidth=3)
        ax.plot(m, i, "o", color=col, markersize=11)
    ax.axvline(0, color=INK, linewidth=1.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Change in macro-F1 when the component is removed")
    ax.grid(axis="y", visible=False)
    save(fig, 7)


def figure8():
    p = os.path.join(args.results_ext, "main_runs.csv")
    if not need(p, 8):
        return
    df = pd.read_csv(p)
    pairs = [("Tiny", "Small CNN"), ("ResNet", "ResNet-1D"), ("Incep", "Inception"),
             ("Spec", "Spectrogram CNN"), ("RF", "Random forest")]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), sharey=True)
    for ax, dom, letter, ttl in [(axs[0], "field_composite", "a", "Field composite"),
                                 (axs[1], "field_extreme_OOD", "b", "Out-of-distribution")]:
        for i, (a, lab) in enumerate(pairs):
            e = df[(df.config == f"{a}_ERM") & (df.domain == dom)].macro_f1.mean()
            q = df[(df.config == f"{a}_PIDR") & (df.domain == dom)].macro_f1.mean()
            if np.isnan(e) or np.isnan(q):
                continue
            ax.hlines(i, e, q, color=GRID, linewidth=5)
            ax.plot(e, i, "o", color=INK2, markersize=12, label="without simulator" if i == 0 else None)
            ax.plot(q, i, "o", color=C[0], markersize=12, label="with simulator" if i == 0 else None)
            ax.text(q + 0.015, i, f"{q - e:+.2f}", fontsize=13, fontweight="bold", color=C[0], va="center")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p_[1] for p_ in pairs])
        ax.set_xlabel("Macro-F1")
        ax.set_title(ttl)
        lo_, hi_ = ax.get_xlim()
        ax.set_xlim(max(0, lo_ - 0.02), hi_ + 0.1 * (hi_ - lo_) + 0.05)
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.grid(axis="y", visible=False)
        panel(ax, letter)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09))
    save(fig, 8)


def figure9():
    sdir = os.path.join(args.results_ext, "scores")
    if not need(sdir, 9):
        return
    from sklearn.metrics import roc_curve, roc_auc_score
    show = [("Tiny_ERM_raw", "Standard CNN"), ("Tiny_GenericAug", "Generic augmentation"),
            ("Spec_ERM", "Spectrogram CNN"), ("Tiny_PIDR", "Small CNN + simulator"),
            ("Spec_PIDR", "Spectrogram CNN + simulator")]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, dom, letter, ttl in [(axs[0], "field_composite", "a", "Field composite"),
                                 (axs[1], "field_extreme_OOD", "b", "Out-of-distribution")]:
        for i, (cfg, lab) in enumerate(show):
            P, Y = [], []
            for f in range(10):
                fn = os.path.join(sdir, f"{cfg}__f{f}__s0.npz")
                if os.path.isfile(fn):
                    z = np.load(fn)
                    P.append(z[dom])
                    Y.append(z["y"])
            if not P:
                continue
            P, Y = np.concatenate(P), np.concatenate(Y)
            yt, sc = (Y != pc.NOISE_CLASS).astype(int), 1 - P[:, pc.NOISE_CLASS]
            fpr, tpr, _ = roc_curve(yt, sc)
            ax.plot(fpr, tpr, color=C[i], label=f"{lab} ({roc_auc_score(yt, sc):.2f})")
        ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1.6)
        ax.axvline(0.05, color=C[7], linestyle="--", linewidth=1.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("False-alarm rate")
        ax.set_ylabel("Discharge detection rate")
        ax.set_title(ttl)
        ax.legend(loc="lower right", title="model (AUROC)")
        panel(ax, letter)
    save(fig, 9)


def figure10():
    p = os.path.join(args.results_ext, "hp_runs.csv")
    if not need(p, 10):
        return
    hp = pd.read_csv(p)
    g = hp.groupby(["config", "lr", "epochs"])["test_field_composite"].mean().reset_index()
    names = {"Tiny_ERM_raw": "Standard CNN", "Tiny_GenericAug": "Generic augmentation",
             "Tiny_PIDR": "Small CNN + simulator"}
    cfgs = [c for c in names if c in g.config.unique()]
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, axs = plt.subplots(1, len(cfgs), figsize=(5.6 * len(cfgs), 5.6))
    axs = np.atleast_1d(axs)
    im = None
    for ax, c, letter in zip(axs, cfgs, "abc"):
        M = g[g.config == c].pivot(index="lr", columns="epochs", values="test_field_composite")
        im = ax.imshow(M.values, cmap=cmap, vmin=0.4, vmax=0.85, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([str(int(v)) for v in M.columns])
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([f"{v:g}" for v in M.index])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Learning rate")
        ax.set_title(names[c])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=15, fontweight="bold",
                        color="#ffffff" if v > 0.66 else INK)
        panel(ax, letter)
    cb = fig.colorbar(im, ax=list(axs), fraction=0.03, pad=0.02)
    cb.set_label("Macro-F1, field composite", fontweight="bold")
    save(fig, 10)


def figure11():
    p = os.path.join(args.results_ext, "xai.csv")
    if not need(p, 11):
        return
    xa = pd.read_csv(p)
    names = {"Tiny_ERM_raw": "Standard CNN", "RF_ERM": "Random forest", "Tiny_GenericAug": "Generic augmentation",
             "Tiny_PIDR": "Small CNN + simulator"}
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.4))
    specs = [("gain", "macro_f1", "Input gain (dB)", "Macro-F1 (clean test)", "Loudness test", "a"),
             ("band_field", "drop", "Removed band centre (kHz)", "Macro-F1 drop (field test)", "Frequency reliance", "b"),
             ("time_clean", "drop", "Occluded window centre (µs)", "Macro-F1 drop (clean test)", "Timing reliance", "c")]
    for ax, (test, col, xl, yl, ttl, letter) in zip(axs, specs):
        for i, (m, lab) in enumerate(names.items()):
            s = xa[(xa["test"] == test) & (xa["model"] == m)].sort_values("x")
            if len(s):
                ax.plot(s["x"], s[col], "o-", color=C[i], label=lab)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        panel(ax, letter)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    save(fig, 11)


def figure12():
    p = os.path.join(args.results_mw, "mw_runs.csv")
    if not need(p, 12):
        return
    df = pd.read_csv(p)
    df = df[(df["agg"] == "logp") & (df["domain"] == "field_composite")]
    show = [("ERM_raw", "Standard CNN"), ("GenericAug", "Generic augmentation"),
            ("RF_PIDR", "Random forest + simulator"), ("PIDR_full", "Small CNN + simulator")]
    Ks = sorted(df.K.unique())
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.4))
    for ax, col, yl, ttl, letter in [(axs[0], "macro_f1", "Macro-F1", "Classification", "a"),
                                     (axs[1], "rec_Corona", "Corona recall", "Weakest discharge type", "b"),
                                     (axs[2], "false_alarm", "False-alarm rate", "Alarms on discharge-free bags", "c")]:
        for i, (cfg, lab) in enumerate(show):
            st = [ci(df[(df.config == cfg) & (df.K == k)][col]) for k in Ks]
            if np.all(np.isnan([s[0] for s in st])):
                continue
            m, lo, hi = map(np.array, zip(*st))
            ax.fill_between(Ks, lo, hi, color=C[i], alpha=0.2, linewidth=0)
            ax.plot(Ks, m, "o-", color=C[i], label=lab)
        ax.set_xscale("log")
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(int(k)) for k in Ks])
        ax.minorticks_off()
        ax.set_xlabel("Windows fused per decision")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        panel(ax, letter)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    save(fig, 12)


if __name__ == "__main__":
    print(f"writing figures to {args.out}")
    figure1()
    raw = load_raw()
    if raw is not None:
        X, y, row = raw
        figure2(X, y)
        figure3(X, y)
        figure4(X, y, row)
    for fn in [figure5, figure6, figure7, figure8, figure9, figure10, figure11, figure12]:
        try:
            fn()
        except Exception as e:                       # one broken input must not stop the rest
            print(f"  {fn.__name__} failed: {e}")
    print("done")
