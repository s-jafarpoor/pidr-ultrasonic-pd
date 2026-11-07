"""
03_statistics.py  -  Step 3: statistical analysis, ablation study, figures, summary
=================================================================================
  T1  main results: mean, sd, 95% CI (t) per method x key domain
  T2  primary-endpoint tests  PIDR_full vs every baseline on 'field_composite' (+ OOD):
        Wilcoxon signed-rank (paired over fold x seed), Holm-corrected,
        rank-biserial effect size, Cohen's dz, paired-bootstrap 95% CI of the difference,
        Nadeau-Bengio corrected resampled t-test (accounts for overlapping CV training sets)
  T3  Friedman test + Nemenyi critical difference (CD diagram) across all methods
  T4  McNemar exact test on pooled per-waveform predictions (seed 0, all folds)
  T5  ablation study: delta vs PIDR_full per domain with CI + Holm-corrected Wilcoxon,
        each component checked on the domain it is supposed to fix
  T6  operational KPIs: detection range [m] at macro-F1 >= 0.9 / 0.8, false-alarm rate,
        calibration (ECE), monaural distance-estimation error, edge-model footprint
  Figures: robustness vs distance, vs noise level, method x domain heat-map, CD diagram,
           ablation forest plot, confusion matrix.
Usage:   python 03_statistics.py            (reads results/ by default)
Output:  results/stats/ ... and results/stats/SUMMARY_FOR_CLAUDE.txt  -> send that file
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_common as pc

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=None)
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
R = args.results or os.path.join(here, "results")
S = os.path.join(R, "stats")
os.makedirs(S, exist_ok=True)

df = pd.read_csv(os.path.join(R, "runs.csv"))
df["run"] = df.fold.astype(str) + "_" + df.seed.astype(str)
PROP = "PIDR_full"
PRIMARY, OOD = "field_composite", "field_extreme_OOD"
MAIN = [c for c in ["Probe_LR", "RF_clean", "RF_PIDR", "ERM_raw", "ERM_norm", "GenericAug", PROP]
        if c in df.config.unique()]
ABL = [c for c in df.config.unique() if c.startswith("abl_")]
KEY = ["clean", "untriggered", "reverb_drr3", "sensor_mems", "impulsive", "dist_04m", "dist_08m",
       "noise_15dB", PRIMARY, OOD]
KEY = [k for k in KEY if k in df.domain.unique()]
ABL_TARGET = {"abl_noAbsorption": ["dist_04m", "dist_08m"], "abl_noDistance": ["dist_04m", "dist_08m"],
              "abl_noReverb": ["reverb_drr3"], "abl_noShift": ["untriggered"],
              "abl_noSensor": ["sensor_mems"], "abl_noFieldNoise": ["noise_15dB", "impulsive"],
              "abl_noAux": [PRIMARY], "abl_noNorm": ["sensor_mems"]}
out_lines = []


def say(s=""):
    print(s)
    out_lines.append(str(s))


def wide(domain, metric="macro_f1", configs=None):
    d = df[df.domain == domain].pivot_table(index="run", columns="config", values=metric)
    if configs is not None:
        d = d[[c for c in configs if c in d.columns]]
    return d.dropna()


def ci_t(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, x.std(ddof=1), m - h, m + h


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = rng.choice(d, (B, len(d)), replace=True).mean(1)
    return np.percentile(bs, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def wilcoxon_rb(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0, 0.0
    try:
        p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
    except ValueError:
        p = 1.0
    r = stats.rankdata(np.abs(d))
    rb = (r[d > 0].sum() - r[d < 0].sum()) / r.sum()
    return float(p), float(rb)


def nadeau_bengio(d, ratio):
    d = np.asarray(d, float)
    J = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return np.nan
    t = d.mean() / np.sqrt((1.0 / J + ratio) * v)
    return float(2 * stats.t.sf(abs(t), J - 1))


# test/train size ratio for the corrected t-test
yt = np.load(os.path.join(R, "preds", "ytrue_f0.npy"))
n_test = len(yt)
n_train = 4000 - n_test - 2 * 10 * pc.N_CLASSES
RATIO = n_test / n_train
n_runs = df[df.config == PROP].run.nunique()
say(f"runs per config = {n_runs}   (test/train ratio for Nadeau-Bengio = {RATIO:.3f})")

# ============================================================ T1 main table
say("\n=== T1. Macro-F1 mean [95% CI] per method x domain ===")
rows = []
for c in MAIN + ABL:
    r = {"config": c}
    for k in KEY:
        x = df[(df.config == c) & (df.domain == k)].macro_f1.values
        if len(x):
            m, sd, lo, hi = ci_t(x)
            r[k] = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
    rows.append(r)
T1 = pd.DataFrame(rows).set_index("config")
T1.to_csv(os.path.join(S, "T1_main_table.csv"))
say(T1[[k for k in KEY]].to_string())

# ============================================================ T2 primary tests
say(f"\n=== T2. {PROP} vs baselines (paired over {n_runs} runs) ===")
t2 = []
for dom in [PRIMARY, OOD]:
    W = wide(dom, configs=MAIN)
    comps = [c for c in W.columns if c != PROP]
    ps = []
    for c in comps:
        d = W[PROP] - W[c]
        p, rb = wilcoxon_rb(W[PROP], W[c])
        lo, hi = boot_ci(d)
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        t2.append(dict(domain=dom, vs=c, mean_prop=W[PROP].mean(), mean_base=W[c].mean(),
                       delta=d.mean(), boot_lo=lo, boot_hi=hi, wins=int((d > 0).sum()),
                       n=len(d), p_wilcoxon=p, rank_biserial=rb, cohen_dz=dz,
                       p_nadeau_bengio=nadeau_bengio(d, RATIO)))
        ps.append(p)
    adj = holm(ps)
    for i, c in enumerate(comps):
        t2[-len(comps) + i]["p_holm"] = adj[i]
T2 = pd.DataFrame(t2)
T2.to_csv(os.path.join(S, "T2_primary_tests.csv"), index=False)
say(T2.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ============================================================ T3 Friedman + Nemenyi
say("\n=== T3. Friedman + Nemenyi (higher macro-F1 = better rank) ===")
plt = pc.plot_style()
for dom in [PRIMARY, OOD, "clean"]:
    W = wide(dom, configs=MAIN)              # methods only; ablations are tested in T5
    if W.shape[1] < 3 or W.shape[0] < 2:
        continue
    chi, p = stats.friedmanchisquare(*[W[c].values for c in W.columns])
    ranks = W.rank(axis=1, ascending=False).mean().sort_values()
    k, Nn = W.shape[1], W.shape[0]
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * Nn))
    say(f"[{dom}] Friedman chi2={chi:.2f}, p={p:.3g}, k={k}, N={Nn}, Nemenyi CD={cd:.3f}")
    say("   avg ranks: " + ", ".join(f"{c}={v:.2f}" for c, v in ranks.items()))
    # CD diagram
    fig, ax = plt.subplots(figsize=(9, 0.45 * k + 1.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1, k)
    ax.invert_xaxis()
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.set_xlabel("average rank (1 = best)")
    ax.xaxis.set_label_position("top")
    for i, (c, v) in enumerate(ranks.items()):
        col = pc.PALETTE[0] if c == PROP else pc.INK2
        ax.plot([v, v], [k - 1 - i, k - 0.6], color=pc.GRID, linewidth=1)
        ax.plot(v, k - 1 - i, "o", color=col, markersize=7)
        ax.text(v - 0.12, k - 1 - i, c, va="center", ha="left", color=pc.INK, fontsize=9)
    best = ranks.iloc[0]
    ax.plot([best, best + cd], [-0.6, -0.6], color=pc.INK, linewidth=2)
    ax.text(best + cd / 2, -0.95, f"CD = {cd:.2f}", ha="center", va="top", fontsize=9)
    ax.set_title(f"Critical-difference diagram — {dom}", pad=28)
    fig.savefig(os.path.join(S, f"fig_cd_{dom}.png"))
    plt.close(fig)

# ============================================================ T4 McNemar
say("\n=== T4. McNemar exact test, pooled predictions (seed 0, all folds) ===")
folds = sorted(df[df.config == PROP].fold.unique())


def pooled(c, dom):
    P, Y = [], []
    for f in folds:
        fn = os.path.join(R, "preds", f"{c}__f{f}__s0.npz")
        if not os.path.isfile(fn):
            return None, None
        P.append(np.load(fn)[dom])
        Y.append(np.load(os.path.join(R, "preds", f"ytrue_f{f}.npy")))
    return np.concatenate(P), np.concatenate(Y)


t4 = []
for dom in [PRIMARY, OOD]:
    pa, y = pooled(PROP, dom)
    if pa is None:
        continue
    for c in MAIN:
        if c == PROP:
            continue
        pb, _ = pooled(c, dom)
        if pb is None:
            continue
        a, b = pa == y, pb == y
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        t4.append(dict(domain=dom, vs=c, prop_right_base_wrong=n01, prop_wrong_base_right=n10, p_exact=p))
T4 = pd.DataFrame(t4)
if len(T4):
    T4["p_holm"] = np.concatenate([holm(g.p_exact.values) for _, g in T4.groupby("domain", sort=False)])
    T4.to_csv(os.path.join(S, "T4_mcnemar.csv"), index=False)
    say(T4.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ============================================================ T5 ablation
say(f"\n=== T5. Ablation: delta macro-F1 = (ablated - {PROP}); negative = component helps ===")
t5 = []
doms5 = [k for k in KEY if k != "clean"] + ["clean"]
for dom in doms5:
    W = wide(dom, configs=[PROP] + ABL)
    ps, idx = [], []
    for c in ABL:
        if c not in W.columns:
            continue
        d = W[c] - W[PROP]
        p, rb = wilcoxon_rb(W[c], W[PROP])
        lo, hi = boot_ci(d)
        t5.append(dict(ablation=c, domain=dom, delta=d.mean(), lo=lo, hi=hi, p=p, rank_biserial=rb,
                       target=dom in ABL_TARGET.get(c, [])))
        ps.append(p)
        idx.append(len(t5) - 1)
    for j, a in zip(idx, holm(ps)):
        t5[j]["p_holm"] = a
T5 = pd.DataFrame(t5)
T5.to_csv(os.path.join(S, "T5_ablation_long.csv"), index=False)
if len(T5):
    piv = T5.assign(cell=T5.apply(lambda r: f"{r.delta:+.3f}{'*' if r.p_holm < 0.05 else ' '}"
                                  f"{'T' if r.target else ' '}", axis=1)
                    ).pivot(index="ablation", columns="domain", values="cell")
    piv = piv[[d for d in doms5 if d in piv.columns]]
    piv.to_csv(os.path.join(S, "T5_ablation_table.csv"))
    say("(*) Holm-adjusted p<0.05   (T) = domain this component is designed for")
    say(piv.to_string())
    # forest plot on target domains
    tg = T5[T5.target].sort_values("delta")
    if len(tg):
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(tg) + 1.2))
        yy = np.arange(len(tg))
        ax.hlines(yy, tg.lo, tg.hi, color=pc.INK2, linewidth=2)
        ax.plot(tg.delta, yy, "o", color=pc.PALETTE[1], markersize=8)
        ax.axvline(0, color=pc.INK, linewidth=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{a.replace('abl_no', 'w/o ')} @ {d}" for a, d in zip(tg.ablation, tg.domain)])
        ax.set_xlabel(f"Δ macro-F1 vs {PROP} (95% bootstrap CI)")
        ax.set_title("Leave-one-component-out ablation, on each component's target domain")
        fig.savefig(os.path.join(S, "fig_ablation_forest.png"))
        plt.close(fig)

# ============================================================ T6 KPIs
say("\n=== T6. Operational KPIs ===")
dist_doms = sorted([d for d in df.domain.unique() if d.startswith("dist_")])
dists = np.array([float(d[5:7]) for d in dist_doms])
noise_doms = sorted([d for d in df.domain.unique() if d.startswith("noise_")])
levels = np.array([float(d[6:8]) for d in noise_doms])


def crossing(xs, ys, thr):
    """largest x before the curve first drops below thr (linear interpolation)."""
    if ys[0] < thr:
        return 0.0
    for i in range(1, len(xs)):
        if ys[i] < thr:
            return xs[i - 1] + (ys[i - 1] - thr) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1])
    return xs[-1]


t6 = []
for c in MAIN + ABL:
    sub = df[df.config == c]
    rngs9, rngs8, ntol = [], [], []
    for run, g in sub.groupby("run"):
        cur = g.set_index("domain").macro_f1
        if all(d in cur for d in dist_doms):
            yv = cur[dist_doms].values
            rngs9.append(crossing(dists, yv, 0.9))
            rngs8.append(crossing(dists, yv, 0.8))
        if all(d in cur for d in noise_doms):
            ntol.append(crossing(levels, cur[noise_doms].values, 0.8))
    p_ = sub[sub.domain == PRIMARY]
    c_ = sub[sub.domain == "clean"]
    row = dict(config=c)
    for nm, v in [("range_m_F1>=0.9", rngs9), ("range_m_F1>=0.8", rngs8), ("noise_tol_dB_F1>=0.8", ntol)]:
        m, sd, lo, hi = ci_t(v) if len(v) else (np.nan,) * 4
        row[nm] = f"{m:.2f} [{lo:.2f},{hi:.2f}]"
        row[nm + "_raw"] = v
    row["false_alarm_field"] = f"{p_.det_false_alarm.mean():.3f}"
    row["det_recall_field"] = f"{p_.det_recall.mean():.3f}"
    row["ECE_clean"] = f"{c_.ece.mean():.3f}"
    row["ECE_field"] = f"{p_.ece.mean():.3f}"
    t6.append(row)
T6 = pd.DataFrame(t6).set_index("config")
show = [c for c in T6.columns if not c.endswith("_raw")]
T6[show].to_csv(os.path.join(S, "T6_kpis.csv"))
say(T6[show].to_string())

# detection-range significance vs each baseline
say("\nDetection range (F1>=0.9) - PIDR_full vs baselines, Wilcoxon (Holm):")
base = [c for c in MAIN if c != PROP]
ps, recs = [], []
for c in base:
    a, b = np.array(T6.loc[PROP, "range_m_F1>=0.9_raw"]), np.array(T6.loc[c, "range_m_F1>=0.9_raw"])
    if len(a) == len(b) and len(a) > 1:
        p, rb = wilcoxon_rb(a, b)
        ps.append(p)
        recs.append((c, (a - b).mean(), rb))
for (c, dm, rb), pa in zip(recs, holm(ps)):
    say(f"   vs {c:12s}  delta_range = {dm:+.2f} m   rank-biserial = {rb:+.2f}   p_holm = {pa:.3g}")

# distance estimation
dd = df[(df.config == PROP) & df.domain.isin(dist_doms)].groupby("domain")[
    ["dist_med_est_m", "dist_med_rel_err"]].mean()
if dd.notna().any().any():
    say("\nMonaural distance estimation by PIDR_full (PD classes; median over waveforms, mean over runs):")
    say(dd.round(3).to_string())

# ============================================================ T7 per-class detection range
say("\n=== T7. Per-class recall vs distance (mean over runs) and per-class range at recall>=0.8 ===")


def class_recall(cm_str):
    m = np.array(list(map(int, cm_str.split()))).reshape(pc.N_CLASSES, pc.N_CLASSES)
    return np.diag(m) / np.maximum(m.sum(1), 1)


t7 = []
for c in [x for x in ["RF_PIDR", "ERM_raw", "GenericAug", PROP] if x in df.config.unique()]:
    sub = df[(df.config == c) & df.domain.isin(dist_doms)].copy()
    rec = np.stack(sub.cm.map(class_recall).values)
    for k in range(pc.N_CLASSES):
        sub[pc.CLASS_NAMES[k]] = rec[:, k]
    mean_rec = sub.groupby("domain")[pc.CLASS_NAMES].mean().reindex(dist_doms)
    say(f"[{c}]")
    say(mean_rec.round(3).to_string())
    for k in range(3):
        rr = []
        for run, g in sub.groupby("run"):
            yv = g.set_index("domain").reindex(dist_doms)[pc.CLASS_NAMES[k]].values
            rr.append(crossing(dists, yv, 0.8))
        m, sd, lo, hi = ci_t(rr)
        t7.append(dict(config=c, cls=pc.CLASS_NAMES[k], range_m=m, lo=lo, hi=hi))
    if c == PROP:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for k in range(pc.N_CLASSES):
            ax.plot(dists, mean_rec[pc.CLASS_NAMES[k]].values, "o-", color=pc.CLASS_COLORS[k],
                    markersize=6, label=pc.CLASS_NAMES[k] + (" (specificity)" if k == pc.NOISE_CLASS else ""))
        ax.axhline(0.8, color=pc.INK2, linewidth=1, linestyle=":")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("source-microphone distance [m]")
        ax.set_ylabel("per-class recall")
        ax.set_title(f"{PROP}: how far each discharge type can be recognised")
        ax.legend(loc="lower left")
        fig.savefig(os.path.join(S, "fig_per_class_range.png"))
        plt.close(fig)
T7 = pd.DataFrame(t7)
T7.to_csv(os.path.join(S, "T7_per_class_range.csv"), index=False)
say("\nPer-class detection range [m] (recall >= 0.8), mean [95% CI]:")
say(T7.pivot(index="config", columns="cls", values="range_m").round(2).to_string())

eff_fn = os.path.join(R, "efficiency.json")
if os.path.isfile(eff_fn):
    say("\nEdge footprint: " + json.dumps(json.load(open(eff_fn))))

# ============================================================ figures
FIG4 = [c for c in ["ERM_raw", "GenericAug", "RF_PIDR", PROP] if c in df.config.unique()]


def curve(domains, xs, xlabel, fname, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, c in enumerate(FIG4):
        M, LO, HI = [], [], []
        for d in domains:
            m, sd, lo, hi = ci_t(df[(df.config == c) & (df.domain == d)].macro_f1.values)
            M.append(m)
            LO.append(lo)
            HI.append(hi)
        ax.fill_between(xs, LO, HI, color=pc.PALETTE[i], alpha=0.18, linewidth=0)
        ax.plot(xs, M, "o-", color=pc.PALETTE[i], markersize=6, label=c)
    ax.axhline(0.10, color=pc.INK2, linewidth=1, linestyle=":")
    ax.text(xs[0], 0.12, "everything → 'Noise' (0.10)", fontsize=8, color=pc.INK2, ha="left")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("macro-F1 (mean, 95% CI)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(S, fname))
    plt.close(fig)


if dist_doms:
    curve(dist_doms, dists, "source-microphone distance [m]", "fig_robustness_distance.png",
          "Robustness to distance (spherical spreading + ISO 9613-1 absorption)")
if noise_doms:
    curve(noise_doms, levels, "industrial noise level [dB re lab floor]", "fig_robustness_noise.png",
          "Robustness to absolute industrial noise level")

hm_doms = ["clean", "lab_floor_x2", "untriggered", "reverb_drr3", "sensor_mems", "impulsive",
           "dist_02m", "dist_04m", "dist_08m", "noise_10dB", "noise_20dB", PRIMARY, OOD]
hm_doms = [d for d in hm_doms if d in df.domain.unique()]
H = df[df.domain.isin(hm_doms)].pivot_table(index="config", columns="domain", values="macro_f1")
H = H.reindex(index=[c for c in MAIN + ABL if c in H.index], columns=hm_doms)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
fig, ax = plt.subplots(figsize=(1.0 * len(hm_doms) + 2.5, 0.42 * len(H) + 1.5))
im = ax.imshow(H.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.grid(False)
ax.set_xticks(range(len(hm_doms)))
ax.set_xticklabels(hm_doms, rotation=45, ha="right")
ax.set_yticks(range(len(H)))
ax.set_yticklabels(H.index)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="#ffffff" if v > 0.6 else pc.INK)
n_main = sum(c in H.index for c in MAIN)
if 0 < n_main < len(H):
    ax.axhline(n_main - 0.5, color=pc.SURFACE, linewidth=3)      # methods | ablations
fig.colorbar(im, ax=ax, fraction=0.025, label="macro-F1")
ax.set_title("Mean macro-F1 per method and test domain")
fig.savefig(os.path.join(S, "fig_heatmap.png"))
plt.close(fig)

cm = np.zeros((4, 4))
for s in df[(df.config == PROP) & (df.domain == PRIMARY)].cm:
    cm += np.array(list(map(int, s.split()))).reshape(4, 4)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
ax.grid(False)
ax.set_xticks(range(4))
ax.set_xticklabels(pc.CLASS_NAMES)
ax.set_yticks(range(4))
ax.set_yticklabels(pc.CLASS_NAMES)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                color="#ffffff" if cmn[i, j] > 0.6 else pc.INK)
ax.set_title(f"{PROP} on {PRIMARY} (row-normalised)")
fig.savefig(os.path.join(S, "fig_confusion.png"))
plt.close(fig)
say("\nConfusion (PIDR_full, field_composite, row-normalised):")
say(pd.DataFrame(cmn, index=pc.CLASS_NAMES, columns=pc.CLASS_NAMES).round(3).to_string())

with open(os.path.join(S, "SUMMARY_FOR_CLAUDE.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines))
print(f"\nAll tables and figures in {S}  ->  send SUMMARY_FOR_CLAUDE.txt")
