"""
04_multiwindow.py  -  Step 4: continuous-listening decision (multi-window aggregation)
====================================================================================
A real sensor does not decide from ONE 250-us window: it listens continuously, and a
discharge source emits a train of pulses. Here each test "bag" = K windows from the SAME
source under the SAME (randomly drawn) field condition (distance, T, RH, reverb, sensor,
background-noise level); trigger position, noise realisation and clicks vary per window.
The decision is the argmax of the mean log-probability over the K windows (naive-Bayes
fusion; majority vote reported as a robustness check). No retraining of the idea is needed:
models are trained exactly as in step 2 (same seeds -> same augmentation stream).

  K in {1, 2, 5, 10, 20, 50}     configs: ERM_raw, GenericAug, RF_PIDR, PIDR_full
  domains: field_composite, field_extreme_OOD, distance sweep
  stats: mean [95% CI], Wilcoxon (Holm) PIDR_full vs each baseline at every K,
         false-alarm rate vs K, per-class detection range vs K

Usage:  python 04_multiwindow.py --quick      (smoke test, ~2 min)
        python 04_multiwindow.py              (full, resumable; ~same time as 4 configs of step 2)
Output: results_mw/mw_runs.csv, figures, MULTIWINDOW_SUMMARY.txt  -> send the summary
"""
import os
import sys
import copy
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_common as pc

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--lam", type=float, default=0.3)
ap.add_argument("--purge", type=int, default=10)
ap.add_argument("--rf_copies", type=int, default=5)
ap.add_argument("--bags", type=int, default=100, help="bags per class per fold")
ap.add_argument("--device", default="auto")
ap.add_argument("--stats_only", action="store_true")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
n_folds_run = args.folds
if args.quick:
    n_folds_run, args.seeds, args.epochs, args.rf_copies, args.bags = 1, 1, 4, 1, 15

here = os.path.dirname(os.path.abspath(__file__))
out = args.out or os.path.join(here, "results_mw_quick" if args.quick else "results_mw")
os.makedirs(out, exist_ok=True)
device = pc.get_device(args.device)
KS = [1, 2, 5, 10, 20, 50]
KMAX = max(KS)
T = pc.TRAIN_SPEC
CONFIGS = {
    "ERM_raw":    dict(kind="cnn", spec=None, norm=False, aux=False),
    "GenericAug": dict(kind="cnn", spec=pc.GENERIC_SPEC, norm=True, aux=False),
    "RF_PIDR":    dict(kind="rf", spec=T),
    "PIDR_full":  dict(kind="cnn", spec=T, norm=True, aux=True),
}
PROP = "PIDR_full"
DOMS_ALL = pc.eval_domains()
BAG_DOMAINS = ["field_composite", "field_extreme_OOD"] + sorted(d for d in DOMS_ALL if d.startswith("dist_"))
csv_path = os.path.join(out, "mw_runs.csv")


# ------------------------------------------------------------------ bag construction
def point_spec(spec, rng):
    """Collapse every environment range to ONE value drawn for the whole bag."""
    s = copy.deepcopy(spec)
    for key in ["prop", "reverb", "sensor", "noise"]:
        if key in s and s[key].get("on"):
            for k, v in list(s[key].items()):
                if isinstance(v, tuple) and k != "imp":          # click amplitudes stay per-click
                    lo, hi = v
                    if key == "prop" and k == "d":
                        val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
                    else:
                        val = rng.uniform(lo, hi)
                    s[key][k] = (val, val)
    return s


def build_bags(Xte, yte, sim, fold):
    xt = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    out_ = {}
    for dname in BAG_DOMAINS:
        spec = DOMS_ALL[dname]
        rng = np.random.default_rng(pc.seed_of("bags", fold, dname))
        g = torch.Generator(device=device)
        g.manual_seed(pc.seed_of("bagsim", fold, dname))
        Z, L = [], []
        for c in range(pc.N_CLASSES):
            rows = np.where(yte == c)[0]
            for _ in range(args.bags):
                idx = rng.choice(rows, KMAX, replace=len(rows) < KMAX)
                z, _, _ = sim(xt[idx], point_spec(spec, rng), g)
                Z.append(z)
                L.append(c)
        out_[dname] = (torch.stack(Z), np.array(L))            # (nb, KMAX, N), (nb,)
    return out_


# ------------------------------------------------------------------ training (identical to step 2)
def train_cnn(cfg, Xtr, ytr, sim, fold, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = pc.TinyPDNet(args.width, cfg["norm"], float(np.std(Xtr))).to(device)
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    n = len(xt)
    spe = int(np.ceil(n / args.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * spe)
    g = torch.Generator(device=device)
    g.manual_seed(pc.seed_of("train", fold, seed))
    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(spe):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            xb, yb = xt[idx], yt[idx]
            with torch.no_grad():
                xb, ld, pm = sim(xb, cfg["spec"], g)
            logits, dpred = model(xb)
            loss = F.cross_entropy(logits, yb)
            if cfg["aux"]:
                m = pm & (yb != pc.NOISE_CLASS)
                if m.any():
                    loss = loss + args.lam * F.smooth_l1_loss(dpred[m], ld[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    model.eval()
    return model


def train_rf(cfg, Xtr, ytr, sim, fold, seed):
    from sklearn.ensemble import RandomForestClassifier
    g = torch.Generator(device=device)
    g.manual_seed(pc.seed_of("rfaug", fold, seed))
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    parts, ys = [Xtr], [ytr]
    for _ in range(args.rf_copies):
        z, _, _ = sim(xt, cfg["spec"], g)
        parts.append(z.cpu().numpy())
        ys.append(ytr)
    clf = RandomForestClassifier(300, n_jobs=-1, random_state=seed, min_samples_leaf=2)
    clf.fit(pc.features(np.concatenate(parts)), np.concatenate(ys))
    return clf


@torch.no_grad()
def window_probs(model_or_clf, kind, Zflat, feat=None):
    if kind == "cnn":
        P = []
        for i in range(0, len(Zflat), 4096):
            P.append(torch.softmax(model_or_clf(Zflat[i:i + 4096])[0], 1).cpu())
        return torch.cat(P).numpy()
    return model_or_clf.predict_proba(feat)


def bag_metrics(prob, labels):
    """prob: (nb, KMAX, C). Returns list of dict rows over K and aggregation rule."""
    from sklearn.metrics import f1_score
    rows = []
    lp = np.log(np.clip(prob, 1e-6, 1.0))
    for K in KS:
        for agg in ["logp", "vote"]:
            if agg == "logp":
                pred = lp[:, :K].mean(1).argmax(1)
            else:
                v = np.apply_along_axis(lambda r: np.bincount(r, minlength=pc.N_CLASSES), 1,
                                        prob[:, :K].argmax(2))
                pred = v.argmax(1)
            r = dict(K=K, agg=agg, macro_f1=f1_score(labels, pred, average="macro",
                                                    labels=list(range(pc.N_CLASSES)), zero_division=0))
            for c in range(pc.N_CLASSES):
                r[f"rec_{pc.CLASS_NAMES[c]}"] = float((pred[labels == c] == c).mean())
            nz = labels == pc.NOISE_CLASS
            r["false_alarm"] = float((pred[nz] != pc.NOISE_CLASS).mean())
            r["det_recall"] = float((pred[~nz] != pc.NOISE_CLASS).mean())
            rows.append(r)
    return rows


# ------------------------------------------------------------------ experiment
if not args.stats_only:
    X, y, row = pc.load_data(pc.find_data_dir(args.data_dir))
    folds = pc.blocked_purged_folds(y, row, args.folds, args.purge)
    done = set()
    if os.path.isfile(csv_path):
        d0 = pd.read_csv(csv_path)
        done = set(zip(d0.config, d0.fold, d0.seed))
    total = len(CONFIGS) * n_folds_run * args.seeds
    print(f"device={device}  {total} runs ({len(done)} done), K={KS}, bags/class/fold={args.bags}", flush=True)
    t0all, n_new = time.time(), 0
    for fold in range(n_folds_run):
        todo = [(c, s) for c in CONFIGS for s in range(args.seeds) if (c, fold, s) not in done]
        if not todo:
            continue
        tr, te = folds[fold]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        bank_tr, bank_te = Xtr[ytr == pc.NOISE_CLASS], Xte[yte == pc.NOISE_CLASS]
        floor = float(np.median(np.sqrt((bank_tr ** 2).mean(1))))
        sim_tr = pc.FieldSimulator(bank_tr, floor, device)
        sim_te = pc.FieldSimulator(bank_te, floor, device)
        print(f"fold {fold}: building bags ...", flush=True)
        bags = build_bags(Xte, yte, sim_te, fold)
        feats = {}
        for cname, seed in todo:
            t0 = time.time()
            cfg = CONFIGS[cname]
            if cfg["kind"] == "cnn":
                model = train_cnn(cfg, Xtr, ytr, sim_tr, fold, seed)
            else:
                model = train_rf(cfg, Xtr, ytr, sim_tr, fold, seed)
            rows = []
            for dname, (Z, L) in bags.items():
                nb = Z.shape[0]
                Zf = Z.reshape(nb * KMAX, pc.N)
                if cfg["kind"] == "rf" and dname not in feats:
                    feats[dname] = pc.features(Zf.cpu().numpy())
                P = window_probs(model, cfg["kind"], Zf, feats.get(dname)).reshape(nb, KMAX, -1)
                for r in bag_metrics(P, L):
                    rows.append(dict(config=cname, fold=fold, seed=seed, domain=dname, **r))
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.isfile(csv_path), index=False)
            n_new += 1
            fc = [r for r in rows if r["domain"] == "field_composite" and r["agg"] == "logp"]
            eta = (time.time() - t0all) / n_new * (total - len(done) - n_new) / 60
            print(f"  {cname:11s} fold={fold} seed={seed}  composite macro-F1: K=1 {fc[0]['macro_f1']:.3f}"
                  f"  K=10 {fc[3]['macro_f1']:.3f}  K=50 {fc[-1]['macro_f1']:.3f}"
                  f"  ({time.time()-t0:.0f}s, ETA {eta:.0f} min)", flush=True)

# ------------------------------------------------------------------ statistics
df = pd.read_csv(csv_path)
df["run"] = df.fold.astype(str) + "_" + df.seed.astype(str)
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


def ci_t(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def crossing(xs, ys, thr):
    if ys[0] < thr:
        return 0.0
    for i in range(1, len(xs)):
        if ys[i] < thr:
            return xs[i - 1] + (ys[i - 1] - thr) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1])
    return xs[-1]


cfgs = [c for c in CONFIGS if c in df.config.unique()]
L = df[df["agg"] == "logp"]
say(f"runs per config = {L[L.config == PROP].run.nunique()}, bags/class/fold = {args.bags}, aggregation = mean log-prob")

for dom in ["field_composite", "field_extreme_OOD"]:
    for metric, name in [("macro_f1", "macro-F1"), ("false_alarm", "false-alarm rate (noise bags flagged as PD)"),
                         ("rec_Corona", "Corona recall"), ("det_recall", "PD detection recall")]:
        say(f"\n=== {dom}: {name}, mean [95% CI] vs K ===")
        tab = {}
        for c in cfgs:
            tab[c] = {K: "{:.3f} [{:.3f},{:.3f}]".format(*ci_t(L[(L.config == c) & (L.domain == dom) & (L.K == K)][metric]))
                      for K in KS}
        say(pd.DataFrame(tab).T.to_string())
    say(f"\n--- {dom}: PIDR_full vs baselines, macro-F1, Wilcoxon signed-rank (Holm within K) ---")
    for K in KS:
        W = L[(L.domain == dom) & (L.K == K)].pivot_table(index="run", columns="config", values="macro_f1").dropna()
        if PROP not in W:
            continue
        comps = [c for c in cfgs if c != PROP and c in W]
        ps, ds = [], []
        for c in comps:
            d = W[PROP] - W[c]
            try:
                p = stats.wilcoxon(W[PROP], W[c], zero_method="zsplit").pvalue if not np.allclose(d, 0) else 1.0
            except ValueError:
                p = 1.0
            ps.append(p)
            ds.append((c, d.mean(), int((d > 0).sum()), len(d)))
        say(f"K={K:2d}: " + " | ".join(f"vs {c}: {dm:+.3f} ({w}/{n} wins, p_holm={pa:.2g})"
                                       for (c, dm, w, n), pa in zip(ds, holm(ps))))

# gain from aggregation for each method (K=50 - K=1), paired
say("\n=== Gain from listening longer: macro-F1(K=50) - macro-F1(K=1), field_composite ===")
for c in cfgs:
    a = L[(L.config == c) & (L.domain == "field_composite") & (L.K == 1)].set_index("run").macro_f1
    b = L[(L.config == c) & (L.domain == "field_composite") & (L.K == KMAX)].set_index("run").macro_f1
    d = (b - a).dropna()
    m, lo, hi = ci_t(d)
    say(f"{c:11s} gain = {m:+.3f} [{lo:+.3f},{hi:+.3f}]")

# vote vs logp
V = df[(df.domain == "field_composite") & (df.config == PROP)].groupby(["agg", "K"]).macro_f1.mean().unstack()
say("\nRobustness check (PIDR_full, field_composite): mean-log-prob vs majority vote")
say(V.round(3).to_string())

# per-class detection range vs K
dist_doms = sorted(d for d in L.domain.unique() if d.startswith("dist_"))
dists = np.array([float(d[5:7]) for d in dist_doms])
say("\n=== Per-class detection range [m] (bag recall >= 0.8), mean over runs ===")
rng_rows = []
for c in cfgs:
    for K in KS:
        r = {"config": c, "K": K}
        for k in range(3):
            col = f"rec_{pc.CLASS_NAMES[k]}"
            vals = []
            for run, g in L[(L.config == c) & (L.K == K) & L.domain.isin(dist_doms)].groupby("run"):
                yv = g.set_index("domain").reindex(dist_doms)[col].values
                vals.append(crossing(dists, yv, 0.8))
            r[pc.CLASS_NAMES[k]] = np.mean(vals) if vals else np.nan
        rng_rows.append(r)
RR = pd.DataFrame(rng_rows)
RR.to_csv(os.path.join(out, "mw_detection_range.csv"), index=False)
for cls in ["Corona", "Void", "Air"]:
    say(f"\n{cls} range [m]  (columns = K)")
    say(RR.pivot(index="config", columns="K", values=cls).round(2).to_string())

# ------------------------------------------------------------------ figures
plt = pc.plot_style()
for metric, ylab, fn in [("macro_f1", "macro-F1 (mean, 95% CI)", "fig_mw_macroF1_vs_K.png"),
                         ("false_alarm", "false-alarm rate (mean, 95% CI)", "fig_mw_falsealarm_vs_K.png")]:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dom in zip(axs, ["field_composite", "field_extreme_OOD"]):
        for i, c in enumerate(["ERM_raw", "GenericAug", "RF_PIDR", PROP]):
            if c not in cfgs:
                continue
            M, LO, HI = zip(*[ci_t(L[(L.config == c) & (L.domain == dom) & (L.K == K)][metric]) for K in KS])
            ax.fill_between(KS, LO, HI, color=pc.PALETTE[i], alpha=0.18, linewidth=0)
            ax.plot(KS, M, "o-", color=pc.PALETTE[i], markersize=6, label=c)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("windows fused per decision, K")
        ax.set_title(dom)
    axs[0].set_ylabel(ylab)
    axs[0].legend(loc="best")
    fig.suptitle("Continuous listening: decision from K consecutive windows")
    fig.savefig(os.path.join(out, fn))
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
sub = RR[RR.config == PROP]
for k in range(3):
    ax.plot(sub.K, sub[pc.CLASS_NAMES[k]], "o-", color=pc.CLASS_COLORS[k], markersize=6, label=pc.CLASS_NAMES[k])
ax.set_xscale("log")
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.set_xlabel("windows fused per decision, K")
ax.set_ylabel("detection range [m] (recall ≥ 0.8)")
ax.set_title(f"{PROP}: detection range grows with listening time")
ax.legend(loc="upper left")
fig.savefig(os.path.join(out, "fig_mw_range_vs_K.png"))
plt.close(fig)

with open(os.path.join(out, "MULTIWINDOW_SUMMARY.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"\nSaved to {out}  ->  send MULTIWINDOW_SUMMARY.txt")
