"""
02_run_experiments.py  -  Step 2: train + evaluate every method and every ablation
=================================================================================
Protocol (pre-registered, fixed before seeing results):
  * 5-fold BLOCKED + PURGED CV (contiguous recording blocks, 10-row guard band)  x  3 seeds
    -> 15 paired runs per configuration.
  * Every model is tested on the SAME seeded field-domain copies of each test fold
    (paired design -> Wilcoxon / Friedman / McNemar in step 3).
  * Noise beds used by the simulator at train time come ONLY from the training fold,
    at test time ONLY from the test fold (no leakage through the noise floor).
  * Fixed epochs, no early stopping, no test-set model selection.
  * PRIMARY endpoint: macro-F1 on 'field_composite'. Key secondary: 'field_extreme_OOD'.

Configurations
  Probe_LR        logistic regression on 3 shortcut numbers (loudness, peak, arrival index)
  RF_clean        random forest, 18 hand-crafted features, clean lab data
  RF_PIDR         same RF trained on PIDR-simulated data (method is model-agnostic)
  ERM_raw         TinyPDNet, no augmentation, raw amplitude kept           (typical lab paper)
  ERM_norm        + per-waveform standardisation
  GenericAug      + generic augmentation (gain, white noise, circular shift)
  PIDR_full       + physics-informed domain randomization + distance head  (PROPOSED)
  abl_noAbsorption / abl_noDistance / abl_noReverb / abl_noShift / abl_noSensor /
  abl_noFieldNoise / abl_noAux / abl_noNorm      leave-one-component-out ablations

Usage
  python 02_run_experiments.py --quick          # 2-5 min smoke test, run this first
  python 02_run_experiments.py                  # full run (resumable: just re-run if stopped)
Output: results/runs.csv, results/preds/*.npz, results/efficiency.json, results/run_log.txt
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

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
ap.add_argument("--lam", type=float, default=0.3, help="weight of distance head")
ap.add_argument("--purge", type=int, default=10)
ap.add_argument("--rf_copies", type=int, default=5, help="simulated copies for RF_PIDR")
ap.add_argument("--device", default="auto")
ap.add_argument("--configs", default="all", help="comma list, or 'all'")
ap.add_argument("--quick", action="store_true", help="2 folds x 1 seed x 4 epochs smoke test")
ap.add_argument("--n_folds_run", type=int, default=None, help="run only the first n folds")
args = ap.parse_args()

if args.quick:
    args.folds_used, args.seeds, args.epochs, args.rf_copies = 2, 1, 4, 1
else:
    args.folds_used = args.n_folds_run or args.folds

here = os.path.dirname(os.path.abspath(__file__))
out = args.out or os.path.join(here, "results_quick" if args.quick else "results")
os.makedirs(os.path.join(out, "preds"), exist_ok=True)
device = pc.get_device(args.device)
torch.set_num_threads(max(1, os.cpu_count() or 1))

T = pc.TRAIN_SPEC
CONFIGS = {
    "Probe_LR":         dict(kind="probe"),
    "RF_clean":         dict(kind="rf", spec=None),
    "RF_PIDR":          dict(kind="rf", spec=T),
    "ERM_raw":          dict(kind="cnn", spec=None, norm=False, aux=False),
    "ERM_norm":         dict(kind="cnn", spec=None, norm=True, aux=False),
    "GenericAug":       dict(kind="cnn", spec=pc.GENERIC_SPEC, norm=True, aux=False),
    "PIDR_full":        dict(kind="cnn", spec=T, norm=True, aux=True),
    "abl_noAbsorption": dict(kind="cnn", spec=pc.ablate(T, "absorption"), norm=True, aux=True),
    "abl_noDistance":   dict(kind="cnn", spec=pc.ablate(T, "distance"), norm=True, aux=False),
    "abl_noReverb":     dict(kind="cnn", spec=pc.ablate(T, "reverb"), norm=True, aux=True),
    "abl_noShift":      dict(kind="cnn", spec=pc.ablate(T, "shift"), norm=True, aux=True),
    "abl_noSensor":     dict(kind="cnn", spec=pc.ablate(T, "sensor"), norm=True, aux=True),
    "abl_noFieldNoise": dict(kind="cnn", spec=pc.ablate(T, "noise"), norm=True, aux=True),
    "abl_noAux":        dict(kind="cnn", spec=T, norm=True, aux=False),
    "abl_noNorm":       dict(kind="cnn", spec=T, norm=False, aux=True),
}
if args.configs != "all":
    keep = [c.strip() for c in args.configs.split(",")]
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in keep}

SAVE_PRED_DOMAINS = ["clean", "field_composite", "field_extreme_OOD"]
runs_csv = os.path.join(out, "runs.csv")
log_fh = open(os.path.join(out, "run_log.txt"), "a", encoding="utf-8")


def log(s):
    print(s, flush=True)
    log_fh.write(s + "\n")
    log_fh.flush()


def load_done():
    import pandas as pd
    if not os.path.isfile(runs_csv):
        return set()
    d = pd.read_csv(runs_csv)
    return set(zip(d.config, d.fold, d.seed))


COLS = ["config", "fold", "seed", "domain", "macro_f1", "bal_acc", "acc", "det_recall",
        "det_false_alarm", "ece", "dist_med_rel_err", "dist_med_est_m", "cm", "train_eval_sec"]


def append_rows(rows):
    import pandas as pd
    df = pd.DataFrame(rows).reindex(columns=COLS)          # fixed schema -> safe appends
    df.to_csv(runs_csv, mode="a", header=not os.path.isfile(runs_csv), index=False)


# ------------------------------------------------------------------ neural training
def train_cnn(cfg, Xtr, ytr, sim, fold, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gscale = float(np.std(Xtr))
    model = pc.TinyPDNet(args.width, cfg["norm"], gscale).to(device)
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    n = len(xt)
    spe = int(np.ceil(n / args.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * spe)
    # common random numbers: same augmentation stream for every config at a given (fold, seed)
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


@torch.no_grad()
def predict_cnn(model, x):
    P, D = [], []
    for i in range(0, len(x), 1024):
        lo, dd = model(x[i:i + 1024])
        P.append(torch.softmax(lo, 1).cpu())
        D.append(dd.cpu())
    return torch.cat(P).numpy(), torch.cat(D).numpy()


# ------------------------------------------------------------------ main loop
X, y, row = pc.load_data(pc.find_data_dir(args.data_dir))
folds = pc.blocked_purged_folds(y, row, args.folds, args.purge)
done = load_done()
total = len(CONFIGS) * args.folds_used * args.seeds
log(f"device={device}  configs={len(CONFIGS)}  folds={args.folds_used}/{args.folds}  seeds={args.seeds}  "
    f"epochs={args.epochs}  -> {total} runs ({len(done)} already done)")

# efficiency of the edge model
eff_path = os.path.join(out, "efficiency.json")
if not os.path.isfile(eff_path):
    m = pc.TinyPDNet(args.width).eval()
    xb = torch.randn(1, pc.N)
    with torch.no_grad():
        for _ in range(20):
            m(xb)
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            m(xb)
            ts.append((time.perf_counter() - t0) * 1e3)
    eff = dict(params=int(sum(p.numel() for p in m.parameters())), macs=int(pc.count_macs(args.width)),
               cpu_latency_ms_median=float(np.median(ts)), cpu_latency_ms_p95=float(np.percentile(ts, 95)),
               weights_kB_fp32=float(sum(p.numel() for p in m.parameters()) * 4 / 1024),
               weights_kB_int8=float(sum(p.numel() for p in m.parameters()) / 1024))
    json.dump(eff, open(eff_path, "w"), indent=2)
    log(f"efficiency: {eff}")

t_start = time.time()
n_new = 0
for fold in range(args.folds_used):
    tr, te = folds[fold]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    bank_tr, bank_te = Xtr[ytr == pc.NOISE_CLASS], Xte[yte == pc.NOISE_CLASS]
    floor = float(np.median(np.sqrt((bank_tr ** 2).mean(1))))
    sim_tr = pc.FieldSimulator(bank_tr, floor, device)
    sim_te = pc.FieldSimulator(bank_te, floor, device)
    evalsets = pc.build_eval_sets(Xte, sim_te, fold, device)
    np.save(os.path.join(out, "preds", f"ytrue_f{fold}.npy"), yte)
    feat_cache = {}

    for cname, cfg in CONFIGS.items():
        for seed in range(args.seeds):
            if (cname, fold, seed) in done:
                continue
            t0 = time.time()
            rows, saved = [], {}
            if cfg["kind"] == "cnn":
                model = train_cnn(cfg, Xtr, ytr, sim_tr, fold, seed)
                if cname == "PIDR_full" and fold == 0 and seed == 0:
                    torch.save(model.state_dict(), os.path.join(out, "PIDR_full_f0_s0.pt"))
                for dname, (xd, ld) in evalsets.items():
                    prob, dpred = predict_cnn(model, xd)
                    pred = prob.argmax(1)
                    met = pc.all_metrics(yte, pred, prob)
                    if cfg["aux"] and dname.startswith("dist_"):
                        dtrue = float(dname[5:7])
                        pdm = yte != pc.NOISE_CLASS
                        est = np.exp(dpred[pdm])
                        met["dist_med_rel_err"] = float(np.median(np.abs(est - dtrue) / dtrue))
                        met["dist_med_est_m"] = float(np.median(est))
                    rows.append(dict(config=cname, fold=fold, seed=seed, domain=dname, **met))
                    if dname in SAVE_PRED_DOMAINS:
                        saved[dname] = pred.astype(np.int8)
            else:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                if cfg["kind"] == "rf" and cfg["spec"] is not None:
                    g = torch.Generator(device=device)
                    g.manual_seed(pc.seed_of("rfaug", fold, seed))
                    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
                    parts, ys = [Xtr], [ytr]
                    for _ in range(args.rf_copies):
                        z, _, _ = sim_tr(xt, cfg["spec"], g)
                        parts.append(z.cpu().numpy())
                        ys.append(ytr)
                    Ftr, ytr_ = pc.features(np.concatenate(parts)), np.concatenate(ys)
                else:
                    Ftr, ytr_ = pc.features(Xtr), ytr
                if cfg["kind"] == "probe":
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
                    cols = pc.SHORTCUT_IDX
                else:
                    clf = RandomForestClassifier(300, n_jobs=-1, random_state=seed, min_samples_leaf=2)
                    cols = list(range(len(pc.FEATURE_NAMES)))
                clf.fit(Ftr[:, cols], ytr_)
                for dname, (xd, ld) in evalsets.items():
                    if dname not in feat_cache:
                        feat_cache[dname] = pc.features(xd.cpu().numpy())
                    prob = clf.predict_proba(feat_cache[dname][:, cols])
                    pred = prob.argmax(1)
                    rows.append(dict(config=cname, fold=fold, seed=seed, domain=dname,
                                     **pc.all_metrics(yte, pred, prob)))
                    if dname in SAVE_PRED_DOMAINS:
                        saved[dname] = pred.astype(np.int8)
            dt = time.time() - t0
            for r in rows:
                r["train_eval_sec"] = dt
            append_rows(rows)
            np.savez_compressed(os.path.join(out, "preds", f"{cname}__f{fold}__s{seed}.npz"), **saved)
            n_new += 1
            fc = [r for r in rows if r["domain"] == "field_composite"][0]["macro_f1"]
            cl = [r for r in rows if r["domain"] == "clean"][0]["macro_f1"]
            ex = [r for r in rows if r["domain"] == "field_extreme_OOD"][0]["macro_f1"]
            remaining = total - len(done) - n_new
            eta = (time.time() - t_start) / n_new * remaining / 60
            log(f"[{len(done)+n_new:3d}/{total}] {cname:17s} fold={fold} seed={seed}  "
                f"clean={cl:.3f}  field={fc:.3f}  OOD={ex:.3f}  ({dt:.0f}s, ETA {eta:.0f} min)")

log(f"finished. results in {out}")
