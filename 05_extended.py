"""
05_extended.py  -  Step 5: reviewer-proofing experiments (Q1 package)
====================================================================
Stages (run all with no arguments; each stage is resumable and can be run alone):

  physics : unit tests of the simulator (nominal vs measured) + spectral-centroid shift with distance
  main    : architecture x augmentation benchmark WITH an inner validation split
            archs  : TinyPDNet (ours, 26k), ResNet1D (~260k), InceptionTime-lite, Spectrogram-2D-CNN, RandomForest
            augs   : ERM (none), GenericAug, Mixup, PIDR (proposed, no distance head)
            extra  : AUROC (PD vs no-PD, Corona vs Noise), operating points at a FIXED false-alarm
                     rate (5 % / 10 %) whose threshold is set on the VALIDATION split only
  hp      : hyper-parameter sensitivity, lr x epochs grid (3x3) for Tiny ERM_raw / GenericAug / PIDR;
            selection by clean-validation, by field-validation and by an ORACLE on the test domain
  xai     : what does the model listen to?  gain sweep, frequency-band occlusion, time occlusion
  stats   : tables, paired tests (Wilcoxon + Holm), Friedman/Nemenyi, figures, EXTENDED_SUMMARY.txt

Usage:  python 05_extended.py --quick          (smoke test, a few minutes)
        python 05_extended.py                  (everything, several hours on CPU; resumable)
        python 05_extended.py --stage main     (one stage only)
Output: results_ext/  ->  send results_ext/EXTENDED_SUMMARY.txt
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_common as pc

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--stage", default="all", choices=["all", "physics", "main", "hp", "xai", "stats"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--purge", type=int, default=10)
ap.add_argument("--val_frac", type=float, default=0.15)
ap.add_argument("--rf_copies", type=int, default=5)
ap.add_argument("--device", default="auto")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

HP_LR = [1e-3, 3e-3, 1e-2]
HP_EP = [30, 60, 120]
n_folds_run = args.folds
if args.quick:
    n_folds_run, args.seeds, args.epochs, args.rf_copies = 1, 1, 3, 1
    HP_LR, HP_EP = [3e-3, 1e-2], [2, 3]

here = os.path.dirname(os.path.abspath(__file__))
OUT = args.out or os.path.join(here, "results_ext_quick" if args.quick else "results_ext")
os.makedirs(os.path.join(OUT, "scores"), exist_ok=True)
device = pc.get_device(args.device)
T = pc.TRAIN_SPEC
PRIMARY, OOD = "field_composite", "field_extreme_OOD"
PROP = "Tiny_PIDR"


# ============================================================================ models
class InputNorm(nn.Module):
    def __init__(self, normalize, gscale):
        super().__init__()
        self.normalize = normalize
        self.register_buffer("gscale", torch.tensor(float(gscale)))

    def forward(self, x):
        if self.normalize:
            return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
        return x / self.gscale


class RB(nn.Module):
    def __init__(self, ci, co, s):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 3, s, 1, bias=False)
        self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm1d(co)
        self.sc = nn.Identity() if (ci == co and s == 1) else nn.Sequential(
            nn.Conv1d(ci, co, 1, s, bias=False), nn.BatchNorm1d(co))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class ResNet1D(nn.Module):
    def __init__(self, normalize=True, gscale=1.0):
        super().__init__()
        self.inp = InputNorm(normalize, gscale)
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, 1, 3, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.body = nn.Sequential(RB(32, 32, 1), RB(32, 32, 1), RB(32, 64, 2), RB(64, 64, 1),
                                  RB(64, 128, 2), RB(128, 128, 1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, pc.N_CLASSES))

    def forward(self, x):
        h = self.body(self.stem(self.inp(x)[:, None, :]))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h), torch.zeros(len(x), device=x.device)


class IncMod(nn.Module):
    def __init__(self, ci, nf=16, ks=(9, 19, 39), bott=16):
        super().__init__()
        self.bott = nn.Conv1d(ci, bott, 1, bias=False) if ci > 1 else nn.Identity()
        bi = bott if ci > 1 else ci
        self.convs = nn.ModuleList([nn.Conv1d(bi, nf, k, padding=k // 2, bias=False) for k in ks])
        self.mp = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(ci, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))

    def forward(self, x):
        z = self.bott(x)
        return F.relu(self.bn(torch.cat([c(z) for c in self.convs] + [self.mp(x)], 1)))


class InceptionLite(nn.Module):
    def __init__(self, normalize=True, gscale=1.0):
        super().__init__()
        self.inp = InputNorm(normalize, gscale)
        self.m = nn.Sequential(IncMod(1), IncMod(64), IncMod(64))
        self.sc = nn.Sequential(nn.Conv1d(1, 64, 1, bias=False), nn.BatchNorm1d(64))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, pc.N_CLASSES))

    def forward(self, x):
        z = self.inp(x)[:, None, :]
        h = F.relu(self.m(z) + self.sc(z))
        return self.head(h.mean(-1)), torch.zeros(len(x), device=x.device)


class Spec2D(nn.Module):
    """log-magnitude STFT (n_fft 32, hop 4 -> 17 x 32) + small 2-D CNN"""

    def __init__(self, normalize=True, gscale=1.0):
        super().__init__()
        self.inp = InputNorm(normalize, gscale)
        self.register_buffer("win", torch.hann_window(32))

        def blk(ci, co, pool):
            L = [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU()]
            return nn.Sequential(*(L + ([nn.MaxPool2d(2)] if pool else [])))
        self.cnn = nn.Sequential(blk(1, 16, True), blk(16, 32, True), blk(32, 64, False))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, pc.N_CLASSES))

    def forward(self, x):
        z = self.inp(x)
        S = torch.stft(z, n_fft=32, hop_length=4, win_length=32, window=self.win, center=True,
                       return_complex=True).abs()
        h = self.cnn(torch.log(S + 1e-4)[:, None])
        return self.head(h.mean((-1, -2))), torch.zeros(len(x), device=x.device)


def make_model(arch, norm, gscale):
    if arch == "Tiny":
        return pc.TinyPDNet(16, norm, gscale)
    return {"ResNet": ResNet1D, "Incep": InceptionLite, "Spec": Spec2D}[arch](norm, gscale)


#             name               arch      aug        norm
CONFIGS = {
    "Tiny_ERM_raw":    ("Tiny",   "ERM",     False),
    "Tiny_ERM":        ("Tiny",   "ERM",     True),
    "Tiny_GenericAug": ("Tiny",   "Generic", True),
    "Tiny_Mixup":      ("Tiny",   "Mixup",   True),
    "Tiny_PIDR":       ("Tiny",   "PIDR",    True),
    "ResNet_ERM":      ("ResNet", "ERM",     True),
    "ResNet_PIDR":     ("ResNet", "PIDR",    True),
    "Incep_ERM":       ("Incep",  "ERM",     True),
    "Incep_PIDR":      ("Incep",  "PIDR",    True),
    "Spec_ERM":        ("Spec",   "ERM",     True),
    "Spec_PIDR":       ("Spec",   "PIDR",    True),
    "RF_ERM":          ("RF",     "ERM",     None),
    "RF_PIDR":         ("RF",     "PIDR",    None),
}
ARCH_PAIRS = [("Tiny_ERM", "Tiny_PIDR"), ("ResNet_ERM", "ResNet_PIDR"), ("Incep_ERM", "Incep_PIDR"),
              ("Spec_ERM", "Spec_PIDR"), ("RF_ERM", "RF_PIDR")]


def aug_spec(aug):
    return {"ERM": None, "Generic": pc.GENERIC_SPEC, "Mixup": None, "PIDR": T}[aug]


# ============================================================================ data & splits
def inner_split(tr, y, row, frac, purge):
    """last `frac` (in recording order) of each class inside the training fold -> validation"""
    itr, iva = [], []
    for c in range(pc.N_CLASSES):
        ids = tr[y[tr] == c]
        ids = ids[np.argsort(row[ids])]
        nv = int(round(len(ids) * frac))
        iva.append(ids[len(ids) - nv:])
        itr.append(ids[:max(0, len(ids) - nv - purge)])
    return np.concatenate(itr), np.concatenate(iva)


class FoldData:
    def __init__(self, X, y, row, folds, fold):
        tr, te = folds[fold]
        itr, iva = inner_split(tr, y, row, args.val_frac, args.purge)
        self.Xtr, self.ytr = X[itr], y[itr]
        self.Xva, self.yva = X[iva], y[iva]
        self.Xte, self.yte = X[te], y[te]
        bank = lambda A, b: A[b == pc.NOISE_CLASS]
        floor = float(np.median(np.sqrt((bank(self.Xtr, self.ytr) ** 2).mean(1))))
        self.floor = floor
        self.sim_tr = pc.FieldSimulator(bank(self.Xtr, self.ytr), floor, device)
        self.sim_va = pc.FieldSimulator(bank(self.Xva, self.yva), floor, device)
        self.sim_te = pc.FieldSimulator(bank(self.Xte, self.yte), floor, device)
        self.evalsets = pc.build_eval_sets(self.Xte, self.sim_te, fold, device)   # same as step 2
        xv = torch.as_tensor(self.Xva, dtype=torch.float32, device=device)
        g = torch.Generator(device=device)
        g.manual_seed(pc.seed_of("val", fold))
        self.val_clean = xv
        self.val_field = self.sim_va(xv, pc.eval_domains()[PRIMARY], g)[0]


# ============================================================================ training
def train_nn(arch, aug, norm, fd, fold, seed, lr, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arch, norm, float(np.std(fd.Xtr))).to(device)
    xt = torch.as_tensor(fd.Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(fd.ytr, dtype=torch.long, device=device)
    n = len(xt)
    spe = int(np.ceil(n / args.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe)
    g = torch.Generator(device=device)
    g.manual_seed(pc.seed_of("train", fold, seed))
    spec = aug_spec(aug)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(spe):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            xb, yb = xt[idx], yt[idx]
            with torch.no_grad():
                xb, _, _ = fd.sim_tr(xb, spec, g)
            if aug == "Mixup":
                lam = float(np.random.beta(0.4, 0.4))
                j = torch.randperm(len(xb), generator=g, device=device)
                logits, _ = model(lam * xb + (1 - lam) * xb[j])
                loss = lam * F.cross_entropy(logits, yb) + (1 - lam) * F.cross_entropy(logits, yb[j])
            else:
                logits, _ = model(xb)
                loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    model.eval()
    return model


def train_rf(aug, fd, fold, seed):
    from sklearn.ensemble import RandomForestClassifier
    parts, ys = [fd.Xtr], [fd.ytr]
    if aug == "PIDR":
        g = torch.Generator(device=device)
        g.manual_seed(pc.seed_of("rfaug", fold, seed))
        xt = torch.as_tensor(fd.Xtr, dtype=torch.float32, device=device)
        for _ in range(args.rf_copies):
            parts.append(fd.sim_tr(xt, T, g)[0].cpu().numpy())
            ys.append(fd.ytr)
    clf = RandomForestClassifier(300, n_jobs=-1, random_state=seed, min_samples_leaf=2)
    clf.fit(pc.features(np.concatenate(parts)), np.concatenate(ys))
    return clf


@torch.no_grad()
def predict(model, x, is_rf=False):
    if is_rf:
        return model.predict_proba(pc.features(x.cpu().numpy()))
    P = []
    for i in range(0, len(x), 2048):
        P.append(torch.softmax(model(x[i:i + 2048])[0], 1).cpu())
    return torch.cat(P).numpy()


def thresholds_from_val(p_val, y_val, targets=(0.05, 0.10)):
    s = 1.0 - p_val[:, pc.NOISE_CLASS]
    sn = s[y_val == pc.NOISE_CLASS]
    return {t: float(np.quantile(sn, 1.0 - t)) for t in targets}


def eval_row(prob, y, thr):
    from sklearn.metrics import roc_auc_score
    m = pc.all_metrics(y, prob.argmax(1), prob)
    s = 1.0 - prob[:, pc.NOISE_CLASS]
    ypd = (y != pc.NOISE_CLASS).astype(int)
    m["auroc_pd"] = float(roc_auc_score(ypd, s)) if 0 < ypd.sum() < len(ypd) else np.nan
    cn = (y == 2) | (y == pc.NOISE_CLASS)
    m["auroc_corona"] = float(roc_auc_score((y[cn] == 2).astype(int), prob[cn, 2] /
                                            (prob[cn, 2] + prob[cn, pc.NOISE_CLASS] + 1e-12)))
    for t, tau in thr.items():
        flag = s > tau
        k = int(round(t * 100))
        m[f"fa_at{k}"] = float(flag[y == pc.NOISE_CLASS].mean())
        m[f"recall_at{k}"] = float(flag[y != pc.NOISE_CLASS].mean())
        m[f"corona_at{k}"] = float(flag[y == 2].mean())
    pred = prob.argmax(1)
    for c in range(pc.N_CLASSES):
        m[f"rec_{pc.CLASS_NAMES[c]}"] = float((pred[y == c] == c).mean())
    return m


def append_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def done_set(path, keys):
    if not os.path.isfile(path):
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[keys].astype(str).values))


# ============================================================================ STAGE physics
def stage_physics(X, y):
    path = os.path.join(OUT, "physics_checks.csv")
    if os.path.isfile(path):
        print("physics: already done")
        return
    print("physics: simulator unit tests ...", flush=True)
    bank = X[y == pc.NOISE_CLASS]
    floor = float(np.median(np.sqrt((bank ** 2).mean(1))))
    sim = pc.FieldSimulator(bank, floor, device)
    g = torch.Generator(device=device)
    g.manual_seed(123)
    rows = []
    xa = torch.as_tensor(X[y == 0][:200], dtype=torch.float32, device=device)       # loud Air pulses
    freqs = np.fft.rfftfreq(pc.NFFT, 1 / pc.FS)
    X0 = np.abs(np.fft.rfft(xa.cpu().numpy(), n=pc.NFFT))
    for d in [2, 4, 8]:
        s = dict(p_clean=0, floor_bed=False, prop=dict(on=True, p=1, d=(d, d), T=(20, 20), RH=(50, 50),
                                                        absorption=False))
        z = sim(xa, s, g)[0].cpu().numpy()
        meas = 20 * np.log10(np.sqrt((z ** 2).mean()) / np.sqrt((xa.cpu().numpy() ** 2).mean()))
        rows.append(dict(check=f"spherical spreading d={d} m [dB]", nominal=-20 * np.log10(d), measured=meas))
        s["prop"]["absorption"] = True
        z = sim(xa, s, g)[0].cpu().numpy()
        Z = np.abs(np.fft.rfft(z, n=pc.NFFT))
        for fk in [30e3, 50e3, 70e3, 90e3]:
            b = int(np.argmin(np.abs(freqs - fk)))
            meas = 20 * np.log10(np.median(Z[:, b] / (X0[:, b] + 1e-30)))
            nom = -20 * np.log10(d) - pc.iso9613_alpha(np.array(freqs[b]), 20.0, 50.0) * (d - 1)
            rows.append(dict(check=f"spreading+absorption d={d} m @ {fk/1e3:.0f} kHz [dB]",
                             nominal=float(nom), measured=float(meas)))
    zeros = torch.zeros(500, pc.N, device=device)
    for L in [0, 10, 20]:
        s = dict(p_clean=0, floor_bed=False, noise=dict(on=True, p=1, level=(L, L), beta=(0, 2), p_imp=0))
        z = sim(zeros, s, g)[0].cpu().numpy()
        rows.append(dict(check=f"background noise level {L} dB re floor [dB]", nominal=float(L),
                         measured=float(20 * np.log10(np.sqrt((z ** 2).mean()) / floor))))
    delta = torch.zeros(500, pc.N, device=device)
    delta[:, 0] = 1.0
    for drr in [0, 10, 20]:
        s = dict(p_clean=0, floor_bed=False, reverb=dict(on=True, p=1, drr=(drr, drr), tau_us=(100, 100)))
        z = sim(delta, s, g)[0].cpu().numpy()
        meas = 10 * np.log10((z[:, 0] ** 2).mean() / ((z[:, 1:] ** 2).sum(1)).mean())
        rows.append(dict(check=f"direct-to-reverberant ratio {drr} dB [dB]", nominal=float(drr), measured=float(meas)))
    dmid = torch.zeros(500, pc.N, device=device)
    dmid[:, pc.N // 2] = 1.0                         # zero-phase filter -> impulse in the middle
    for gain in [-10, 0, 10]:
        s = dict(p_clean=0, floor_bed=False, sensor=dict(on=True, p=1, fc=(50, 50), q=(1, 1), peak=(6, 6),
                                                         tilt=(0, 0), gain=(gain, gain)))
        z = sim(dmid, s, g)[0].cpu().numpy()
        H = np.abs(np.fft.rfft(z, n=pc.NFFT)).mean(0)
        b = int(np.argmin(np.abs(freqs - 50e3)))
        rows.append(dict(check=f"sensor gain {gain} dB + 6 dB resonance @ fc [dB]", nominal=float(gain + 6),
                         measured=float(20 * np.log10(H[b]))))
    xv = torch.as_tensor(X[y == 1][:500], dtype=torch.float32, device=device)   # Void: peak mid-window
    s = dict(p_clean=0, floor_bed=False, shift=dict(on=True, p=1, max=45))
    z = sim(xv, s, g)[0].cpu().numpy()
    pk = np.abs(z).argmax(1) - np.abs(xv.cpu().numpy()).argmax(1)
    rows.append(dict(check="trigger-loss shift: std of peak displacement [samples]",
                     nominal=float(np.sqrt(((2 * 45 + 1) ** 2 - 1) / 12)), measured=float(pk.std())))
    df = pd.DataFrame(rows)
    df["abs_error"] = (df.measured - df.nominal).abs()
    df.to_csv(path, index=False)
    # spectral centroid vs distance (simulated) for each PD class - the "physics changes the shape" figure
    cent = []
    for c in range(3):
        xc = torch.as_tensor(X[y == c][:300], dtype=torch.float32, device=device)
        for d in [1, 2, 4, 6, 8, 10, 15, 20]:
            s = dict(p_clean=0, floor_bed=False, prop=dict(on=True, p=1, d=(d, d), T=(20, 20), RH=(50, 50)))
            z = sim(xc, s, g)[0].cpu().numpy()
            cent.append(dict(cls=pc.CLASS_NAMES[c], d=d, f_centroid_khz=float(np.median(pc.features(z)[:, 9]))))
    pd.DataFrame(cent).to_csv(os.path.join(OUT, "physics_centroid.csv"), index=False)
    print(df.round(3).to_string(index=False))


# ============================================================================ STAGE main
def stage_main(X, y, row, folds):
    path = os.path.join(OUT, "main_runs.csv")
    done = done_set(path, ["config", "fold", "seed"])
    total = len(CONFIGS) * n_folds_run * args.seeds
    print(f"main: {total} runs ({len(done)} done)", flush=True)
    t0all, n_new = time.time(), 0
    for fold in range(n_folds_run):
        todo = [(c, s) for c in CONFIGS for s in range(args.seeds) if (c, str(fold), str(s)) not in done]
        if not todo:
            continue
        fd = FoldData(X, y, row, folds, fold)
        for cname, seed in todo:
            arch, aug, norm = CONFIGS[cname]
            t0 = time.time()
            if arch == "RF":
                model, is_rf, npar = train_rf(aug, fd, fold, seed), True, 0
            else:
                model = train_nn(arch, aug, norm, fd, fold, seed, args.lr, args.epochs)
                is_rf, npar = False, int(sum(p.numel() for p in model.parameters()))
            thr = thresholds_from_val(predict(model, fd.val_field, is_rf), fd.yva)
            rows, saved = [], {}
            for dname, (xd, _) in fd.evalsets.items():
                prob = predict(model, xd, is_rf)
                m = eval_row(prob, fd.yte, thr)
                rows.append(dict(config=cname, arch=arch, aug=aug, fold=fold, seed=seed, domain=dname,
                                 n_params=npar, **m))
                if dname in (PRIMARY, OOD, "clean"):
                    saved[dname] = prob.astype(np.float32)
            saved["y"] = fd.yte
            np.savez_compressed(os.path.join(OUT, "scores", f"{cname}__f{fold}__s{seed}.npz"), **saved)
            append_csv(path, rows)
            n_new += 1
            r = {x["domain"]: x for x in rows}
            eta = (time.time() - t0all) / n_new * (total - len(done) - n_new) / 60
            print(f"  [{len(done)+n_new}/{total}] {cname:16s} f{fold} s{seed}  clean={r['clean']['macro_f1']:.3f} "
                  f"field={r[PRIMARY]['macro_f1']:.3f} OOD={r[OOD]['macro_f1']:.3f} "
                  f"recall@FA5%={r[PRIMARY]['recall_at5']:.3f}  ({time.time()-t0:.0f}s, ETA {eta:.0f} min)",
                  flush=True)


# ============================================================================ STAGE hp
def stage_hp(X, y, row, folds):
    path = os.path.join(OUT, "hp_runs.csv")
    done = done_set(path, ["config", "lr", "epochs", "fold"])
    methods = ["Tiny_ERM_raw", "Tiny_GenericAug", "Tiny_PIDR"]
    grid = [(m, lr, ep) for m in methods for lr in HP_LR for ep in HP_EP]
    total = len(grid) * n_folds_run
    print(f"hp: {total} runs ({len(done)} done)", flush=True)
    t0all, n_new = time.time(), 0
    for fold in range(n_folds_run):
        todo = [g_ for g_ in grid if (g_[0], str(g_[1]), str(g_[2]), str(fold)) not in done]
        if not todo:
            continue
        fd = FoldData(X, y, row, folds, fold)
        for cname, lr, ep in todo:
            arch, aug, norm = CONFIGS[cname]
            t0 = time.time()
            model = train_nn(arch, aug, norm, fd, fold, 0, lr, ep)
            from sklearn.metrics import f1_score
            f1 = lambda p, yy: f1_score(yy, p.argmax(1), average="macro")
            r = dict(config=cname, lr=lr, epochs=ep, fold=fold,
                     val_clean=f1(predict(model, fd.val_clean), fd.yva),
                     val_field=f1(predict(model, fd.val_field), fd.yva))
            for dname in ["clean", PRIMARY, OOD]:
                r[f"test_{dname}"] = f1(predict(model, fd.evalsets[dname][0]), fd.yte)
            append_csv(path, [r])
            n_new += 1
            eta = (time.time() - t0all) / n_new * (total - len(done) - n_new) / 60
            print(f"  [{len(done)+n_new}/{total}] {cname:16s} lr={lr:g} ep={ep:3d} f{fold}  "
                  f"field={r['test_'+PRIMARY]:.3f}  ({time.time()-t0:.0f}s, ETA {eta:.0f} min)", flush=True)


# ============================================================================ STAGE xai
def stage_xai(X, y, row, folds):
    path = os.path.join(OUT, "xai.csv")
    if os.path.isfile(path):
        print("xai: already done")
        return
    print("xai: gain sweep, band occlusion, time occlusion (fold 0, seed 0) ...", flush=True)
    from sklearn.metrics import f1_score
    fd = FoldData(X, y, row, folds, 0)
    models = {}
    for cname in ["Tiny_ERM_raw", "Tiny_GenericAug", "Tiny_PIDR"]:
        arch, aug, norm = CONFIGS[cname]
        models[cname] = (train_nn(arch, aug, norm, fd, 0, 0, args.lr, args.epochs), False)
    models["RF_ERM"] = (train_rf("ERM", fd, 0, 0), True)
    rows = []
    xc = fd.evalsets["clean"][0]
    xf = fd.evalsets[PRIMARY][0]
    F_ = lambda m, x: f1_score(fd.yte, predict(m[0], x, m[1]).argmax(1), average="macro")
    freqs = torch.fft.rfftfreq(pc.NFFT, 1 / pc.FS).to(device)
    bands = [(25e3, 40e3), (40e3, 55e3), (55e3, 70e3), (70e3, 85e3), (85e3, 100e3)]
    g = torch.Generator(device=device)
    g.manual_seed(7)
    bank = torch.as_tensor(fd.Xte[fd.yte == pc.NOISE_CLASS], dtype=torch.float32, device=device)
    for cname, m in models.items():
        base_c, base_f = F_(m, xc), F_(m, xf)
        for gdb in [-40, -30, -20, -10, 0, 10, 20, 30, 40]:
            rows.append(dict(test="gain", model=cname, x=gdb, macro_f1=F_(m, xc * 10 ** (gdb / 20))))
        for lo, hi in bands:
            for nm, xx, base in [("band_clean", xc, base_c), ("band_field", xf, base_f)]:
                Z = torch.fft.rfft(F.pad(xx, (0, pc.NFFT - pc.N)), n=pc.NFFT)
                Z[:, (freqs >= lo) & (freqs < hi)] = 0
                z = torch.fft.irfft(Z, n=pc.NFFT)[:, :pc.N]
                rows.append(dict(test=nm, model=cname, x=(lo + hi) / 2e3, macro_f1=F_(m, z), drop=base - F_(m, z)))
        for start in range(0, pc.N - 20 + 1, 15):
            z = xc.clone()
            idx = torch.randint(0, len(bank), (len(z),), generator=g, device=device)
            z[:, start:start + 20] = bank[idx][:, start:start + 20]          # replace by real floor
            rows.append(dict(test="time_clean", model=cname, x=(start + 10) * 2.0, macro_f1=F_(m, z),
                             drop=base_c - F_(m, z)))
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================================ STAGE stats
def stage_stats():
    lines = []
    say = lambda s="": (print(s), lines.append(str(s)))
    plt = pc.plot_style()

    def ci(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return (x.mean() if len(x) else np.nan), np.nan, np.nan
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return x.mean(), x.mean() - h, x.mean() + h

    def fmt(x):
        m, lo, hi = ci(x)
        return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

    def holm(p):
        p = np.asarray(p, float)
        o = np.argsort(p)
        adj, run = np.empty_like(p), 0.0
        for i, k in enumerate(o):
            run = max(run, (len(p) - i) * p[k])
            adj[k] = min(1.0, run)
        return adj

    def wil(a, b):
        d = np.asarray(a) - np.asarray(b)
        if np.allclose(d, 0):
            return 1.0
        try:
            return float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
        except ValueError:
            return 1.0

    # ---------------- physics
    p = os.path.join(OUT, "physics_checks.csv")
    if os.path.isfile(p):
        ph = pd.read_csv(p)
        say("=== S1. Simulator unit tests (nominal vs measured) ===")
        say(ph.round(3).to_string(index=False))
        say(f"max |error| = {ph.abs_error.max():.3f} dB-equivalent")
        cc = pd.read_csv(os.path.join(OUT, "physics_centroid.csv"))
        say("\nSpectral centroid [kHz] vs simulated distance (absorption shifts the spectrum down):")
        say(cc.pivot(index="cls", columns="d", values="f_centroid_khz").round(1).to_string())
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, cls in enumerate(["Air", "Void", "Corona"]):
            s = cc[cc.cls == cls]
            ax.plot(s.d, s.f_centroid_khz, "o-", color=pc.CLASS_COLORS[k], markersize=6, label=cls)
        ax.set_xlabel("simulated distance [m]")
        ax.set_ylabel("spectral centroid [kHz]")
        ax.set_title("ISO 9613-1 absorption changes the spectral shape with distance")
        ax.legend()
        fig.savefig(os.path.join(OUT, "fig_phys_centroid.png"))
        plt.close(fig)

    # ---------------- main
    p = os.path.join(OUT, "main_runs.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p)
        df["run"] = df.fold.astype(str) + "_" + df.seed.astype(str)
        cfgs = [c for c in CONFIGS if c in df.config.unique()]
        say(f"\n=== S2. Main benchmark (with inner validation), runs/config = {df[df.config == PROP].run.nunique()} ===")
        say("macro-F1 mean [95% CI]")
        keyd = ["clean", "untriggered", "reverb_drr3", "sensor_mems", "dist_04m", "dist_08m", "noise_15dB", PRIMARY, OOD]
        keyd = [k for k in keyd if k in df.domain.unique()]
        tab = pd.DataFrame({k: {c: fmt(df[(df.config == c) & (df.domain == k)].macro_f1) for c in cfgs} for k in keyd})
        say(tab.to_string())
        npar = df.groupby("config").n_params.first()
        say("\nparameters: " + ", ".join(f"{c}={int(npar[c])}" for c in cfgs if npar[c] > 0))

        for dom in [PRIMARY, OOD]:
            W = df[df.domain == dom].pivot_table(index="run", columns="config", values="macro_f1").dropna()
            comps = [c for c in cfgs if c != PROP and c in W]
            ps = [wil(W[PROP], W[c]) for c in comps]
            say(f"\n--- {dom}: {PROP} vs every other config (Wilcoxon, Holm) ---")
            for c, pa in zip(comps, holm(ps)):
                d = W[PROP] - W[c]
                say(f"  vs {c:16s} delta={d.mean():+.3f}  wins={int((d > 0).sum())}/{len(d)}  p_holm={pa:.2g}")
            k, Nn = W.shape[1], W.shape[0]
            if k >= 3 and Nn >= 2:
                chi, pf = stats.friedmanchisquare(*[W[c].values for c in W.columns])
                ranks = W.rank(axis=1, ascending=False).mean().sort_values()
                cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * Nn))
                say(f"  Friedman chi2={chi:.1f} p={pf:.2g}, Nemenyi CD={cd:.2f}; ranks: " +
                    ", ".join(f"{c}={v:.2f}" for c, v in ranks.items()))

        say("\n=== S3. Is PIDR architecture-agnostic? delta = (X_PIDR - X_ERM), paired ===")
        rows3 = []
        for dom in ["clean", PRIMARY, OOD]:
            W = df[df.domain == dom].pivot_table(index="run", columns="config", values="macro_f1")
            ps, rr = [], []
            for a, b in ARCH_PAIRS:
                if a in W and b in W:
                    d = (W[b] - W[a]).dropna()
                    ps.append(wil(W[b].dropna(), W[a].dropna()))
                    m, lo, hi = ci(d)
                    rr.append(dict(domain=dom, arch=a.split("_")[0], erm=W[a].mean(), pidr=W[b].mean(),
                                   delta=m, lo=lo, hi=hi, wins=int((d > 0).sum()), n=len(d)))
            for r_, pa in zip(rr, holm(ps)):
                r_["p_holm"] = pa
                rows3.append(r_)
        R3 = pd.DataFrame(rows3)
        say(R3.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        if len(R3):
            fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
            for ax, dom in zip(axs, [PRIMARY, OOD]):
                s = R3[R3.domain == dom].reset_index(drop=True)
                yy = np.arange(len(s))
                ax.hlines(yy, s.erm, s.pidr, color=pc.GRID, linewidth=3)
                ax.plot(s.erm, yy, "o", color=pc.INK2, markersize=8, label="ERM (no augmentation)")
                ax.plot(s.pidr, yy, "o", color=pc.PALETTE[0], markersize=8, label="+ PIDR")
                ax.set_yticks(yy)
                ax.set_yticklabels(s.arch)
                ax.set_xlabel("macro-F1")
                ax.set_title(dom)
            h_, l_ = axs[0].get_legend_handles_labels()
            fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
            fig.suptitle("Each architecture without (ERM) and with PIDR")
            fig.savefig(os.path.join(OUT, "fig_ext_arch_x_aug.png"))
            plt.close(fig)

        say("\n=== S4. Detection view: AUROC and operating points (threshold set on VALIDATION, target FA 5% / 10%) ===")
        for dom in [PRIMARY, OOD]:
            sub = df[df.domain == dom]
            t4 = pd.DataFrame({c: {
                "AUROC PD-vs-noPD": fmt(sub[sub.config == c].auroc_pd),
                "AUROC Corona-vs-Noise": fmt(sub[sub.config == c].auroc_corona),
                "FA @5% target": fmt(sub[sub.config == c].fa_at5),
                "recall @5%": fmt(sub[sub.config == c].recall_at5),
                "Corona det @5%": fmt(sub[sub.config == c].corona_at5),
                "FA @10% target": fmt(sub[sub.config == c].fa_at10),
                "recall @10%": fmt(sub[sub.config == c].recall_at10)} for c in cfgs}).T
            say(f"[{dom}]")
            say(t4.to_string())
        # ROC figure (pooled seed 0)
        from sklearn.metrics import roc_curve
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        show = [c for c in ["Tiny_ERM_raw", "Tiny_GenericAug", "ResNet_ERM", PROP] if c in cfgs]
        for ax, dom in zip(axs, [PRIMARY, OOD]):
            for i, c in enumerate(show):
                P, Y = [], []
                for f in sorted(df.fold.unique()):
                    fn = os.path.join(OUT, "scores", f"{c}__f{f}__s0.npz")
                    if os.path.isfile(fn):
                        z = np.load(fn)
                        P.append(z[dom])
                        Y.append(z["y"])
                if not P:
                    continue
                P, Y = np.concatenate(P), np.concatenate(Y)
                fpr, tpr, _ = roc_curve((Y != pc.NOISE_CLASS).astype(int), 1 - P[:, pc.NOISE_CLASS])
                ax.plot(fpr, tpr, color=pc.PALETTE[i], label=c)
            ax.plot([0, 1], [0, 1], color=pc.INK2, linewidth=1, linestyle=":")
            ax.axvline(0.05, color=pc.GRID, linewidth=1)
            ax.set_xlabel("false-alarm rate")
            ax.set_ylabel("PD detection rate")
            ax.set_title(dom)
            ax.legend(loc="lower right")
        fig.suptitle("PD-vs-no-PD ROC (pooled folds, seed 0)")
        fig.savefig(os.path.join(OUT, "fig_ext_roc.png"))
        plt.close(fig)

    # ---------------- hp
    p = os.path.join(OUT, "hp_runs.csv")
    if os.path.isfile(p):
        hp = pd.read_csv(p)
        say("\n=== S5. Hyper-parameter sensitivity (lr x epochs, 5 folds, seed 0) ===")
        g = hp.groupby(["config", "lr", "epochs"])[[f"test_{PRIMARY}", f"test_{OOD}", "test_clean",
                                                    "val_clean", "val_field"]].mean().reset_index()
        for c in g.config.unique():
            s = g[g.config == c]
            say(f"[{c}] mean field macro-F1 per setting (rows lr, cols epochs):")
            say(s.pivot(index="lr", columns="epochs", values=f"test_{PRIMARY}").round(3).to_string())
        say("\nSelection protocols -> resulting field macro-F1 (mean over folds):")
        sel = []
        for c in g.config.unique():
            s = g[g.config == c]
            r = dict(config=c,
                     worst=s[f"test_{PRIMARY}"].min(), best_oracle=s[f"test_{PRIMARY}"].max(),
                     chosen_by_clean_val=s.loc[s.val_clean.idxmax(), f"test_{PRIMARY}"],
                     chosen_by_field_val=s.loc[s.val_field.idxmax(), f"test_{PRIMARY}"],
                     spread=s[f"test_{PRIMARY}"].max() - s[f"test_{PRIMARY}"].min())
            sel.append(r)
        S5 = pd.DataFrame(sel)
        say(S5.round(3).to_string(index=False))
        if PROP in S5.config.values:
            worst_prop = float(S5[S5.config == PROP].worst.iloc[0])
            for _, r in S5[S5.config != PROP].iterrows():
                say(f"  worst {PROP} setting ({worst_prop:.3f}) vs ORACLE-best {r.config} ({r.best_oracle:.3f}): "
                    f"{'PIDR still better' if worst_prop > r.best_oracle else 'NOT better'}")
        fig, axs = plt.subplots(1, len(g.config.unique()), figsize=(4.2 * len(g.config.unique()), 3.6))
        axs = np.atleast_1d(axs)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
        for ax, c in zip(axs, g.config.unique()):
            M = g[g.config == c].pivot(index="lr", columns="epochs", values=f"test_{PRIMARY}")
            ax.imshow(M.values, cmap=cmap, vmin=0.3, vmax=0.9, aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(M.shape[1]))
            ax.set_xticklabels(M.columns)
            ax.set_yticks(range(M.shape[0]))
            ax.set_yticklabels([f"{v:g}" for v in M.index])
            ax.set_xlabel("epochs")
            ax.set_ylabel("learning rate")
            ax.set_title(c)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="#ffffff" if v > 0.7 else pc.INK)
        fig.suptitle(f"Hyper-parameter sensitivity, macro-F1 on {PRIMARY}")
        fig.savefig(os.path.join(OUT, "fig_ext_hp.png"))
        plt.close(fig)

    # ---------------- xai
    p = os.path.join(OUT, "xai.csv")
    if os.path.isfile(p):
        xa = pd.read_csv(p)
        say("\n=== S6. What does each model listen to? (fold 0, seed 0) ===")
        say("Gain sweep on clean test (macro-F1 when the whole input is scaled by x dB):")
        say(xa[xa.test == "gain"].pivot(index="model", columns="x", values="macro_f1").round(3).to_string())
        say("\nFrequency-band occlusion on field test: drop in macro-F1 when a band is removed (band centre kHz):")
        say(xa[xa.test == "band_field"].pivot(index="model", columns="x", values="drop").round(3).to_string())
        say("\nTime occlusion on clean test: drop in macro-F1 when a 40-us slice is replaced by floor noise (centre us):")
        say(xa[xa.test == "time_clean"].pivot(index="model", columns="x", values="drop").round(3).to_string())
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        mods = list(xa.model.unique())
        for i, mname in enumerate(mods):
            s = xa[(xa.test == "gain") & (xa.model == mname)]
            axs[0].plot(s.x, s.macro_f1, "o-", color=pc.PALETTE[i], markersize=6, label=mname)
            s = xa[(xa.test == "band_field") & (xa.model == mname)]
            axs[1].plot(s.x, s["drop"], "o-", color=pc.PALETTE[i], markersize=6, label=mname)
            s = xa[(xa.test == "time_clean") & (xa.model == mname)]
            axs[2].plot(s.x, s["drop"], "o-", color=pc.PALETTE[i], markersize=6, label=mname)
        axs[0].set_xlabel("input gain [dB]")
        axs[0].set_ylabel("macro-F1 (clean test)")
        axs[0].set_title("Loudness shortcut test")
        axs[1].set_xlabel("removed band centre [kHz]")
        axs[1].set_ylabel("macro-F1 drop (field test)")
        axs[1].set_title("Frequency reliance")
        axs[2].set_xlabel("occluded slice centre [µs]")
        axs[2].set_ylabel("macro-F1 drop (clean test)")
        axs[2].set_title("Timing reliance")
        axs[0].legend(loc="lower center")
        fig.savefig(os.path.join(OUT, "fig_xai.png"))
        plt.close(fig)

    with open(os.path.join(OUT, "EXTENDED_SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nSaved to {OUT}  ->  send EXTENDED_SUMMARY.txt")


# ============================================================================ run
if __name__ == "__main__":
    st = args.stage
    if st != "stats":
        X, y, row = pc.load_data(pc.find_data_dir(args.data_dir))
        folds = pc.blocked_purged_folds(y, row, args.folds, args.purge)
        print(f"device={device}  output={OUT}", flush=True)
        if st in ("all", "physics"):
            stage_physics(X, y)
        if st in ("all", "main"):
            stage_main(X, y, row, folds)
        if st in ("all", "hp"):
            stage_hp(X, y, row, folds)
        if st in ("all", "xai"):
            stage_xai(X, y, row, folds)
    stage_stats()
