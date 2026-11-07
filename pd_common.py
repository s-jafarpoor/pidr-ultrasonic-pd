"""
pd_common.py  -  shared core for the PIDR-PD project
=====================================================
"From Anechoic Chamber to Substation: Physics-Informed Domain Randomization
 for deployable airborne-ultrasonic partial-discharge classification"

Contents
  1. Data loading (4 CSV files, calibrated pascals)
  2. Leakage-safe blocked + purged cross-validation
  3. ISO 9613-1 atmospheric absorption (frequency-, temperature-, humidity-dependent)
  4. FieldSimulator: a differentiable-free, batched, physics-based "acoustic digital twin"
     that turns a 1 m anechoic-chamber waveform into a field-like observation:
        spherical spreading + atmospheric absorption  (distance d, T, RH)
        multipath / reverberation                     (direct-to-reverberant ratio)
        loss of the HV-side trigger                   (random event position)
        sensor transfer-function mismatch             (MEMS-like resonance, tilt, gain)
        industrial noise at an ABSOLUTE level         (coloured noise + impulsive clicks)
        lab noise-floor restoration + 25-100 kHz band-pass (same as the original chain)
  5. Evaluation domains (fixed, seeded, identical for every model -> paired statistics)
  6. TinyPDNet (1-D CNN, ~25k params, edge-deployable) with optional distance head
  7. Hand-crafted features for classical baselines and the shortcut probe
  8. Metrics (macro-F1, balanced accuracy, ECE, distance error)
"""
import os
import sys
import math
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                    # Windows consoles: avoid UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------- constants
FS = 500_000          # sampling rate [Hz]
N = 125               # samples per waveform (250 us)
NFFT = 256            # zero-padded FFT length (avoids circular wrap)
CLASS_FILES = {0: "PD_in_Air_12kV.csv", 1: "PD_in_Void_25kV.csv",
               2: "Corona_11kV.csv", 3: "Noise_NoPD.csv"}
CLASS_NAMES = ["Air", "Void", "Corona", "Noise"]
NOISE_CLASS = 3
N_CLASSES = 4


# ============================================================================ 1. data
def find_data_dir(user_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [user_dir, here, os.path.join(here, "dataset"), os.getcwd(),
             os.path.join(os.getcwd(), "dataset")]
    for c in cands:
        if c and all(os.path.isfile(os.path.join(c, f)) for f in CLASS_FILES.values()):
            return c
    raise FileNotFoundError(
        "Could not find the 4 CSV files. Put this script next to them or pass --data_dir. "
        f"Looked in: {[c for c in cands if c]}")


def load_data(data_dir):
    """Returns X (4000,125) float32 [Pa], y (4000,), row (4000,) = row index inside its file."""
    import pandas as pd
    Xs, ys, rows = [], [], []
    for c, fn in CLASS_FILES.items():
        path = os.path.join(data_dir, fn)
        with open(path, "r", encoding="utf-8-sig") as fh:
            first = fh.readline()
        sep = ";" if first.count(";") > first.count(",") else ","      # files use ';' separators
        dec = "," if sep == ";" and "." not in first and "," in first else "."
        a = pd.read_csv(path, header=None, sep=sep, decimal=dec).dropna(axis=1, how="all")
        a = a.values.astype(np.float32)
        if a.shape[1] != N:
            raise ValueError(f"{fn}: expected {N} columns, got {a.shape[1]}")
        Xs.append(a)
        ys.append(np.full(len(a), c, dtype=np.int64))
        rows.append(np.arange(len(a)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(rows)


# ============================================================================ 2. CV
def blocked_purged_folds(y, row, k=5, purge=10):
    """Each class file is cut into k CONTIGUOUS blocks (recording order is preserved).
    Fold f tests on block f of every class; training excludes `purge` rows on each side
    of the test block, so temporally adjacent (possibly correlated) waveforms never leak."""
    folds = []
    for f in range(k):
        tr, te = [], []
        for c in range(N_CLASSES):
            ids = np.where(y == c)[0]
            ids = ids[np.argsort(row[ids])]
            n = len(ids)
            edges = np.linspace(0, n, k + 1).astype(int)
            lo, hi = edges[f], edges[f + 1]
            pos = np.arange(n)
            te.append(ids[lo:hi])
            tr.append(ids[(pos < lo - purge) | (pos >= hi + purge)])
        folds.append((np.concatenate(tr), np.concatenate(te)))
    return folds


def random_folds(y, k=5, seed=0):
    """Ordinary stratified random CV (used only in the audit, to show the optimism gap)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(a, b) for a, b in skf.split(np.zeros(len(y)), y)]


# ============================================================================ 3. physics
def iso9613_alpha(f, T_c, rh, pa_kpa=101.325):
    """ISO 9613-1:1993 pure-tone atmospheric absorption coefficient in dB/m.
    f [Hz], T_c [deg C], rh [%]. Works with numpy arrays or torch tensors (broadcasting)."""
    lib = torch if isinstance(f, torch.Tensor) or isinstance(T_c, torch.Tensor) else np
    pr = 101.325
    T = T_c + 273.15
    T0, T01 = 293.15, 273.16
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    h = rh * (10.0 ** C) / (pa_kpa / pr)                     # molar concentration of H2O [%]
    frO = (pa_kpa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa_kpa / pr) * (T / T0) ** (-0.5) * (
        9.0 + 280.0 * h * lib.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    f2 = f ** 2
    return 8.686 * f2 * (1.84e-11 * (pa_kpa / pr) ** (-1) * (T / T0) ** 0.5 + (T / T0) ** (-2.5) * (
        0.01275 * lib.exp(-2239.1 / T) / (frO + f2 / frO) +
        0.1068 * lib.exp(-3352.0 / T) / (frN + f2 / frN)))


def bandpass_mask(freqs, f1=20e3, f2=25e3, f3=100e3, f4=110e3):
    """Zero-phase raised-cosine band-pass (25-100 kHz pass band), mimics the original chain."""
    m = torch.zeros_like(freqs)
    up = (freqs >= f1) & (freqs < f2)
    m[up] = 0.5 - 0.5 * torch.cos(math.pi * (freqs[up] - f1) / (f2 - f1))
    m[(freqs >= f2) & (freqs <= f3)] = 1.0
    dn = (freqs > f3) & (freqs <= f4)
    m[dn] = 0.5 + 0.5 * torch.cos(math.pi * (freqs[dn] - f3) / (f4 - f3))
    return m


# ============================================================================ 4. simulator
def _u(gen, B, lo, hi, device, log=False):
    r = torch.rand(B, generator=gen, device=device)
    if log:
        return torch.exp(math.log(lo) + r * (math.log(hi) - math.log(lo)))
    return lo + r * (hi - lo)


def _bern(gen, B, p, device):
    return torch.rand(B, generator=gen, device=device) < p


class FieldSimulator:
    """Batched physics-based lab->field transform.  spec = dict (see TRAIN_SPEC)."""

    def __init__(self, noise_bank, floor_rms, device="cpu"):
        self.device = device
        self.bank = torch.as_tensor(noise_bank, dtype=torch.float32, device=device)
        self.floor_rms = float(floor_rms)
        self.freqs = torch.fft.rfftfreq(NFFT, 1.0 / FS).to(device)            # (129,)
        self.bp = bandpass_mask(self.freqs)
        self.fsafe = self.freqs.clamp(min=1e3)

    def _bed(self, B, gen):
        i = torch.randint(0, self.bank.shape[0], (B,), generator=gen, device=self.device)
        return self.bank[i]

    def __call__(self, x, spec, gen):
        """x: (B,N) tensor on device.  Returns (x_out, logd, prop_mask)."""
        dev = self.device
        B = x.shape[0]
        logd = torch.zeros(B, device=dev)
        if spec is None:
            return x, logd, torch.zeros(B, dtype=torch.bool, device=dev)
        if spec.get("generic"):
            return self._generic(x, spec["generic"], gen), logd, torch.zeros(B, dtype=torch.bool, device=dev)

        t = ~_bern(gen, B, spec.get("p_clean", 0.0), dev)                 # transformed samples
        if not t.any():
            return x, logd, torch.zeros(B, dtype=torch.bool, device=dev)
        xt = x[t]
        b = xt.shape[0]
        Fq = self.freqs[None, :]

        # ---- stage A: propagation + multipath (frequency domain)
        X = torch.fft.rfft(F.pad(xt, (0, NFFT - N)), n=NFFT)
        H = torch.ones(b, Fq.shape[1], device=dev)
        pm = torch.zeros(b, dtype=torch.bool, device=dev)
        p = spec.get("prop")
        if p and p.get("on"):
            pm = _bern(gen, b, p.get("p", 1.0), dev)
            d = torch.where(pm, _u(gen, b, *p["d"], dev, log=True), torch.ones(b, device=dev))
            T = _u(gen, b, *p["T"], dev)
            RH = _u(gen, b, *p["RH"], dev)
            Hd = (1.0 / d)[:, None].expand(b, Fq.shape[1]).clone()
            if p.get("absorption", True):
                alpha = iso9613_alpha(Fq, T[:, None], RH[:, None])          # dB/m
                Hd = Hd * 10.0 ** (-alpha * (d[:, None] - 1.0) / 20.0)
            H = H * Hd
            ld = torch.log(d)
        else:
            ld = torch.zeros(b, device=dev)
        r = spec.get("reverb")
        if r and r.get("on"):
            rm = _bern(gen, b, r.get("p", 1.0), dev)
            drr = _u(gen, b, *r["drr"], dev)
            tau = _u(gen, b, *r["tau_us"], dev) * 1e-6 * FS                  # samples
            n = torch.arange(1, N, device=dev, dtype=torch.float32)[None, :]
            w = torch.randn(b, N - 1, generator=gen, device=dev) * torch.exp(-n / tau[:, None])
            w = w / w.pow(2).sum(1, keepdim=True).sqrt().clamp(min=1e-12)
            w = w * (10.0 ** (-drr / 20.0))[:, None]                          # reverb/direct energy
            h = torch.zeros(b, NFFT, device=dev)
            h[:, 0] = 1.0
            h[:, 1:N] = w
            Hr = torch.fft.rfft(h, n=NFFT)
            Hr = torch.where(rm[:, None], Hr, torch.ones_like(Hr))
            X = X * Hr
        X = X * H
        z = torch.fft.irfft(X, n=NFFT)[:, :N]

        # ---- lab floor restoration (propagation also attenuated the floor; put it back)
        if spec.get("floor_bed", True):
            z = z + self._bed(b, gen)

        # ---- loss of trigger: event at a random position, gaps filled with real floor
        s = spec.get("shift")
        if s and s.get("on"):
            sm = _bern(gen, b, s.get("p", 1.0), dev)
            K = int(s["max"])
            k = torch.randint(-K, K + 1, (b,), generator=gen, device=dev)
            k = torch.where(sm, k, torch.zeros_like(k))
            long = torch.cat([self._bed(b, gen), z, self._bed(b, gen)], dim=1)   # (b, 3N)
            idx = (N - k)[:, None] + torch.arange(N, device=dev)[None, :]
            z = torch.gather(long, 1, idx)

        # ---- industrial noise at an absolute level (dB re. lab floor RMS)
        nz = spec.get("noise")
        if nz and nz.get("on"):
            nm = _bern(gen, b, nz.get("p", 1.0), dev)
            L = _u(gen, b, *nz["level"], dev)
            beta = _u(gen, b, *nz["beta"], dev)
            W = torch.complex(torch.randn(b, Fq.shape[1], generator=gen, device=dev),
                              torch.randn(b, Fq.shape[1], generator=gen, device=dev))
            W = W * (self.fsafe[None, :] / 50e3) ** (-beta[:, None] / 2.0) * self.bp[None, :]
            col = torch.fft.irfft(W, n=NFFT)[:, :N]
            col = col / col.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
            col = col * (self.floor_rms * 10.0 ** (L / 20.0))[:, None]
            z = z + col * nm[:, None]
            pi = nz.get("p_imp", 0.0)
            if pi > 0:
                im = _bern(gen, b, pi, dev)
                amp = self.floor_rms * 10.0 ** (_u(gen, b, *nz["imp"], dev) / 20.0)
                nimp = torch.randint(1, 4, (b,), generator=gen, device=dev)
                train_ = torch.zeros(b, NFFT, device=dev)
                for j in range(3):
                    pos = torch.randint(0, N, (b,), generator=gen, device=dev)
                    sign = torch.where(_bern(gen, b, 0.5, dev), 1.0, -1.0)
                    val = sign * amp * (j < nimp).float() * im.float()
                    train_.scatter_add_(1, pos[:, None], val[:, None])
                f0 = _u(gen, b, 30e3, 90e3, dev)
                tk = _u(gen, b, 2.0, 6.0, dev)
                nn_ = torch.arange(16, device=dev, dtype=torch.float32)[None, :]
                ker = torch.exp(-nn_ / tk[:, None]) * torch.cos(2 * math.pi * f0[:, None] * nn_ / FS)
                kpad = F.pad(ker, (0, NFFT - 16))
                clicks = torch.fft.irfft(torch.fft.rfft(train_) * torch.fft.rfft(kpad) * self.bp[None, :],
                                         n=NFFT)[:, :N]
                z = z + clicks

        # ---- stage B: sensor transfer function.  NOTE: no extra band-pass on the signal path:
        # the recordings are already chain-filtered and the added noise/clicks are band-limited
        # at generation, so a "transformed vs untouched" filtering artefact cannot leak.
        se = spec.get("sensor")
        if se and se.get("on"):
            Z = torch.fft.rfft(F.pad(z, (0, NFFT - N)), n=NFFT)
            sem = _bern(gen, b, se.get("p", 1.0), dev)
            fc = _u(gen, b, *se["fc"], dev) * 1e3
            Q = _u(gen, b, *se["q"], dev)
            Gp = _u(gen, b, *se["peak"], dev)
            tilt = _u(gen, b, *se["tilt"], dev)
            gain = _u(gen, b, *se["gain"], dev)
            fr = self.fsafe[None, :] / fc[:, None]
            bell = Gp[:, None] / (1.0 + (Q[:, None] * (fr - 1.0 / fr)) ** 2)
            fband = self.freqs.clamp(25e3, 100e3)[None, :]          # tilt is flat outside the band
            slope = tilt[:, None] / 2.0 * torch.log2(fband / 50e3)   # dB, -tilt/2 @25k ... +tilt @100k
            Hs = 10.0 ** ((bell + slope + gain[:, None]) / 20.0)
            Hs = torch.where(sem[:, None], Hs, torch.ones_like(Hs))
            z = torch.fft.irfft(Z * Hs, n=NFFT)[:, :N]

        out = x.clone()
        out[t] = z
        logd[t] = torch.where(pm, ld, torch.zeros_like(ld))
        pmask = torch.zeros(B, dtype=torch.bool, device=dev)
        pmask[t] = pm
        return out, logd, pmask

    def _generic(self, x, g, gen):
        """Generic (non-physical) augmentation baseline: gain, white noise, circular shift."""
        dev = self.device
        B = x.shape[0]
        gain = 10.0 ** (_u(gen, B, -10, 10, dev) / 20.0)
        sd = x.std(1, keepdim=True)
        sig = _u(gen, B, 0.0, g.get("sigma", 0.5), dev)[:, None]
        z = x * gain[:, None] + torch.randn(x.shape, generator=gen, device=dev) * sd * sig * gain[:, None]
        k = torch.randint(-45, 46, (B,), generator=gen, device=dev)
        idx = (torch.arange(N, device=dev)[None, :] - k[:, None]) % N
        return torch.gather(z, 1, idx)


# ---------------------------------------------------------------- training distribution
TRAIN_SPEC = dict(
    p_clean=0.15,
    prop=dict(on=True, p=0.9, d=(1.0, 8.0), T=(0.0, 35.0), RH=(20.0, 90.0), absorption=True),
    reverb=dict(on=True, p=0.5, drr=(0.0, 20.0), tau_us=(20.0, 200.0)),
    shift=dict(on=True, p=0.7, max=45),
    sensor=dict(on=True, p=0.7, fc=(25.0, 70.0), q=(0.7, 3.0), peak=(0.0, 10.0),
                tilt=(-6.0, 6.0), gain=(-10.0, 10.0)),
    noise=dict(on=True, p=0.7, level=(-10.0, 15.0), beta=(0.0, 2.0), p_imp=0.3, imp=(5.0, 25.0)),
    floor_bed=True,
)
GENERIC_SPEC = dict(generic=dict(sigma=0.5))


def ablate(spec, what):
    import copy
    s = copy.deepcopy(spec)
    if what == "absorption":
        s["prop"]["absorption"] = False
    elif what == "distance":
        s["prop"]["on"] = False
    else:
        s[what]["on"] = False
    return s


# ============================================================================ 5. eval domains
def _fx(v):
    return (float(v), float(v))


def _base(**kw):
    s = dict(p_clean=0.0, floor_bed=True)
    s.update(kw)
    return s


def eval_domains():
    D = {}
    D["clean"] = None
    D["lab_floor_x2"] = _base()
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        D[f"dist_{d:02d}m"] = _base(prop=dict(on=True, p=1, d=_fx(d), T=_fx(20), RH=_fx(50)))
    D["untriggered"] = _base(shift=dict(on=True, p=1, max=55))
    D["reverb_drr3"] = _base(reverb=dict(on=True, p=1, drr=_fx(3), tau_us=(50, 150)))
    D["sensor_mems"] = _base(sensor=dict(on=True, p=1, fc=(30, 60), q=(1, 3), peak=_fx(6),
                                         tilt=(-6, 6), gain=(-10, 10)))
    for L in [0, 5, 10, 15, 20, 25, 30]:
        D[f"noise_{L:02d}dB"] = _base(noise=dict(on=True, p=1, level=_fx(L), beta=(0, 2), p_imp=0.0))
    D["impulsive"] = _base(noise=dict(on=True, p=1, level=_fx(-10), beta=(0, 2), p_imp=1.0, imp=(15, 25)))
    D["field_composite"] = _base(
        # calibrated on the audit: corona is only ~13 dB above the lab floor at 1 m,
        # so a realistic inspection envelope is 1-5 m and moderate background noise
        prop=dict(on=True, p=1, d=(1, 5), T=(0, 35), RH=(20, 90)),
        reverb=dict(on=True, p=1, drr=(0, 20), tau_us=(20, 200)),
        shift=dict(on=True, p=1, max=45),
        sensor=dict(on=True, p=1, fc=(25, 70), q=(0.7, 3), peak=(0, 10), tilt=(-6, 6), gain=(-10, 10)),
        noise=dict(on=True, p=1, level=(-10, 10), beta=(0, 2), p_imp=0.3, imp=(5, 25)))
    D["field_extreme_OOD"] = _base(
        # every factor partly OUTSIDE the training ranges, but still physically detectable
        prop=dict(on=True, p=1, d=(4, 10), T=(-10, 40), RH=(10, 100)),
        reverb=dict(on=True, p=1, drr=(-3, 10), tau_us=(100, 300)),
        shift=dict(on=True, p=1, max=60),
        sensor=dict(on=True, p=1, fc=(20, 80), q=(2, 5), peak=(6, 12), tilt=(-9, 9), gain=(-15, 15)),
        noise=dict(on=True, p=1, level=(5, 20), beta=(0, 2.5), p_imp=0.5, imp=(15, 30)))
    return D


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def build_eval_sets(X_te, sim, fold, device):
    """Deterministic transformed copies of the test fold, identical for every model."""
    out = {}
    xt = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    for name, spec in eval_domains().items():
        if spec is None:
            out[name] = (xt.clone(), torch.zeros(len(xt), device=device))
            continue
        g = torch.Generator(device=device)
        g.manual_seed(seed_of("eval", fold, name))
        z, ld, _ = sim(xt, spec, g)
        out[name] = (z, ld)
    return out


# ============================================================================ 6. model
def _cbr(ci, co, k, s):
    return nn.Sequential(nn.Conv1d(ci, co, k, s, padding=k // 2, bias=False),
                         nn.BatchNorm1d(co), nn.ReLU(inplace=True))


class TinyPDNet(nn.Module):
    """Edge-sized 1-D CNN. Per-waveform standardisation removes absolute-amplitude
    shortcuts (unknown distance / gain in the field); global avg+max pooling gives
    approximate translation invariance (no trigger in the field)."""

    def __init__(self, width=16, normalize=True, global_scale=1.0, n_classes=N_CLASSES):
        super().__init__()
        c = width
        self.normalize = normalize
        self.register_buffer("gscale", torch.tensor(float(global_scale)))
        self.body = nn.Sequential(_cbr(1, c, 7, 1), _cbr(c, 2 * c, 5, 2),
                                  _cbr(2 * c, 4 * c, 5, 2), _cbr(4 * c, 4 * c, 3, 1))
        self.drop = nn.Dropout(0.2)
        self.cls = nn.Linear(8 * c, n_classes)
        self.dist = nn.Linear(8 * c, 1)

    def forward(self, x):
        if self.normalize:
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
        else:
            x = x / self.gscale
        h = self.body(x[:, None, :])
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        h = self.drop(h)
        return self.cls(h), self.dist(h).squeeze(1)


def count_macs(width=16):
    c = width
    L1, L2, L3, L4 = N, (N + 1) // 2, ((N + 1) // 2 + 1) // 2, ((N + 1) // 2 + 1) // 2
    return (1 * c * 7 * L1 + c * 2 * c * 5 * L2 + 2 * c * 4 * c * 5 * L3 + 4 * c * 4 * c * 3 * L4
            + 8 * c * (N_CLASSES + 1))


# ============================================================================ 7. features
FEATURE_NAMES = ["log_rms", "log_peak", "peak_idx", "crest", "kurtosis", "skew", "zcr",
                 "t_centroid", "t_spread", "f_centroid", "f_bandwidth", "f_rolloff85",
                 "f_flatness", "band25_40", "band40_55", "band55_70", "band70_85", "band85_100"]
SHORTCUT_IDX = [0, 1, 2]          # log_rms, log_peak, peak_idx  -> the "shortcut probe"


def features(X):
    from scipy.stats import kurtosis, skew
    X = np.asarray(X, dtype=np.float64)
    eps = 1e-20
    rms = np.sqrt((X ** 2).mean(1)) + eps
    pk = np.abs(X).max(1) + eps
    pidx = np.abs(X).argmax(1) / N
    e = X ** 2
    t = np.arange(N)[None, :]
    es = e.sum(1) + eps
    tc = (e * t).sum(1) / es
    ts = np.sqrt((e * (t - tc[:, None]) ** 2).sum(1) / es) / N
    zcr = (np.diff(np.sign(X), axis=1) != 0).mean(1)
    S = np.abs(np.fft.rfft(X, n=NFFT)) ** 2
    f = np.fft.rfftfreq(NFFT, 1 / FS)[None, :]
    band = (f >= 25e3) & (f <= 100e3)
    Sb = S * band
    tot = Sb.sum(1) + eps
    fc = (Sb * f).sum(1) / tot
    fbw = np.sqrt((Sb * (f - fc[:, None]) ** 2).sum(1) / tot)
    cs = np.cumsum(Sb, 1) / tot[:, None]
    roll = f[0, np.argmax(cs >= 0.85, axis=1)]
    Sin = S[:, band[0]] + eps
    flat = np.exp(np.log(Sin).mean(1)) / Sin.mean(1)
    bands = []
    for lo, hi in [(25e3, 40e3), (40e3, 55e3), (55e3, 70e3), (70e3, 85e3), (85e3, 100e3)]:
        m = (f >= lo) & (f < hi)
        bands.append((S * m).sum(1) / tot)
    out = np.column_stack([np.log10(rms), np.log10(pk), pidx, pk / rms, kurtosis(X, axis=1),
                           skew(X, axis=1), zcr, tc / N, ts, fc / 1e3, fbw / 1e3, roll / 1e3,
                           flat] + bands)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================ 8. metrics
def ece_score(prob, y, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def all_metrics(y, pred, prob=None):
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
    out = dict(macro_f1=f1_score(y, pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0),
               bal_acc=balanced_accuracy_score(y, pred), acc=accuracy_score(y, pred))
    # PD-vs-noPD detection view (what a maintenance crew needs first)
    yb, pb = (y != NOISE_CLASS), (pred != NOISE_CLASS)
    out["det_recall"] = float((pb & yb).sum() / max(yb.sum(), 1))
    out["det_false_alarm"] = float((pb & ~yb).sum() / max((~yb).sum(), 1))
    out["ece"] = ece_score(prob, y) if prob is not None else np.nan
    out["cm"] = " ".join(map(str, confusion_matrix(y, pred, labels=list(range(N_CLASSES))).ravel()))
    return out


def get_device(pref="auto"):
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================================ 9. plotting
# fixed categorical order (validated palette), never cycled; neutral grey for "no PD"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = [PALETTE[0], PALETTE[1], PALETTE[2], "#8a8984"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "figure.dpi": 110,
        "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt
