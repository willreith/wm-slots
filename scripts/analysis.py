"""Cross-temporal LDA decoding and cosine RSA on single-object trials (Stroud-style)."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from extract import load_single_object_trials, load_units

BINS = np.arange(0, 2.0001, 0.05)  # 20 x 100 ms, stim onset -> delay end
NBIN = len(BINS) - 1


def rate_tensor(session, region, quality="good"):
    """Return (rates [unit x trial x bin] Hz, observed [unit x trial] bool, trials df)."""
    trials = load_single_object_trials(session)
    units, obs = load_units(session)
    md = units.metadata
    keep = md["region"] == region
    if quality is not None:
        keep &= md["quality"] == quality
    uids = list(md.index[keep])
    obs_m = obs.loc[uids, trials.index].to_numpy()
    rates = np.empty((len(uids), len(trials), NBIN))
    t0s = trials["stim_time"].to_numpy()
    for i, uid in enumerate(uids):
        spk = units[uid].t
        for j, t0 in enumerate(t0s):
            rates[i, j] = np.histogram(spk - t0, bins=BINS)[0] / 0.1
    return rates, obs_m, trials


def _labels(trials, factor):
    return trials["identity"].to_numpy() if factor == "identity" else trials["position"].to_numpy()


def _filter_units(rates, obs_m, y, classes, min_obs):
    """Keep units observed in >= min_obs trials of every class."""
    ok = np.array([all((obs_m[u] & (y == c)).sum() >= min_obs for c in classes)
                   for u in range(rates.shape[0])])
    return rates[ok], obs_m[ok], ok


def _pseudo(rates, obs_m, tr_idx, y, classes, n_pseudo, rng):
    """Build (n_pseudo*n_class, n_unit, nbin) pseudo-trials + labels from trial indices tr_idx.

    Each pseudo-trial draws, per unit, a whole real trial (all 20 bins) from that unit's
    observed trials of the class, so per-unit temporal trajectories stay coherent.
    """
    nU = rates.shape[0]
    X, yy = [], []
    for c in classes:
        pool = tr_idx[y[tr_idx] == c]
        block = np.zeros((n_pseudo, nU, NBIN))
        for u in range(nU):
            avail = pool[obs_m[u, pool]]
            if len(avail) == 0:
                continue  # unobserved in this fold/class: leave at 0
            pick = rng.choice(avail, size=n_pseudo, replace=True)
            block[:, u, :] = rates[u, pick, :]
        X.append(block)
        yy.append(np.full(n_pseudo, c))
    return np.concatenate(X), np.concatenate(yy)


def _zscore(Xtr, Xte):
    mu = Xtr.mean((0, 2), keepdims=True)
    sd = Xtr.std((0, 2), keepdims=True)
    sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def decode(session, region, factor, n_pseudo_train=60, n_pseudo_test=30,
           n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None):
    """Cross-temporal LDA accuracy (train-bin x test-bin), averaged over folds and repeats."""
    rates, obs_m, trials = data if data is not None else rate_tensor(session, region)
    y = _labels(trials, factor)
    classes = np.unique(y)
    rates, obs_m, ok = _filter_units(rates, obs_m, y, classes, min_obs)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    acc = np.zeros((n_repeats, n_folds, NBIN, NBIN))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(idx, y)):
            Xtr, ytr = _pseudo(rates, obs_m, tr, y, classes, n_pseudo_train, rng)
            Xte, yte = _pseudo(rates, obs_m, te, y, classes, n_pseudo_test, rng)
            Xtr, Xte = _zscore(Xtr, Xte)
            for i in range(NBIN):
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                clf.fit(Xtr[:, :, i], ytr)
                for j in range(NBIN):
                    acc[r, f, i, j] = clf.score(Xte[:, :, j], yte)
    return acc.mean((0, 1)), int(ok.sum())


def _coding(X, y, classes):
    """Class-coding vectors per bin: class mean minus grand mean over classes. (nClass, nU, nbin)."""
    means = np.stack([X[y == c].mean(0) for c in classes])
    return means - means.mean(0, keepdims=True)


def _xtemp_cos(A, B):
    """Mean over classes of cosine(A[k][:,i], B[k][:,j]) -> (nbin, nbin)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.mean([An[k].T @ Bn[k] for k in range(A.shape[0])], axis=0)


def rsa(session, region, factor, n_pseudo=60, n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None):
    """Cross-validated cross-temporal cosine similarity of class-coding vectors (bin x bin)."""
    rates, obs_m, trials = data if data is not None else rate_tensor(session, region)
    y = _labels(trials, factor)
    classes = np.unique(y)
    rates, obs_m, _ = _filter_units(rates, obs_m, y, classes, min_obs)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    sim = np.zeros((n_repeats, n_folds, NBIN, NBIN))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(idx, y)):
            Xtr, ytr = _pseudo(rates, obs_m, tr, y, classes, n_pseudo, rng)
            Xte, yte = _pseudo(rates, obs_m, te, y, classes, n_pseudo, rng)
            Xtr, Xte = _zscore(Xtr, Xte)
            sim[r, f] = _xtemp_cos(_coding(Xtr, ytr, classes), _coding(Xte, yte, classes))
    m = sim.mean((0, 1))
    return (m + m.T) / 2
