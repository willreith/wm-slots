"""Cross-temporal LDA decoding and cosine RSA on single-object trials (Stroud-style)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from extract import load_single_object_trials, load_units, _ROOT

BIN_SIZE = 0.05  # stim onset -> delay end, 2.0 s window
T_END = 2.0


def bin_edges(bin_size=BIN_SIZE):
    return np.arange(0, T_END + bin_size / 2, bin_size)


BINS = bin_edges()
NBIN = len(BINS) - 1

WINDOWS = [(0.2, 0.5), (0.7, 1.0), (1.0, 1.3), (1.35, 1.65), (1.7, 2.0)]  # s from stim onset


def _window_slices(bin_size=BIN_SIZE):
    edges = bin_edges(bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    return [np.where((centers >= a) & (centers < b))[0] for a, b in WINDOWS]


def rate_tensor(session, region, quality="good", subject="Perle", bin_size=BIN_SIZE):
    """Return (rates [unit x trial x bin] Hz, observed [unit x trial] bool, trials df)."""
    edges = bin_edges(bin_size)
    nbin = len(edges) - 1
    trials = load_single_object_trials(session, subject)
    units, obs = load_units(session, subject)
    md = units.metadata
    keep = md["region"] == region
    if quality is not None:
        keep &= md["quality"] == quality
    uids = list(md.index[keep])
    obs_m = obs.loc[uids, trials.index].to_numpy()
    rates = np.empty((len(uids), len(trials), nbin))
    t0s = trials["stim_time"].to_numpy()
    for i, uid in enumerate(uids):
        spk = units[uid].t
        for j, t0 in enumerate(t0s):
            rates[i, j] = np.histogram(spk - t0, bins=edges)[0] / bin_size
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
    nU, nbin = rates.shape[0], rates.shape[2]
    X, yy = [], []
    for c in classes:
        pool = tr_idx[y[tr_idx] == c]
        block = np.zeros((n_pseudo, nU, nbin))
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


def _prep(session, region, factor, min_obs, data, subject="Perle", bin_size=BIN_SIZE):
    """Load/reuse the rate tensor, label trials, and drop units without enough obs per class."""
    rates, obs_m, trials = (data if data is not None
                            else rate_tensor(session, region, subject=subject, bin_size=bin_size))
    y = _labels(trials, factor)
    classes = np.unique(y)
    rates, obs_m, ok = _filter_units(rates, obs_m, y, classes, min_obs)
    return rates, obs_m, y, classes, ok


def decode(session, region, factor, n_pseudo_train=60, n_pseudo_test=30,
           n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None,
           subject="Perle", bin_size=BIN_SIZE, n_jobs=1):
    """Cross-temporal LDA accuracy (train-bin x test-bin), averaged over folds and repeats."""
    rates, obs_m, y, classes, ok = _prep(session, region, factor, min_obs, data, subject, bin_size)
    m = _run_folds(rates, obs_m, y, classes, n_folds, n_repeats,
                   n_pseudo_train, n_pseudo_test, "decode", seed, n_jobs)
    return m, int(ok.sum())


def _coding(X, y, classes):
    """Class-coding vectors per bin: class mean minus grand mean over classes. (nClass, nU, nbin)."""
    means = np.stack([X[y == c].mean(0) for c in classes])
    return means - means.mean(0, keepdims=True)


def _xtemp_cos(A, B):
    """Mean over classes of cosine(A[k][:,i], B[k][:,j]) -> (nbin, nbin)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.mean([An[k].T @ Bn[k] for k in range(A.shape[0])], axis=0)


def _fit_score(Xtr, ytr, Xte, yte, classes, kind):
    """One fold's cross-temporal matrix (LDA accuracy for 'decode', cosine for 'rsa')."""
    if kind == "decode":
        nbin = Xtr.shape[2]
        m = np.zeros((nbin, nbin))
        for i in range(nbin):
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Xtr[:, :, i], ytr)
            for j in range(nbin):
                m[i, j] = clf.score(Xte[:, :, j], yte)
        return m
    return _xtemp_cos(_coding(Xtr, ytr, classes), _coding(Xte, yte, classes))


def _one_fold(rates, obs_m, y, classes, tr, te, rng, n_tr, n_te, kind):
    """Build train/test pseudopopulations for one fold and return its cross-temporal matrix."""
    Xtr, ytr = _pseudo(rates, obs_m, tr, y, classes, n_tr, rng)
    Xte, yte = _pseudo(rates, obs_m, te, y, classes, n_te, rng)
    Xtr, Xte = _zscore(Xtr, Xte)
    return _fit_score(Xtr, ytr, Xte, yte, classes, kind)


def _splits(y, n_folds, n_repeats, seed):
    """List of (r, f, train_idx, test_idx) over repeats x folds."""
    idx = np.arange(len(y))
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(idx, y)):
            out.append((r, f, tr, te))
    return out


def _parallel(n_jobs):
    """Sequential for n_jobs=1, else loky with one BLAS thread per worker (avoid oversubscription)."""
    if n_jobs == 1:
        return parallel_config(n_jobs=1)
    return parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1)


def _run_folds(rates, obs_m, y, classes, n_folds, n_repeats, n_tr, n_te, kind, seed, n_jobs):
    """Run every (repeat, fold) in parallel; return the mean cross-temporal matrix."""
    tasks = _splits(y, n_folds, n_repeats, seed)
    with _parallel(n_jobs):
        mats = Parallel()(
            delayed(_one_fold)(rates, obs_m, y, classes, tr, te,
                               np.random.default_rng(np.random.SeedSequence([seed, r, f])),
                               n_tr, n_te, kind)
            for (r, f, tr, te) in tasks)
    return np.mean(mats, axis=0)


def rsa(session, region, factor, n_pseudo=60, n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None,
        subject="Perle", bin_size=BIN_SIZE, n_jobs=1):
    """Cross-validated cross-temporal cosine similarity of class-coding vectors (bin x bin)."""
    rates, obs_m, y, classes, _ = _prep(session, region, factor, min_obs, data, subject, bin_size)
    m = _run_folds(rates, obs_m, y, classes, n_folds, n_repeats,
                   n_pseudo, n_pseudo, "rsa", seed, n_jobs)
    return (m + m.T) / 2


def _principal_angles(Ba, Bb):
    """Mean principal angle (deg) between two orthonormal-row bases (2, nU)."""
    s = np.linalg.svd(Ba @ Bb.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)).mean())


def subspace_trajectories(session, region, factor, n_pseudo_train=60, n_pseudo_test=30,
                          n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None,
                          subject="Perle", bin_size=BIN_SIZE):
    """Per-window LDA coding subspaces (Xie Fig3D) and cross-validated projected trajectories.

    Fit shrinkage LDA on the window-averaged pseudopopulation for each WINDOW; the top-2 right
    singular vectors of the (nClass x nUnit) coef, accumulated over folds, give an orthonormal 2D
    coding plane per window. Project the held-out class-mean trajectories onto each plane.
    """
    rates, obs_m, y, classes, ok = _prep(session, region, factor, min_obs, data, subject, bin_size)
    wins = _window_slices(bin_size)
    nU, nbin = rates.shape[0], rates.shape[2]
    coef_sum = [np.zeros((len(classes), nU)) for _ in wins]
    traj_sum = np.zeros((len(classes), nU, nbin))
    tasks = _splits(y, n_folds, n_repeats, seed)
    for (r, f, tr, te) in tasks:
        rng = np.random.default_rng(np.random.SeedSequence([seed, r, f]))
        Xtr, ytr = _pseudo(rates, obs_m, tr, y, classes, n_pseudo_train, rng)
        Xte, yte = _pseudo(rates, obs_m, te, y, classes, n_pseudo_test, rng)
        Xtr, Xte = _zscore(Xtr, Xte)
        for w, sl in enumerate(wins):
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Xtr[:, :, sl].mean(2), ytr)
            coef_sum[w] += clf.coef_
        traj_sum += np.stack([Xte[yte == c].mean(0) for c in classes])
    n = len(tasks)
    bases = [np.linalg.svd(cs / n, full_matrices=False)[2][:2] for cs in coef_sum]
    traj = traj_sum / n
    proj = {w: np.stack([bases[w] @ traj[c] for c in range(len(classes))]) for w in range(len(wins))}
    angles = np.array([[_principal_angles(bases[i], bases[j]) for j in range(len(wins))]
                       for i in range(len(wins))])
    return dict(traj=proj, angles=angles, bases=bases, classes=classes, nU=int(ok.sum()))


# --- multi-session selection and aggregation ---

def _counts():
    return pd.read_csv(_ROOT / "session_counts.csv")


def select_sessions(region, min_mua=0, min_good=0, min_trials=0, subject=None, three_position=None):
    """Return [(subject, session), ...] meeting the criteria (region-scoped MUA/good counts)."""
    df = _counts()
    reg = region.lower()
    mua = df[f"{reg}_units"] - df[f"{reg}_good"]
    m = (mua >= min_mua) & (df[f"{reg}_good"] >= min_good) & (df["single_object_success"] >= min_trials)
    subj = df["subject"].str.replace("sub-", "", regex=False)
    if subject is not None:
        m &= subj == subject
    if three_position is not None:
        m &= df["three_position"] == three_position
    return list(zip(subj[m], df["session"][m]))


def _check_position(sessions, factor):
    if factor != "position":
        return
    df = _counts()
    subj = df["subject"].str.replace("sub-", "", regex=False)
    tp = {(s, ses): b for s, ses, b in zip(subj, df["session"], df["three_position"])}
    bad = [(s, ses) for s, ses in sessions if not tp.get((s, ses), False)]
    if bad:
        raise ValueError(f"position decoding needs 3-position sessions; not 3-position: {bad}")


def decode_group(sessions, region, factor, method="average", **kw):
    """Aggregate cross-temporal decoding over sessions. method='average' | 'pool'."""
    _check_position(sessions, factor)
    if method == "average":
        accs, per = [], []
        for subj, sess in sessions:
            acc, nU = decode(sess, region, factor, subject=subj, **kw)
            accs.append(acc)
            per.append((subj, sess, nU))
        return dict(mean=np.mean(accs, 0), per_session=per, n_sessions=len(accs))
    if method == "pool":
        return _pool(sessions, region, factor, "decode", **kw)
    raise ValueError(f"unknown method {method!r}")


def rsa_group(sessions, region, factor, method="average", **kw):
    """Aggregate cross-temporal RSA over sessions. method='average' | 'pool'."""
    _check_position(sessions, factor)
    if method == "average":
        sims, per = [], []
        for subj, sess in sessions:
            sims.append(rsa(sess, region, factor, subject=subj, **kw))
            per.append((subj, sess))
        return dict(mean=np.mean(sims, 0), per_session=per, n_sessions=len(sims))
    if method == "pool":
        return _pool(sessions, region, factor, "rsa", **kw)
    raise ValueError(f"unknown method {method!r}")


def _one_fold_pool(preps, classes, fold_splits, rng, n_tr, n_te, kind):
    """One fold of the pooled analysis: concatenate per-session pseudopopulations across units."""
    Xtr_p, Xte_p, ytr, yte = [], [], None, None
    for (rates, obs_m, y, _), (tr, te) in zip(preps, fold_splits):
        Xtr_s, ytr = _pseudo(rates, obs_m, tr, y, classes, n_tr, rng)
        Xte_s, yte = _pseudo(rates, obs_m, te, y, classes, n_te, rng)
        Xtr_p.append(Xtr_s)
        Xte_p.append(Xte_s)
    Xtr, Xte = np.concatenate(Xtr_p, axis=1), np.concatenate(Xte_p, axis=1)
    Xtr, Xte = _zscore(Xtr, Xte)
    return _fit_score(Xtr, ytr, Xte, yte, classes, kind)


def _pool(sessions, region, factor, kind, n_pseudo_train=60, n_pseudo_test=30, n_pseudo=60,
          n_repeats=20, n_folds=4, min_obs=4, seed=0, bin_size=BIN_SIZE, n_jobs=1):
    """One cross-session pseudopopulation (units concatenated) for decode or rsa."""
    n_tr = n_pseudo_train if kind == "decode" else n_pseudo
    n_te = n_pseudo_test if kind == "decode" else n_pseudo
    preps = [_prep(sess, region, factor, min_obs, None, subj, bin_size)[:4]
             for subj, sess in sessions]
    classes = preps[0][3]
    nU = sum(p[0].shape[0] for p in preps)
    tasks = []
    for r in range(n_repeats):
        per_sess = [list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
                         .split(np.arange(len(y)), y)) for (_, _, y, _) in preps]
        for f in range(n_folds):
            tasks.append((r, f, [per_sess[si][f] for si in range(len(preps))]))
    with _parallel(n_jobs):
        mats = Parallel()(
            delayed(_one_fold_pool)(preps, classes, fs,
                                    np.random.default_rng(np.random.SeedSequence([seed, r, f])),
                                    n_tr, n_te, kind)
            for (r, f, fs) in tasks)
    m = np.mean(mats, axis=0)
    if kind == "rsa":
        m = (m + m.T) / 2
    return dict(mean=m, nU=nU, n_sessions=len(sessions))
