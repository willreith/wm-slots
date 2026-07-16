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
    """Bin edges (s) tiling the analysis window, stim onset (0) to delay end (T_END = 2.0 s).

    Args:
        bin_size: bin width in seconds.
    Returns:
        1D array of len (T_END / bin_size) + 1 edges; the last is T_END.
    """
    return np.arange(0, T_END + bin_size / 2, bin_size)


BINS = bin_edges()
NBIN = len(BINS) - 1

WINDOWS = [(0.2, 0.5), (0.7, 1.0), (1.0, 1.3), (1.35, 1.65), (1.7, 2.0)]  # s from stim onset


def _window_slices(bin_size=BIN_SIZE):
    edges = bin_edges(bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    return [np.where((centers >= a) & (centers < b))[0] for a, b in WINDOWS]


def rate_tensor(session, region, quality="good", subject="Perle", bin_size=BIN_SIZE,
                correct_only=True):
    """Bin single-object-trial spikes into a per-unit firing-rate tensor for one session/region.

    Args:
        session: session date, e.g. '2022-06-01'.
        region: 'DMFC' (Neuropixel, right hemi) or 'FEF' (V-probe, left hemi).
        quality: unit quality to keep ('good'), or None to keep all units.
        subject: animal name, e.g. 'Perle'.
        bin_size: time bin width in seconds.
        correct_only: if True, keep only rewarded (correct) trials; False keeps correct + error.
    Returns:
        rates: float array [n_unit x n_trial x n_bin], firing rate in Hz, aligned to stim onset.
        observed: bool array [n_unit x n_trial], the per-trial observation mask (unobserved =
            missing, not silence).
        trials: the single-object trials DataFrame (rows aligned to the trial axis of rates).
    """
    edges = bin_edges(bin_size)
    nbin = len(edges) - 1
    trials = load_single_object_trials(session, subject)
    if correct_only:
        trials = trials[trials["correct"]]
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
    if factor == "identity":
        return trials["identity"].to_numpy()
    if factor == "position":
        return trials["position"].to_numpy()
    if factor == "conjunction":  # 3*position + identity code (a/b/c -> 0/1/2)
        return trials["position"].to_numpy() * 3 + pd.Categorical(trials["identity"]).codes
    raise ValueError(f"unknown factor {factor!r}")


def _sub_map(factor, subspace_factor, classes):
    """Map each (fine) class to the coarser LDA class that defines the coding plane.

    Keyed on the raw class value so string labels (identity 'a'/'b'/'c') work; int() is applied
    only to the conjunction codes, which are integers by construction.
    """
    if subspace_factor == factor:
        return {c: c for c in classes}
    if factor == "conjunction" and subspace_factor == "position":
        return {c: int(c) // 3 for c in classes}
    if factor == "conjunction" and subspace_factor == "identity":
        return {c: int(c) % 3 for c in classes}
    raise ValueError(f"cannot map {factor!r} onto {subspace_factor!r}")


def _filter_units(rates, obs_m, y, classes, min_obs):
    """Keep units observed in >= min_obs trials of every class."""
    ok = np.array([all((obs_m[u] & (y == c)).sum() >= min_obs for c in classes)
                   for u in range(rates.shape[0])])
    return rates[ok], obs_m[ok], ok


def _pseudo(rates, obs_m, tr_idx, y, classes, n_pseudo, rng):
    """Build (n_pseudo*n_class, n_unit, nbin) pseudo-trials + labels from trial indices tr_idx.

    Each pseudo-trial draws, per unit, a whole real trial (all n bins) from that unit's
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


def _prep(session, region, factor, min_obs, data, subject="Perle", bin_size=BIN_SIZE,
          correct_only=True):
    """Load/reuse the rate tensor, label trials, and drop units without enough obs per class."""
    rates, obs_m, trials = (data if data is not None
                            else rate_tensor(session, region, subject=subject, bin_size=bin_size,
                                             correct_only=correct_only))
    y = _labels(trials, factor)
    classes = np.unique(y)
    rates, obs_m, ok = _filter_units(rates, obs_m, y, classes, min_obs)
    return rates, obs_m, y, classes, ok


def decode(session, region, factor, n_pseudo_train=60, n_pseudo_test=30,
           n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None,
           subject="Perle", bin_size=BIN_SIZE, n_jobs=1, correct_only=True):
    """Cross-temporal shrinkage-LDA decoding accuracy, averaged over CV folds and pseudopop redraws.

    Trains one LDA per training bin on a class-balanced pseudopopulation and scores it at every
    test bin; the off-diagonal structure distinguishes a stable (filled square) from a rotating
    (diagonal band) code.

    Args:
        session, region, subject, bin_size, correct_only: see rate_tensor.
        factor: label to decode -- 'identity' (a/b/c), 'position' (0/1/2), or 'conjunction'
            (9-way position x identity); trials are marginalised over the other factor.
        n_pseudo_train: pseudo-trials drawn per class for the training pseudopopulation each fold.
        n_pseudo_test: pseudo-trials drawn per class for the test pseudopopulation each fold.
        n_repeats: independent pseudopopulation redraws; the returned matrix is their mean.
        n_folds: number of stratified cross-validation folds.
        min_obs: drop units observed in fewer than this many trials of any class.
        seed: base RNG seed; each (repeat, fold) derives a deterministic sub-seed from it.
        data: precomputed (rates, observed, trials) from rate_tensor to avoid reloading; None loads.
        n_jobs: parallel workers over the (repeat x fold) tasks; -1 = all cores.
    Returns:
        acc: float array [n_bin x n_bin], accuracy for (train bin i, test bin j); chance = 1/n_class.
        n_units: number of units surviving the min_obs filter.
    """
    rates, obs_m, y, classes, ok = _prep(session, region, factor, min_obs, data, subject, bin_size,
                                          correct_only)
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
        subject="Perle", bin_size=BIN_SIZE, n_jobs=1, correct_only=True):
    """Cross-validated cross-temporal cosine RSA of class-coding vectors (decoder-free complement).

    For each bin builds a per-class coding vector (class mean minus grand mean) and takes the
    cosine similarity between coding vectors at every pair of bins, averaged over classes. The two
    bins in a comparison come from independent CV folds, so the diagonal is not trivially 1.

    Args:
        session, region, subject, bin_size, correct_only: see rate_tensor.
        factor: label whose coding geometry is measured -- 'identity', 'position', or 'conjunction'.
        n_pseudo: pseudo-trials drawn per class for both the two independent pseudopopulations.
        n_repeats: independent pseudopopulation redraws; the returned matrix is their mean.
        n_folds: number of stratified cross-validation folds.
        min_obs: drop units observed in fewer than this many trials of any class.
        seed: base RNG seed; each (repeat, fold) derives a deterministic sub-seed from it.
        data: precomputed (rates, observed, trials) from rate_tensor to avoid reloading; None loads.
        n_jobs: parallel workers over the (repeat x fold) tasks; -1 = all cores.
    Returns:
        sim: symmetric float array [n_bin x n_bin] of cosine similarities between bins (1 = identical
            coding geometry, 0 = orthogonal / rotated).
    """
    rates, obs_m, y, classes, _ = _prep(session, region, factor, min_obs, data, subject, bin_size,
                                        correct_only)
    m = _run_folds(rates, obs_m, y, classes, n_folds, n_repeats,
                   n_pseudo, n_pseudo, "rsa", seed, n_jobs)
    return (m + m.T) / 2


def _principal_angles(Ba, Bb):
    """Mean principal angle (deg) between two orthonormal-row bases (2, nU)."""
    s = np.linalg.svd(Ba @ Bb.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)).mean())


def _accumulate_subspace(preps, sub_map, wins, n_tr, n_te, n_repeats, n_folds, seed):
    """Per-window coding planes + projected class-mean trajectories from one or many sessions.

    preps: list of (rates, obs_m, y) unit-filtered on the (fine) trajectory classes; pooled means
    the pseudopopulations are concatenated across units. LDA is fit on labels mapped through
    sub_map (e.g. conjunction -> position); trajectories are the held-out fine-class means.
    """
    traj_cls = np.unique(preps[0][2])
    sub_cls = sorted({sub_map[c] for c in traj_cls})
    nU = sum(p[0].shape[0] for p in preps)
    nbin = preps[0][0].shape[2]
    coef_sum = [np.zeros((len(sub_cls), nU)) for _ in wins]
    traj_sum = np.zeros((len(traj_cls), nU, nbin))
    ntask = 0
    for r in range(n_repeats):
        per_sess = [list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
                         .split(np.arange(len(y)), y)) for (_, _, y) in preps]
        for f in range(n_folds):
            rng = np.random.default_rng(np.random.SeedSequence([seed, r, f]))
            Xtr_p, Xte_p, ytr, yte = [], [], None, None
            for (rates, obs_m, y), splits in zip(preps, per_sess):
                tr, te = splits[f]
                Xtr_s, ytr = _pseudo(rates, obs_m, tr, y, traj_cls, n_tr, rng)
                Xte_s, yte = _pseudo(rates, obs_m, te, y, traj_cls, n_te, rng)
                Xtr_p.append(Xtr_s)
                Xte_p.append(Xte_s)
            Xtr, Xte = _zscore(np.concatenate(Xtr_p, 1), np.concatenate(Xte_p, 1))
            ytr_sub = np.array([sub_map[c] for c in ytr])
            for w, sl in enumerate(wins):
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                clf.fit(Xtr[:, :, sl].mean(2), ytr_sub)
                coef_sum[w] += clf.coef_
            traj_sum += np.stack([Xte[yte == c].mean(0) for c in traj_cls])
            ntask += 1
    bases = [np.linalg.svd(cs / ntask, full_matrices=False)[2][:2] for cs in coef_sum]
    traj = traj_sum / ntask
    proj = {w: np.stack([bases[w] @ traj[k] for k in range(len(traj_cls))]) for w in range(len(wins))}
    angles = np.array([[_principal_angles(bases[i], bases[j]) for j in range(len(wins))]
                       for i in range(len(wins))])
    return dict(traj=proj, angles=angles, bases=bases, classes=traj_cls, nU=nU)


def subspace_trajectories(session, region, factor, subspace_factor=None, n_pseudo_train=60,
                          n_pseudo_test=30, n_repeats=20, n_folds=4, min_obs=4, seed=0, data=None,
                          subject="Perle", bin_size=BIN_SIZE, correct_only=True):
    """Per-window LDA coding subspaces (Xie Fig3D) and cross-validated projected trajectories.

    Fit shrinkage LDA on the window-averaged pseudopopulation for each WINDOW; the top-2 right
    singular vectors of the (n_class x n_unit) coef, accumulated over folds, give an orthonormal 2D
    coding plane per window. The held-out class-mean trajectories are projected onto each plane, and
    principal angles between planes quantify how much the code rotates across windows.

    Args:
        session, region, subject, bin_size, correct_only: see rate_tensor.
        factor: label that defines the trajectory classes -- 'identity', 'position', or
            'conjunction' (the 9 position x identity means).
        subspace_factor: label whose LDA defines the coding plane (default = factor). E.g.
            factor='conjunction', subspace_factor='position' projects the 9 conjunction-mean
            trajectories onto the 3-class position plane.
        n_pseudo_train: pseudo-trials per class used to fit each window's LDA plane.
        n_pseudo_test: pseudo-trials per class used for the held-out projected trajectories.
        n_repeats: pseudopopulation redraws; planes and trajectories are accumulated over them.
        n_folds: number of stratified cross-validation folds.
        min_obs: drop units observed in fewer than this many trials of any class.
        seed: base RNG seed; each (repeat, fold) derives a deterministic sub-seed from it.
        data: precomputed (rates, observed, trials) from rate_tensor to avoid reloading; None loads.
    Returns:
        dict with:
            traj: {window_index -> array [n_class x 2 x n_bin]} trajectories in each window's plane.
            angles: float array [n_window x n_window], mean principal angle (deg) between planes
                (0 = identical plane, 90 = orthogonal).
            bases: list of [2 x n_unit] orthonormal plane bases, one per window.
            classes: the trajectory class labels (row order of traj).
            nU: number of units surviving the min_obs filter.
    """
    subspace_factor = subspace_factor or factor
    rates, obs_m, y, classes, ok = _prep(session, region, factor, min_obs, data, subject, bin_size,
                                         correct_only)
    res = _accumulate_subspace([(rates, obs_m, y)], _sub_map(factor, subspace_factor, classes),
                               _window_slices(bin_size), n_pseudo_train, n_pseudo_test,
                               n_repeats, n_folds, seed)
    return res


# --- multi-session selection and aggregation ---

def _counts():
    return pd.read_csv(_ROOT / "session_counts.csv")


def select_sessions(region, min_mua=0, min_good=0, min_trials=0, subject=None, three_position=None):
    """Select sessions from session_counts.csv meeting unit- and trial-count criteria.

    Args:
        region: 'DMFC' or 'FEF'; the MUA/good-unit thresholds are counted within this region.
        min_mua: minimum multi-unit count (region units minus good units) in the region.
        min_good: minimum good-unit count in the region.
        min_trials: minimum correct single-object trials (the `single_object_success` column, which
            equals the correct-trial pool used when correct_only=True).
        subject: restrict to this animal, or None for all.
        three_position: if True/False, restrict to the discrete-3-position / other-design sessions;
            None keeps both.
    Returns:
        List of (subject, session) tuples passing all criteria.
    """
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
    if factor not in ("position", "conjunction"):
        return
    df = _counts()
    subj = df["subject"].str.replace("sub-", "", regex=False)
    tp = {(s, ses): b for s, ses, b in zip(subj, df["session"], df["three_position"])}
    bad = [(s, ses) for s, ses in sessions if not tp.get((s, ses), False)]
    if bad:
        raise ValueError(f"position decoding needs 3-position sessions; not 3-position: {bad}")


def decode_group(sessions, region, factor, method="average", **kw):
    """Aggregate cross-temporal decoding over sessions.

    Args:
        sessions: list of (subject, session) tuples, e.g. from select_sessions.
        region: 'DMFC' or 'FEF'.
        factor: 'identity', 'position', or 'conjunction' ('position'/'conjunction' require
            3-position sessions).
        method: 'average' runs the full decode within each session and means the matrices (session =
            replicate); 'pool' concatenates units across sessions into one pseudopopulation and
            decodes once (max power).
        **kw: forwarded to decode (average) or _pool (pool) -- e.g. n_repeats, n_folds, min_obs,
            bin_size, seed, n_jobs, correct_only.
    Returns:
        method='average': dict(mean [n_bin x n_bin], per_session [(subject, session, n_units)],
            n_sessions).
        method='pool': dict(mean [n_bin x n_bin], nU total pooled units, n_sessions).
    """
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
    """Aggregate cross-temporal cosine RSA over sessions.

    Args:
        sessions: list of (subject, session) tuples, e.g. from select_sessions.
        region: 'DMFC' or 'FEF'.
        factor: 'identity', 'position', or 'conjunction' ('position'/'conjunction' require
            3-position sessions).
        method: 'average' runs the full RSA within each session and means the matrices (session =
            replicate); 'pool' concatenates units across sessions into one pseudopopulation.
        **kw: forwarded to rsa (average) or _pool (pool) -- e.g. n_pseudo, n_repeats, n_folds,
            min_obs, bin_size, seed, n_jobs, correct_only.
    Returns:
        method='average': dict(mean [n_bin x n_bin], per_session [(subject, session)], n_sessions).
        method='pool': dict(mean [n_bin x n_bin], nU total pooled units, n_sessions).
    """
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


def subspace_trajectories_group(sessions, region, factor, method="pool", subspace_factor=None,
                                n_pseudo_train=60, n_pseudo_test=30, n_repeats=20, n_folds=4,
                                min_obs=4, seed=0, bin_size=BIN_SIZE, correct_only=True):
    """Aggregate per-window coding subspaces over sessions.

    Args:
        sessions: list of (subject, session) tuples, e.g. from select_sessions.
        region: 'DMFC' or 'FEF'.
        factor: label defining the trajectory classes -- 'identity', 'position', or 'conjunction'
            ('position'/'conjunction' require 3-position sessions).
        method: 'pool' concatenates units across sessions into one pseudopopulation, giving a single
            basis + trajectories + angles; 'average' aggregates only the orientation-free principal-
            angle matrices (trajectories are not comparable across sessions' arbitrary orientations).
        subspace_factor: label whose LDA defines the coding plane (default = factor); see
            subspace_trajectories.
        n_pseudo_train: pseudo-trials per class used to fit each window's LDA plane.
        n_pseudo_test: pseudo-trials per class used for the held-out projected trajectories.
        n_repeats: pseudopopulation redraws accumulated into the planes/trajectories.
        n_folds: number of stratified cross-validation folds.
        min_obs: drop units observed in fewer than this many trials of any class.
        seed: base RNG seed; each (repeat, fold) derives a deterministic sub-seed from it.
        bin_size: time bin width in seconds.
        correct_only: if True, use only rewarded (correct) trials.
    Returns:
        method='pool': same dict as subspace_trajectories (traj, angles, bases, classes, nU) plus
            n_sessions.
        method='average': dict(angles_mean [n_window x n_window], angles_sem, per_session
            [(subject, session, n_units)], n_sessions).
    """
    subspace_factor = subspace_factor or factor
    _check_position(sessions, factor)
    if method == "pool":
        preps = [_prep(sess, region, factor, min_obs, None, subj, bin_size, correct_only)[:3]
                 for subj, sess in sessions]
        sub_map = _sub_map(factor, subspace_factor, np.unique(preps[0][2]))
        res = _accumulate_subspace(preps, sub_map, _window_slices(bin_size),
                                   n_pseudo_train, n_pseudo_test, n_repeats, n_folds, seed)
        res["n_sessions"] = len(sessions)
        return res
    if method == "average":
        angs, per = [], []
        for subj, sess in sessions:
            r = subspace_trajectories(sess, region, factor, subspace_factor, n_pseudo_train,
                                      n_pseudo_test, n_repeats, n_folds, min_obs, seed,
                                      subject=subj, bin_size=bin_size, correct_only=correct_only)
            angs.append(r["angles"])
            per.append((subj, sess, r["nU"]))
        angs = np.stack(angs)
        sem = angs.std(0, ddof=1) / np.sqrt(len(angs)) if len(angs) > 1 else np.zeros_like(angs[0])
        return dict(angles_mean=angs.mean(0), angles_sem=sem, per_session=per, n_sessions=len(angs))
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
          n_repeats=20, n_folds=4, min_obs=4, seed=0, bin_size=BIN_SIZE, n_jobs=1, correct_only=True):
    """One cross-session pseudopopulation (units concatenated) for decode or rsa."""
    n_tr = n_pseudo_train if kind == "decode" else n_pseudo
    n_te = n_pseudo_test if kind == "decode" else n_pseudo
    preps = [_prep(sess, region, factor, min_obs, None, subj, bin_size, correct_only)[:4]
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
