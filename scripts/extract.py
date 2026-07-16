"""Extract single-object trials and spikes from the Watters WM dataset."""
import ast
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pynapple as nap
import lindi

_ROOT = Path(__file__).resolve().parents[1]
SESSION = "2022-05-28"
SUBJECT = "Perle"
_DANDISET, _VERSION = "000620", "0.260127.2208"
_local_cache = lindi.LocalCache()
_ASSETS = None


def _behav(session=SESSION, subject=SUBJECT):
    return _ROOT / f"data/{subject}/behav/sub-{subject}_ses-{session}_behavior+task.nwb"


def _spikes(session=SESSION, subject=SUBJECT):
    return _ROOT / f"data/{subject}/ephys/spikes/sub-{subject}_ses-{session}_spikesorting.nwb"


def _asset_index():
    """{(subject, session): {'behav': asset_id, 'spikes': asset_id}} from the DANDI API (cached)."""
    global _ASSETS
    if _ASSETS is None:
        url = (f"https://api.dandiarchive.org/api/dandisets/{_DANDISET}"
               f"/versions/{_VERSION}/assets/?page_size=1000")
        idx = {}
        while url:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.load(r)
            for a in d["results"]:
                m = re.search(r"sub-([^/_]+)_ses-([0-9-]+)", a["path"])
                if not m:
                    continue
                key = (m.group(1), m.group(2))
                if a["path"].endswith("behavior+task.nwb"):
                    idx.setdefault(key, {})["behav"] = a["asset_id"]
                elif a["path"].endswith("spikesorting.nwb"):
                    idx.setdefault(key, {})["spikes"] = a["asset_id"]
            url = d.get("next")
        _ASSETS = idx
    return _ASSETS


def _read(path, kind, session, subject):
    """Read an NWB file from disk if present, else stream it from DANDI via lindi."""
    if path.exists():
        return pynwb.NWBHDF5IO(str(path), mode="r").read()
    aid = _asset_index()[(subject, session)][kind]
    url = f"https://api.dandiarchive.org/api/assets/{aid}/download/"
    f = lindi.LindiH5pyFile.from_hdf5_file(url, local_cache=_local_cache)
    return pynwb.NWBHDF5IO(file=f, mode="r").read()


def _region_of(egroup, ml):
    """DMFC = Neuropixel in right hemi (ml>0); FEF = V-probe in left hemi (ml<0); else other."""
    npx = "Imec" in (egroup.device.manufacturer or "")
    return "DMFC" if (npx and ml > 0) else "FEF" if (not npx and ml < 0) else "other"


def load_single_object_trials(session=SESSION, subject=SUBJECT):
    """Return a DataFrame of clean single-object trials (index = NWB trial id)."""
    nwb = _read(_behav(session, subject), "behav", session, subject)
    df = nwb.intervals["trials"].to_dataframe()

    ids = df["stimulus_object_identities"].map(ast.literal_eval)
    pos = df["stimulus_object_positions"].map(ast.literal_eval)
    keep = (ids.map(len) == 1) & (~df["broke_fixation"].to_numpy())

    xy = np.array([p[0] for p in pos[keep]])
    # Stable position id (0..2) from the three unique locations.
    uniq = np.unique(xy.round(3), axis=0)
    pos_id = [int(np.argmin(np.abs(uniq - r).sum(axis=1))) for r in xy.round(3)]

    # Saccade endpoint (post-hoc, high-res) and its nearest canonical position.
    resp = np.array([np.asarray(r, dtype=float) for r in df.loc[keep, "response_position"]])
    chosen = [int(np.argmin(np.abs(uniq - r).sum(axis=1))) for r in resp.round(3)]

    out = pd.DataFrame(
        {
            "identity": [i[0] for i in ids[keep]],
            "position": pos_id,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "correct": df.loc[keep, "reward_duration"].to_numpy() > 0,
            "chosen": chosen,
            "resp_x": resp[:, 0],
            "resp_y": resp[:, 1],
            "stim_time": df.loc[keep, "phase_stimulus_time"].to_numpy(),
            "delay_time": df.loc[keep, "phase_delay_time"].to_numpy(),
            "cue_time": df.loc[keep, "phase_cue_time"].to_numpy(),
        },
        index=df.index[keep].rename("trial_id"),
    )
    return out


def _probe_entry_coords(group):
    """Stereotactic entry coords (ml, ap, si) mm from an electrode_group description; si is nan for the 2D Neuropixel."""
    m = re.search(r'first_channel = \[([^\]]+)\]', group.description)  # V-probe (3D)
    if m is None:
        m = re.search(r'coordinates = \[([^\]]+)\]', group.description)  # Neuropixel (2D surface)
    v = [float(x) for x in m.group(1).split(',')]
    return v + [np.nan] * (3 - len(v))


def load_units(session=SESSION, subject=SUBJECT):
    """Return (TsGroup of spike times, obs_trials DataFrame [unit x trial-id, bool])."""
    nwb = _read(_spikes(session, subject), "spikes", session, subject)
    u = nwb.processing["ecephys"]["units"]
    n = len(u.id)

    coords = {name: _probe_entry_coords(g) for name, g in nwb.electrode_groups.items()}
    group = np.asarray(u["electrodes_group"][:])
    ml, ap, si = np.array([coords[g] for g in group]).T
    region = np.array([_region_of(nwb.electrode_groups[g], m) for g, m in zip(group, ml)])

    units = nap.TsGroup({i: nap.Ts(np.asarray(u["spike_times"][i])) for i in range(n)})
    units.set_info(
        quality=np.asarray(u["quality"][:]),
        depth=np.asarray(u["depth"][:]),
        group=group,
        region=region,
        ml=ml, ap=ap, si=si,
    )
    obs = pd.DataFrame(
        np.stack([np.asarray(x) for x in u["obs_trials"][:]]),
        index=pd.Index(range(n), name="unit"),
    )
    obs.columns.name = "trial_id"
    return units, obs
