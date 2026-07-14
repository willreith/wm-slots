"""Extract single-object trials and spikes from the Watters WM dataset (Perle session)."""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pynapple as nap

_ROOT = Path(__file__).resolve().parents[1]
SESSION = "2022-05-28"


def _behav(session=SESSION):
    return _ROOT / f"data/Perle/behav/sub-Perle_ses-{session}_behavior+task.nwb"


def _spikes(session=SESSION):
    return _ROOT / f"data/Perle/ephys/spikes/sub-Perle_ses-{session}_spikesorting.nwb"


def load_single_object_trials(session=SESSION):
    """Return a DataFrame of clean single-object trials (index = NWB trial id)."""
    nwb = pynwb.NWBHDF5IO(str(_behav(session)), mode="r").read()
    df = nwb.intervals["trials"].to_dataframe()

    ids = df["stimulus_object_identities"].map(ast.literal_eval)
    pos = df["stimulus_object_positions"].map(ast.literal_eval)
    keep = (ids.map(len) == 1) & (~df["broke_fixation"].to_numpy())

    xy = np.array([p[0] for p in pos[keep]])
    # Stable position id (0..2) from the three unique locations.
    uniq = np.unique(xy.round(3), axis=0)
    pos_id = [int(np.argmin(np.abs(uniq - r).sum(axis=1))) for r in xy.round(3)]

    out = pd.DataFrame(
        {
            "identity": [i[0] for i in ids[keep]],
            "position": pos_id,
            "x": xy[:, 0],
            "y": xy[:, 1],
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


def load_units(session=SESSION):
    """Return (TsGroup of spike times, obs_trials DataFrame [unit x trial-id, bool])."""
    nwb = pynwb.NWBHDF5IO(str(_spikes(session)), mode="r").read()
    u = nwb.processing["ecephys"]["units"]
    n = len(u.id)

    coords = {name: _probe_entry_coords(g) for name, g in nwb.electrode_groups.items()}
    group = np.asarray(u["electrodes_group"][:])
    ml, ap, si = np.array([coords[g] for g in group]).T
    # s0 = Neuropixel in right hemisphere = DMFC; vprobe* = V-probe in left hemisphere = FEF.
    region = np.where(group == "s0", "DMFC", "FEF")

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
