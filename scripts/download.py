"""Download missing behavior+task and spikesorting NWB files from DANDI (skips raw ecephys)."""
import json
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DANDISET, _VERSION = "000620", "0.260127.2208"


def _target(subject, session, kind):
    if kind == "behav":
        return _ROOT / f"data/{subject}/behav/sub-{subject}_ses-{session}_behavior+task.nwb"
    return _ROOT / f"data/{subject}/ephys/spikes/sub-{subject}_ses-{session}_spikesorting.nwb"


def _assets():
    url = (f"https://api.dandiarchive.org/api/dandisets/{_DANDISET}"
           f"/versions/{_VERSION}/assets/?page_size=1000")
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        yield from d["results"]
        url = d.get("next")


def _missing():
    """List of (asset, dest) for behav/spikes assets not yet on disk."""
    out = []
    for a in _assets():
        path = a["path"]
        if path.endswith("behavior+task.nwb"):
            kind = "behav"
        elif path.endswith("spikesorting.nwb"):
            kind = "spikes"
        else:
            continue
        m = re.search(r"sub-([^/_]+)_ses-([0-9-]+)", path)
        if not m:
            continue
        dest = _target(m.group(1), m.group(2), kind)
        if not dest.exists():
            out.append((a, dest))
    return out


def download_missing(dry_run=True):
    miss = _missing()
    total = sum(a.get("size", 0) for a, _ in miss)
    print(f"{len(miss)} files missing, {total / 1e9:.2f} GB")
    for a, dest in miss:
        print(f"  {a.get('size', 0) / 1e6:7.0f} MB  {dest.relative_to(_ROOT)}")
    if dry_run:
        print("dry run; pass --go to download")
        return
    for i, (a, dest) in enumerate(miss, 1):
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
        print(f"[{i}/{len(miss)}] {dest.relative_to(_ROOT)}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    download_missing(dry_run="--go" not in sys.argv)
