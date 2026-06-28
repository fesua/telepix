"""Keep crops common to the 3 CONTEXT views only (slots 0, 1, 2).
Drop the unnecessary requirement that held-out views (3, 4) also have the crop —
that filter was throwing away most of the joint overlap region between 0012/0018/0010."""
import re
from pathlib import Path

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")
CONTEXT_SLOTS = [0, 1, 2]
ALL_SLOTS = [0, 1, 2, 3, 4]

def crop_ids(view: int) -> set[int]:
    d = ROOT / "image" / str(view)
    if not d.exists():
        return set()
    ids = set()
    for p in d.glob("*.tif"):
        m = re.search(r"_crop_(\d+)\.tif$", p.name)
        if m:
            ids.add(int(m.group(1)))
    return ids

common = None
for v in CONTEXT_SLOTS:
    s = crop_ids(v)
    common = s if common is None else common & s
    print(f"view{v}: {len(s)} crops (context)")
for v in [s for s in ALL_SLOTS if s not in CONTEXT_SLOTS]:
    print(f"view{v}: {len(crop_ids(v))} crops (target, not required)")
print(f"common-across-context: {len(common)} crops -> {sorted(common)}")

# Delete every crop artifact NOT in `common` (across all dirs).
for sub, ext in [
    ("image", ".tif"),
    ("height", ".tif"),
    ("rpc", "_170.rpc"),
    ("cameras", ".json"),
    ("cameras_others", ".json"),
]:
    for v in ALL_SLOTS:
        d = ROOT / sub / str(v)
        if not d.exists():
            continue
        for p in d.iterdir():
            m = re.search(r"_crop_(\d+)(?=[._])", p.name)
            if m and int(m.group(1)) not in common:
                p.unlink()

# Recount
print("\nAfter cleanup:")
for v in ALL_SLOTS:
    d = ROOT / "image" / str(v)
    n = len(list(d.glob("*.tif"))) if d.exists() else 0
    print(f"  view{v}: {n} crops")
