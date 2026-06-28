"""Keep only crops that exist in every view's image/ directory.
Removes orphan files (where save_cropped_view succeeded for some views
but the camera-fitting step then failed for a later view in the same crop)."""
import re
from pathlib import Path

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")

def crop_ids_in_view(view: int) -> set[int]:
    d = ROOT / "image" / str(view)
    ids = set()
    for p in d.glob("*.tif"):
        m = re.search(r"_crop_(\d+)\.tif$", p.name)
        if m:
            ids.add(int(m.group(1)))
    return ids

common = None
for v in range(5):
    s = crop_ids_in_view(v)
    common = s if common is None else common & s
    print(f"view{v}: {len(s)} crops")
print(f"common: {len(common)} crops -> {sorted(common)}")

# Delete every artifact for crops not in `common`.
for sub, ext in [
    ("image", ".tif"),
    ("height", ".tif"),
    ("rpc", "_170.rpc"),
    ("cameras", ".json"),
    ("cameras_others", ".json"),
]:
    for v in range(5):
        d = ROOT / sub / str(v)
        if not d.exists():
            continue
        for p in d.iterdir():
            m = re.search(r"_crop_(\d+)(?=[._])", p.name)
            if m and int(m.group(1)) not in common:
                p.unlink()

# Sanity recount.
for v in range(5):
    cnt = len(list((ROOT / "image" / str(v)).glob("*.tif")))
    print(f"view{v} after cleanup: {cnt}")
