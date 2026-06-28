"""Tensor-level histogram matching, applied AFTER the standard preprocessing pipeline.

For each non-reference slot, walk every cropped uint8 TIF in output256/test/image/<slot>/
and apply skimage.match_histograms against the reference slot's same-crop_id image.
The TIF is overwritten in place. The reference slot's TIFs are left untouched.

We identify the reference slot by reading the most recent VIEW_MAP from convert_planet.py
and matching on the Planet stem `20200308_021307_ssc2d2_0010`.
"""
import re, sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.exposure import match_histograms

CONVERT_SCRIPT = Path(__file__).parent / "convert_planet.py"
ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")
REFERENCE_PLANET_ID = "20200308_021307_ssc2d2_0010"


def find_ref_slot() -> int | None:
    """Read convert_planet.py's current VIEW_MAP and locate the ref Planet ID's slot."""
    src = CONVERT_SCRIPT.read_text()
    m = re.search(r"VIEW_MAP\s*=\s*\[(.*?)\n\]", src, re.DOTALL)
    if not m:
        return None
    for line in m.group(1).splitlines():
        m2 = re.match(r"\s*\(\"([^\"]+)\",\s*(\d+),", line)
        if m2 and m2.group(1) == REFERENCE_PLANET_ID:
            return int(m2.group(2))
    return None


def histmatch_dir(src_slot: int, ref_slot: int):
    rgb_no = ["001", "002", "003", "004", "005"][src_slot]
    ref_rgb_no = ["001", "002", "003", "004", "005"][ref_slot]
    src_dir = ROOT / "image" / str(src_slot)
    ref_dir = ROOT / "image" / str(ref_slot)
    if not src_dir.exists() or not ref_dir.exists():
        return 0, 0
    converted, skipped = 0, 0
    for src_p in sorted(src_dir.glob(f"JAX_Tile_999_RGB_{rgb_no}_crop_*.tif")):
        m = re.search(r"_crop_(\d+)\.tif$", src_p.name)
        if not m:
            skipped += 1
            continue
        crop_id = m.group(1)
        ref_p = ref_dir / f"JAX_Tile_999_RGB_{ref_rgb_no}_crop_{crop_id}.tif"
        if not ref_p.exists():
            skipped += 1
            continue
        src = tifffile.imread(src_p).astype(np.float32)
        ref = tifffile.imread(ref_p).astype(np.float32)
        matched = match_histograms(src, ref, channel_axis=-1)
        out = np.clip(matched, 0, 255).astype(np.uint8)
        tifffile.imwrite(src_p, out)
        converted += 1
    return converted, skipped


def main():
    ref_slot = find_ref_slot()
    if ref_slot is None:
        print(f"ERROR: could not locate slot for {REFERENCE_PLANET_ID} in convert_planet.py VIEW_MAP")
        sys.exit(1)
    print(f"Reference Planet ID = {REFERENCE_PLANET_ID}  → slot {ref_slot}")
    for slot in range(5):
        if slot == ref_slot:
            print(f"  slot {slot}: REFERENCE — skipped")
            continue
        c, s = histmatch_dir(slot, ref_slot)
        print(f"  slot {slot}: matched {c} crops, skipped {s}")


if __name__ == "__main__":
    main()
