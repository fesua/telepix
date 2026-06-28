"""Fill in artifacts that the example dataset is missing so test mode can run end-to-end.
Numbers are meaningless — purpose is to demo the pipeline."""
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")

# 1) Generate <name>_height_minmax.json next to every height tif
for tif in sorted(ROOT.glob("height/*/*.tif")):
    arr = tifffile.imread(tif).astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        mn, mx = 0.0, 1.0
    else:
        mn, mx = float(finite.min()), float(finite.max())
        if mx == mn:
            mx = mn + 1.0
    out = tif.with_name(tif.stem + "_height_minmax.json")
    out.write_text(json.dumps({"min_height": mn, "max_height": mx}))
print("height_minmax JSONs created")

# 2) Mirror height/ as height_DAM3/ (placeholder for DAM3-estimated heights)
dam3 = ROOT / "height_DAM3"
if dam3.exists():
    shutil.rmtree(dam3)
shutil.copytree(ROOT / "height", dam3)
print("height_DAM3 mirrored from height")

# 3) Provide target views 3 and 4 by copying views 1 and 2
for subdir in ["image", "cameras"]:
    for src_view, dst_view in [("1", "3"), ("2", "4")]:
        src = ROOT / subdir / src_view
        dst = ROOT / subdir / dst_view
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
print("views 3 and 4 populated from views 1 and 2")
