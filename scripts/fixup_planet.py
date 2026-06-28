"""Planet-specific fixup: only generates height_minmax JSONs and mirrors height/ as height_DAM3/.
Does NOT touch views 3 and 4 — those hold real Planet target imagery."""
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")

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

dam3 = ROOT / "height_DAM3"
if dam3.exists():
    shutil.rmtree(dam3)
shutil.copytree(ROOT / "height", dam3)
print("height_DAM3 mirrored from height (dummy zero heights)")
