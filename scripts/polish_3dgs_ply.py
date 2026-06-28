"""Make a viewer-friendly 3DGS PLY:
* drop sub-pixel-tiny and giant gaussians (clip log-scale to a sane band)
* boost opacities so visible gaussians render solid (not 0.12 mean)
* keep size manageable (~500k or 1M gaussians)
* still standard 3DGS PLY format → SuperSplat / antimatter15 viewer compatible
"""
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData

SRC = Path("/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-34-50_ctx_0012-0018-0006_3dgs_ply/visuals/merged_3dgs.ply")
DST_DIR = SRC.parent
print(f"Reading {SRC}...")
d = PlyData.read(str(SRC))
v = d['vertex'].data  # structured ndarray

# Filters
log_scale = v['scale_0']  # use scale_0 as proxy for size

v = v.copy()

# 1) Cap scales so no single gaussian is absurdly large or vanishingly small.
#    Median scales are -2/-6/-0.6 → anisotropic pancake (correct for stereo depth).
SCALE_LOG_MAX = 0.0   # exp = 1.0 m (avoid 2.4m blobs)
SCALE_LOG_MIN = -8.0  # exp = 3e-4 m  (avoid sub-pixel zeros that look like point cloud)
for k in ['scale_0', 'scale_1', 'scale_2']:
    v[k] = np.clip(v[k], SCALE_LOG_MIN, SCALE_LOG_MAX)

# 2) Boost opacity. Median logit ~ -2 → sigmoid 0.12. Add +3 → median 0.73.
v['opacity'] = v['opacity'] + 3.0
print(f"Opacity after +3 boost: sigmoid mean = {(1/(1+np.exp(-v['opacity']))).mean():.3f}, median = {(1/(1+np.exp(-np.median(v['opacity'])))):.3f}")

# Optionally cap scale to log -2.5 (0.08m) to avoid bright over-blending
# v['scale_0'] = np.clip(v['scale_0'], -np.inf, -2.5)
# v['scale_1'] = np.clip(v['scale_1'], -np.inf, -2.5)
# v['scale_2'] = np.clip(v['scale_2'], -np.inf, -2.5)


def write_subset(arr, name, n_target=None):
    if n_target is not None and len(arr) > n_target:
        idx = np.random.default_rng(0).choice(len(arr), size=n_target, replace=False)
        arr = arr[idx]
    out = DST_DIR / name
    from plyfile import PlyElement
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False, byte_order='<').write(str(out))
    print(f"  saved {name}  ({out.stat().st_size/1e6:.1f} MB, {len(arr):,} gaussians)")


write_subset(v, "merged_3dgs_polished_full.ply")
write_subset(v, "merged_3dgs_polished_1000k.ply", n_target=1_000_000)
write_subset(v, "merged_3dgs_polished_500k.ply",  n_target=500_000)
write_subset(v, "merged_3dgs_polished_200k.ply",  n_target=200_000)

print(f"\nAll in {DST_DIR}/")
