"""Merge per-crop PLY files into one big point cloud, then render multiple 3D views
(top-down, perspective, side) as PNGs. Also save a merged PLY for interactive viewing."""
import sys, glob, os
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path("/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-16-38_ctx_0012-0018-0006_height_ply_v2")
PLY_DIR = RUN / "renders/re10k/JAX_Tile_999/gaussians"
OUT = RUN / "visuals"
OUT.mkdir(parents=True, exist_ok=True)


def read_ply(path):
    """Read XYZ-RGB binary little-endian PLY."""
    with open(path, "rb") as fp:
        while True:
            line = fp.readline()
            if line.strip() == b"end_header":
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        return np.fromfile(fp, dtype=dt)


def write_ply(path, recs):
    """Write XYZ-RGB binary little-endian PLY."""
    n = len(recs)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as fp:
        fp.write(header)
        fp.write(recs.tobytes())


# --- 1) Merge ---
files = sorted(PLY_DIR.glob("*.ply"))
print(f"Merging {len(files)} PLY files...")
chunks = []
for p in files:
    chunks.append(read_ply(p))
merged = np.concatenate(chunks)
print(f"Total points: {len(merged):,}")

# Drop bottom/top 0.5% in z to reject outliers, then subsample for fast plotting
z = merged["z"]
lo, hi = np.percentile(z, [0.5, 99.5])
keep = (z >= lo) & (z <= hi)
clean = merged[keep]
print(f"After clipping outliers: {len(clean):,} points, z range {clean['z'].min():.1f}–{clean['z'].max():.1f}")

# Save merged PLY
merged_path = OUT / "merged_lotte_pointcloud.ply"
write_ply(merged_path, clean)
print(f"Saved merged PLY: {merged_path}  ({merged_path.stat().st_size/1e6:.1f} MB, {len(clean):,} points)")

# --- 2) Matplotlib 3D views ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Subsample for plotting
SUB = 50_000
if len(clean) > SUB:
    idx = np.random.default_rng(0).choice(len(clean), size=SUB, replace=False)
    sub = clean[idx]
else:
    sub = clean

xs = sub["x"]; ys = sub["y"]; zs = sub["z"]
rgb = np.stack([sub["r"], sub["g"], sub["b"]], axis=1) / 255.0

# top-down (look from +z down to -z) — show ground footprint
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, c=rgb, s=0.5, marker=".")
ax.view_init(elev=85, azim=-90)
ax.set_box_aspect([1, 1, 0.3])
ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
ax.set_title("Top-down view (looking nadir)")
plt.tight_layout()
plt.savefig(OUT / "pc_topdown.png", dpi=120)
plt.close()

# perspective oblique
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, c=rgb, s=0.5, marker=".")
ax.view_init(elev=35, azim=-60)
ax.set_box_aspect([1, 1, 0.6])
ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
ax.set_title("Oblique 3D view — Seoul Lotte area")
plt.tight_layout()
plt.savefig(OUT / "pc_oblique.png", dpi=120)
plt.close()

# side view (look from south, +y direction)
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, c=rgb, s=0.5, marker=".")
ax.view_init(elev=5, azim=-90)
ax.set_box_aspect([2, 1, 0.6])
ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
ax.set_title("Side view from south — building heights vs ground")
plt.tight_layout()
plt.savefig(OUT / "pc_side_south.png", dpi=120)
plt.close()

# Color points by HEIGHT instead of RGB (clearer for building structure)
from matplotlib import cm
height_norm = np.clip((zs - zs.min()) / (zs.max() - zs.min() + 1e-6), 0, 1)
plasma = cm.plasma(height_norm)[:, :3]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, c=plasma, s=0.5, marker=".")
ax.view_init(elev=35, azim=-60)
ax.set_box_aspect([1, 1, 0.6])
ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
ax.set_title(f"Oblique 3D — colored by height ({zs.min():.0f}–{zs.max():.0f} m)")
plt.tight_layout()
plt.savefig(OUT / "pc_oblique_heightcolor.png", dpi=120)
plt.close()

print("\n3D view PNGs:")
for f in sorted(OUT.glob("pc_*.png")):
    print(f"  {f}")
