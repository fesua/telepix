"""Render 3D point cloud with proper density + an interactive plotly HTML.
Plotly HTML rotates/zooms in any browser without installing anything."""
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

RUN = Path("/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-16-38_ctx_0012-0018-0006_height_ply_v2")
OUT = RUN / "visuals"

# Read merged PLY directly
ply_path = OUT / "merged_lotte_pointcloud.ply"
with open(ply_path, "rb") as fp:
    while fp.readline().strip() != b"end_header":
        pass
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.fromfile(fp, dtype=dt)
print(f"Loaded {len(arr):,} points from merged PLY")

# Center
center = np.array([arr["x"].mean(), arr["y"].mean(), arr["z"].mean()])
xs = arr["x"] - center[0]
ys = arr["y"] - center[1]
zs = arr["z"] - center[2]
rgb = np.stack([arr["r"], arr["g"], arr["b"]], axis=1).astype(np.float32) / 255.0

# Build height colors too
zabs = arr["z"]
zn = (zabs - zabs.min()) / max(zabs.max() - zabs.min(), 1e-6)
height_colors = cm.plasma(zn)[:, :3]

# ----- Matplotlib still images (denser, better aspect) -----
SUB = 150_000   # bigger subsample for denser plot
if len(arr) > SUB:
    idx = np.random.default_rng(0).choice(len(arr), size=SUB, replace=False)
    xs_s, ys_s, zs_s = xs[idx], ys[idx], zs[idx]
    rgb_s = rgb[idx]
    hc_s = height_colors[idx]
else:
    xs_s, ys_s, zs_s = xs, ys, zs
    rgb_s = rgb
    hc_s = height_colors

# Equal-aspect bounds
rng = max(xs_s.max() - xs_s.min(), ys_s.max() - ys_s.min(), zs_s.max() - zs_s.min()) / 2
mx, my, mz = xs_s.mean(), ys_s.mean(), zs_s.mean()


def render_mpl(colors, name, elev, azim, title, point_size=1.2, alpha=0.7):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs_s, ys_s, zs_s, c=colors, s=point_size, marker=".", alpha=alpha, edgecolors='none')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(mx - rng, mx + rng)
    ax.set_ylim(my - rng, my + rng)
    ax.set_zlim(zs_s.min(), zs_s.min() + 2 * rng * 0.4)  # compress vertical
    ax.set_box_aspect([1, 1, 0.4])
    ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
    ax.set_title(title, fontsize=14)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=120, facecolor='white')
    plt.close()
    print(f"  saved {name}")


print("Rendering matplotlib views...")
render_mpl(rgb_s,  "pc_rgb_oblique.png",       elev=25, azim=-55, title="3D point cloud — RGB (oblique 25°)")
render_mpl(hc_s,   "pc_height_oblique.png",    elev=25, azim=-55, title=f"3D point cloud — height colormap (plasma, {zabs.min():.0f}-{zabs.max():.0f} m)")
render_mpl(rgb_s,  "pc_rgb_high_oblique.png",  elev=45, azim=-55, title="3D — RGB (high oblique 45°)")
render_mpl(hc_s,   "pc_height_high_oblique.png", elev=45, azim=-55, title="3D — height (high oblique 45°)")
render_mpl(rgb_s,  "pc_rgb_side.png",          elev=8,  azim=-90, title="Side view from south — RGB")
render_mpl(hc_s,   "pc_height_side.png",       elev=8,  azim=-90, title=f"Side view — height ({zabs.min():.0f}-{zabs.max():.0f} m)")

# ----- Interactive plotly HTML -----
print("\nBuilding interactive plotly HTML...")
import plotly.graph_objects as go

# Limit to ~200k for plotly (browser perf)
SUB_PLOTLY = 200_000
if len(arr) > SUB_PLOTLY:
    idx = np.random.default_rng(0).choice(len(arr), size=SUB_PLOTLY, replace=False)
    xs_p = arr["x"][idx]; ys_p = arr["y"][idx]; zs_p = arr["z"][idx]
    rgb_p = np.stack([arr["r"], arr["g"], arr["b"]], axis=1)[idx]
else:
    xs_p, ys_p, zs_p = arr["x"], arr["y"], arr["z"]
    rgb_p = np.stack([arr["r"], arr["g"], arr["b"]], axis=1)

rgb_str = [f"rgb({r},{g},{b})" for r, g, b in rgb_p]

fig = go.Figure(data=[go.Scatter3d(
    x=xs_p, y=ys_p, z=zs_p,
    mode="markers",
    marker=dict(size=1.5, color=rgb_str, opacity=0.7),
    hoverinfo="skip",
)])
fig.update_layout(
    scene=dict(
        xaxis_title="E (m)",
        yaxis_title="N (m)",
        zaxis_title="U (m) — altitude",
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.4),
        bgcolor="black",
    ),
    paper_bgcolor="black",
    font=dict(color="white"),
    title=f"Lotte World area 3D point cloud — {len(xs_p):,} pts, altitude {zs_p.min():.0f}-{zs_p.max():.0f} m",
    margin=dict(l=0, r=0, t=40, b=0),
)
html_path = OUT / "interactive_3d.html"
fig.write_html(html_path, include_plotlyjs="cdn", config=dict(displaylogo=False))
print(f"  saved {html_path}  ({html_path.stat().st_size/1e6:.1f} MB)")

# Also height-colored interactive
hc_p = (cm.plasma((zs_p - zs_p.min()) / max(zs_p.max() - zs_p.min(), 1e-6))[:, :3] * 255).astype(np.uint8)
hc_str = [f"rgb({r},{g},{b})" for r, g, b in hc_p]
fig2 = go.Figure(data=[go.Scatter3d(
    x=xs_p, y=ys_p, z=zs_p,
    mode="markers",
    marker=dict(size=1.5, color=hc_str, opacity=0.7),
    hoverinfo="skip",
)])
fig2.update_layout(
    scene=dict(
        xaxis_title="E (m)", yaxis_title="N (m)", zaxis_title="U (m) — altitude",
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.4),
        bgcolor="black",
    ),
    paper_bgcolor="black", font=dict(color="white"),
    title=f"Lotte World 3D — height-colored plasma ({zs_p.min():.0f}-{zs_p.max():.0f} m)",
    margin=dict(l=0, r=0, t=40, b=0),
)
html2_path = OUT / "interactive_3d_height.html"
fig2.write_html(html2_path, include_plotlyjs="cdn", config=dict(displaylogo=False))
print(f"  saved {html2_path}  ({html2_path.stat().st_size/1e6:.1f} MB)")

print("\nAll outputs:")
for p in sorted(OUT.glob("pc_*")) + sorted(OUT.glob("interactive*")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")
