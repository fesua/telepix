"""Stitch per-crop predicted-height grids (.npy, in meters) into a single big elevation
map over the Lotte World area, then save a plasma-colormapped PNG + a hillshade-ish gray PNG."""
import json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")
from preprocess.rpc_model import RPCModel

REPO = Path("/home/imlab-p6000/js/SkySplat-main")
RUN = Path("/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-16-38_ctx_0012-0018-0006_height_ply_v2")
ROOT = REPO / "CreatDataset/example/output256/test"
HEIGHT_DIR = RUN / "renders/re10k/JAX_Tile_999/height"
OUT = RUN / "visuals"
OUT.mkdir(parents=True, exist_ok=True)

CTX_VIEW = 0   # 0012 — reference grid
RGB_NO = "001"

CROPS = sorted(int(re.search(r"crop_(\d+)\.tif", p.name).group(1))
               for p in (ROOT / "image/0").glob("*.tif"))
print("crops:", CROPS)


def load_height(fb):
    p = HEIGHT_DIR / f"ctx_view_{CTX_VIEW}_JAX_Tile_999_RGB_{RGB_NO}_crop_{fb}.npy"
    return np.load(p) if p.exists() else None


def crop_center_pixel(fb):
    """Where in view 0's full image does this crop's center lie? (col, row).
    flag_block = i * w_blocks + j; w_blocks = 2560 // 256 = 10."""
    i, j = fb // 10, fb % 10
    return (j * 256 + 128, i * 256 + 128)


# Build a canvas the size of view 0's image (1080×2560), stitched continuously.
H_CANVAS, W_CANVAS = 1080, 2560
canvas = np.full((H_CANVAS, W_CANVAS), np.nan, dtype=np.float32)

for fb in CROPS:
    h = load_height(fb)
    if h is None:
        continue
    cx, cy = crop_center_pixel(fb)
    x0, y0 = cx - 128, cy - 128
    x1, y1 = x0 + 256, y0 + 256
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(W_CANVAS, x1); dy1 = min(H_CANVAS, y1)
    sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
    canvas[dy0:dy1, dx0:dx1] = h[sy0:sy1, sx0:sx1]

# Tight-crop to the area that has data
mask = np.isfinite(canvas)
if mask.sum() == 0:
    raise SystemExit("no height data")
ys, xs = np.where(mask)
y0, y1 = ys.min(), ys.max() + 1
x0, x1 = xs.min(), xs.max() + 1
tight = canvas[y0:y1, x0:x1]
tmask = np.isfinite(tight)
H, W = tight.shape
print(f"tight: {W}×{H}, range {np.nanmin(tight):.1f}m – {np.nanmax(tight):.1f}m")

# Robust range: 1st to 99th percentile, ignoring NaN
finite = tight[tmask]
lo, hi = np.percentile(finite, [1, 99])
print(f"colormap range: {lo:.1f} – {hi:.1f} m")

# Plasma colormap
from matplotlib import cm
norm = np.clip((tight - lo) / max(hi - lo, 1e-6), 0, 1)
plasma = (cm.plasma(norm)[:, :, :3] * 255).astype(np.uint8)
plasma[~tmask] = 0  # black for no-data

# Hillshade-ish: gradient magnitude on a smoothed version
from scipy.ndimage import gaussian_filter
smooth = np.where(tmask, tight, np.nanmean(finite))
smooth = gaussian_filter(smooth, sigma=1.5)
gy, gx = np.gradient(smooth)
slope = np.sqrt(gx ** 2 + gy ** 2)
slope_n = np.clip(slope / (np.percentile(slope, 98) + 1e-6), 0, 1)
gray = (255 * (1 - slope_n)).astype(np.uint8)
gray_rgb = np.stack([gray] * 3, axis=-1)
gray_rgb[~tmask] = 0

# Upscale 4× and label
def upscale(arr, f=4):
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((w * f, h * f), Image.LANCZOS))

def label(arr, t, fs=44):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), t, font=font)
    bar = bbox[3] - bbox[1] + 18
    d.rectangle([0, 0, im.width, bar], fill=(0, 0, 0))
    d.text((10, 6), t, fill=(255, 255, 255), font=font)
    return np.asarray(im)

Image.fromarray(upscale(plasma)).save(OUT / "height_colormap_view0012.png")
Image.fromarray(upscale(gray_rgb)).save(OUT / "height_gradient_view0012.png")

# Color RGB (existing render) next to height for comparison
color_path = REPO / "outputs/run_2026-06-28_12-16-38_ctx_0012-0018-0006_height_ply_v2/visuals/view0_0012_render.png"
if not color_path.exists():
    # Run compose_21crops first
    print("(skipping color-vs-height comparison — view0_0012_render.png not yet built)")
else:
    color = np.asarray(Image.open(color_path))
    # Resize plasma to same size
    plasma_big = np.asarray(Image.fromarray(plasma).resize((color.shape[1], color.shape[0]), Image.LANCZOS))
    sbs = np.concatenate([
        label(color, "RGB render @ 0012"),
        label(plasma_big, f"Predicted height (plasma, {lo:.0f}–{hi:.0f} m)"),
    ], axis=0)
    Image.fromarray(sbs).save(OUT / "color_vs_height_view0012.png")

print(f"\nsaved height visuals to {OUT}")
for p in sorted(OUT.glob("height_*.png")) + sorted(OUT.glob("color_vs_height*.png")):
    arr = np.asarray(Image.open(p))
    print(f"  {p.name}: {arr.shape[1]}×{arr.shape[0]}")

# Drop a hint about PLY viewers
print("\nPLY files (open in MeshLab / CloudCompare / Blender):")
for p in sorted((RUN / "renders/re10k/JAX_Tile_999/gaussians").glob("*.ply")):
    print(f"  {p}")
