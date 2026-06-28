"""Place each 256x256 rendered crop at its actual position in each of the 5 target views,
reconstructing a single big image for every viewpoint."""
import json, sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")
from preprocess.rpc_model import RPCModel
from preprocess.parse_tif_image import parse_tif_image

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")
RUN = Path(open("/tmp/run_ts.txt").read().strip().split("=", 1)[1])
RENDERS = RUN / "renders/re10k/JAX_Tile_999/color"   # renamed file pattern: tgt_view_{i}_*.png
OUT = RUN / "visuals"
OUT.mkdir(parents=True, exist_ok=True)

CROPS = [16, 17, 18, 26, 27, 28]
INPUT2048 = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/input2048/test/image")

VIEW_INFO = {
    0: ("0012", "2021-02-09 near-nadir 0.9°"),
    1: ("0018", "2022-05-12 oblique ~10° from 0012"),
    2: ("0010", "2020-03-08 cam d2, ~8° from 0012"),
    3: ("0005", "2020-03-08 cam d1, held-out"),
    4: ("0006", "2020-03-08 cam d1, held-out (same strip as 0005)"),
}


def load_full_view(view_slot: int):
    rgb_no = ["001", "002", "003", "004", "005"][view_slot]
    p = INPUT2048 / str(view_slot) / f"JAX_Tile_999_RGB_{rgb_no}.tif"
    img, meta = parse_tif_image(str(p))
    return img, RPCModel(meta), meta["height"], meta["width"]


def load_render(view_slot: int, fb: int):
    rgb_no = ["001", "002", "003", "004", "005"][view_slot]
    p = RENDERS / f"tgt_view_{view_slot}_JAX_Tile_999_RGB_{rgb_no}_crop_{fb}.png"
    if not p.exists():
        return None
    return np.asarray(Image.open(p))


def crop_center_latlon(fb: int):
    bbx = json.load(open(ROOT / "cameras_others" / "0" / f"JAX_Tile_999_RGB_001_crop_{fb}_latlonalt_bbx.json"))
    return ((bbx["lat_minmax"][0] + bbx["lat_minmax"][1]) / 2.0,
            (bbx["lon_minmax"][0] + bbx["lon_minmax"][1]) / 2.0)


def compose_target(target_view_slot: int, alt_for_proj=50.0):
    full_img, full_rpc, full_h, full_w = load_full_view(target_view_slot)
    canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    placed = np.zeros((full_h, full_w), dtype=np.uint8)
    placements = []
    for fb in CROPS:
        lat, lon = crop_center_latlon(fb)
        col, row = full_rpc.projection(np.array([lat]), np.array([lon]), np.array([alt_for_proj]))
        cx, cy = float(col[0]), float(row[0])
        x0, y0 = int(round(cx - 128)), int(round(cy - 128))
        rend = load_render(target_view_slot, fb)
        if rend is None:
            continue
        x1, y1 = x0 + 256, y0 + 256
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        dx0 = max(0, x0); dy0 = max(0, y0)
        dx1 = min(full_w, x1); dy1 = min(full_h, y1)
        sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        canvas[dy0:dy1, dx0:dx1] = rend[sy0:sy1, sx0:sx1]
        placed[dy0:dy1, dx0:dx1] = 1
        placements.append(fb)
    return full_img, canvas, placed, placements


def tight(arrs, placed):
    if placed.sum() == 0:
        return arrs
    ys, xs = np.where(placed)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return [a[y0:y1, x0:x1] for a in arrs]


def label_panel(arr, text, font_size=44):
    img = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    bar = bbox[3] - bbox[1] + 24
    d.rectangle([0, 0, img.width, bar], fill=(0, 0, 0))
    d.text((12, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def upscale(arr, factor):
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((w * factor, h * factor), Image.LANCZOS))


def pad_to(arr, h_target, w_target):
    """Pad arr (top-left aligned) with black to (h_target, w_target)."""
    h, w = arr.shape[:2]
    out = np.zeros((h_target, w_target, 3), dtype=arr.dtype)
    out[:h, :w] = arr
    return out


# Build per-view triptychs and a 5-row summary
all_panels = []   # each entry: (label, gt_tight, render_tight)

for view_slot in range(5):
    pid, desc = VIEW_INFO[view_slot]
    print(f"\n=== view {view_slot} ({pid}) — {desc} ===")
    gt_full, rendered, placed, placements = compose_target(view_slot)
    print(f"  placed {len(placements)}/{len(CROPS)} crops: {placements}")
    if placed.sum() == 0:
        print(f"  no valid placement — skipping")
        continue
    gt_t, rend_t, placed_t = tight([gt_full, rendered, placed], placed)
    H, W = gt_t.shape[:2]
    print(f"  bbox in view {pid}: {W}×{H}")

    # Standalone files
    Image.fromarray(rend_t).save(OUT / f"view{view_slot}_{pid}_render_in_target_coords.png")
    Image.fromarray(gt_t).save(OUT / f"view{view_slot}_{pid}_gt_in_target_coords.png")

    # Per-view triptych (upscaled)
    factor = max(1, 2048 // max(H, W))
    sbs = np.concatenate([
        label_panel(upscale(gt_t, factor), f"GT  view{view_slot} ({pid})  {desc}"),
        label_panel(upscale(rend_t, factor), f"Render @ view{view_slot} ({pid})"),
    ], axis=1)
    Image.fromarray(sbs).save(OUT / f"view{view_slot}_{pid}_compare.png")
    print(f"  saved compare ({sbs.shape[1]}×{sbs.shape[0]})")

    all_panels.append((view_slot, pid, desc, gt_t, rend_t))

# 5-view summary: each column = view, top row = GT, bottom row = render
# Normalize all panels to a common width by padding with black.
if all_panels:
    max_h = max(p[3].shape[0] for p in all_panels)
    max_w = max(p[3].shape[1] for p in all_panels)
    F = 2
    cols = []
    for view_slot, pid, desc, gt_t, rend_t in all_panels:
        gt_p = pad_to(gt_t, max_h, max_w)
        rd_p = pad_to(rend_t, max_h, max_w)
        col = np.concatenate([
            label_panel(upscale(gt_p, F), f"GT  {pid}  ({desc})", font_size=30),
            label_panel(upscale(rd_p, F), f"Render @ {pid}", font_size=30),
        ], axis=0)
        cols.append(col)
    grid = np.concatenate(cols, axis=1)
    Image.fromarray(grid).save(OUT / "all5_views_grid.png")
    print(f"\nsaved all5_views_grid.png ({grid.shape[1]}×{grid.shape[0]})")

print("\nAll outputs in:", OUT)
for p in sorted(OUT.glob("*.png")):
    arr = np.asarray(Image.open(p))
    print(f"  {p.name}: {arr.shape[1]}×{arr.shape[0]}")
