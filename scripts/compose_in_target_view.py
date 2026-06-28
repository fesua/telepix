"""Place each 256x256 rendered crop at its actual position in the held-out
target view (0005 or 0006), reconstructing a single big image from that
viewpoint. Position comes from RPC: project each crop's lat/lon center
through the target view's RPC to get the pixel where the crop should sit."""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")
from preprocess.rpc_model import RPCModel
from preprocess.parse_tif_image import parse_tif_image

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")
RUN = Path(open("/tmp/run_ts.txt").read().strip().split("=", 1)[1])
RENDERS = RUN / "renders/JAX_Tile_999/color"
OUT = RUN / "visuals"
OUT.mkdir(parents=True, exist_ok=True)

CROPS = [16, 17, 18, 26, 27, 28]
# View slot → original Planet TIFF path (need full uncropped image for canvas + full-view RPC)
INPUT2048 = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/input2048/test/image")


def load_full_view(view_slot: int):
    """Return (image array h×w×3 uint8, full RPCModel) for the uncropped Planet tile."""
    rgb_no = ["001", "002", "003", "004", "005"][view_slot]
    p = INPUT2048 / str(view_slot) / f"JAX_Tile_999_RGB_{rgb_no}.tif"
    img, meta = parse_tif_image(str(p))
    return img, RPCModel(meta), meta["height"], meta["width"]


def load_render(target_zip: int, fb: int):
    """target_zip = 0 → render at view 3 (= 0005); target_zip = 1 → render at view 4 (= 0006).
    Filenames are quirky because of how model_wrapper.py builds them; the file path
    differs between the two saved-as positions."""
    if target_zip == 0:
        p = RENDERS / f"JAX_Tile_999_RGB_001_crop_{fb}.png"
    else:
        p = RENDERS / "CreatDataset/example/output256/test/image/1" / f"JAX_Tile_999_RGB_002_crop_{fb}.png"
    return np.asarray(Image.open(p))


def crop_center_latlon(fb: int):
    """Center lat/lon of crop fb's footprint (from preprocessing-saved bbx)."""
    bbx = json.load(open(ROOT / "cameras_others" / "0" / f"JAX_Tile_999_RGB_001_crop_{fb}_latlonalt_bbx.json"))
    return ((bbx["lat_minmax"][0] + bbx["lat_minmax"][1]) / 2.0,
            (bbx["lon_minmax"][0] + bbx["lon_minmax"][1]) / 2.0)


def compose_into_target(target_view_slot: int, target_zip: int, alt_for_proj: float = 50.0):
    """Place every render at its projected position in the target view's image.
    target_view_slot: 3 for 0005, 4 for 0006.
    target_zip: 0 if render is saved under "RGB_001_" filename, 1 if under nested path.
    """
    full_img, full_rpc, full_h, full_w = load_full_view(target_view_slot)
    canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    placed = np.zeros((full_h, full_w), dtype=np.uint8)  # marker of which pixels have content

    placements = []
    for fb in CROPS:
        lat, lon = crop_center_latlon(fb)
        col, row = full_rpc.projection(np.array([lat]), np.array([lon]), np.array([alt_for_proj]))
        cx, cy = float(col[0]), float(row[0])
        # Place a 256×256 render centered on (cx, cy)
        x0 = int(round(cx - 128))
        y0 = int(round(cy - 128))
        rend = load_render(target_zip, fb)
        # Clip to canvas bounds
        x1, y1 = x0 + 256, y0 + 256
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        dx0 = max(0, x0); dy0 = max(0, y0)
        dx1 = min(full_w, x1); dy1 = min(full_h, y1)
        sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
        if dx1 <= dx0 or dy1 <= dy0:
            print(f"  crop {fb} fell outside target view {target_view_slot}: ({cx:.0f}, {cy:.0f})")
            continue
        canvas[dy0:dy1, dx0:dx1] = rend[sy0:sy1, sx0:sx1]
        placed[dy0:dy1, dx0:dx1] = 1
        placements.append((fb, dx0, dy0, dx1, dy1))
    return full_img, canvas, placed, placements


def crop_to_active(arr_list, placed):
    """Tight-crop all arrays to the bounding box of placed=1."""
    if placed.sum() == 0:
        return arr_list
    ys, xs = np.where(placed)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return [a[y0:y1, x0:x1] for a in arr_list]


def label_panel(arr, text, font_size=48):
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


for target_view_slot, target_zip, label in [(3, 0, "0005"), (4, 1, "0006")]:
    print(f"\n=== Compose into target view {target_view_slot} ({label}) ===")
    gt_full, rendered_canvas, placed, places = compose_into_target(target_view_slot, target_zip)
    for fb, x0, y0, x1, y1 in places:
        print(f"  crop {fb} placed at view {label} pixels [{x0}-{x1}] × [{y0}-{y1}]")

    gt_tight, render_tight, placed_tight = crop_to_active([gt_full, rendered_canvas, placed], placed)
    H, W = gt_tight.shape[:2]
    print(f"  active bbox: {W}×{H}")

    # Standalone render (Lotte-area-only, target's viewpoint)
    factor = max(1, 3072 // max(W, 1))
    big_rend = upscale(render_tight, factor)
    Image.fromarray(big_rend).save(OUT / f"big_lotte_render_in_{label}_view.png")

    big_gt = upscale(gt_tight, factor)
    Image.fromarray(big_gt).save(OUT / f"big_lotte_gt_in_{label}_view.png")

    # Triptych: GT | render side-by-side
    sbs = np.concatenate([
        label_panel(big_gt, f"GT held-out  {label}  (Planet imagery)"),
        label_panel(big_rend, f"Model render @ {label} viewpoint"),
    ], axis=1)
    Image.fromarray(sbs).save(OUT / f"big_lotte_compare_{label}.png")
    print(f"  saved: big_lotte_compare_{label}.png ({sbs.shape[1]}×{sbs.shape[0]})")

print("\nAll outputs in:", OUT)
