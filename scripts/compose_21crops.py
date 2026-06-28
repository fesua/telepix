"""Compose wide Lotte view from 21-crop run. Uses current output256/ as-is."""
import json, sys, re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")
from preprocess.rpc_model import RPCModel

REPO = Path("/home/imlab-p6000/js/SkySplat-main")
ROOT = REPO / "CreatDataset/example/output256/test"
RUN = Path('/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-16-38_ctx_0012-0018-0006_height_ply_v2')
RENDERS = RUN / "renders"
OUT = RUN / "visuals"
OUT.mkdir(parents=True, exist_ok=True)

SLOT_TO_PID = ['0012', '0018', '0006', '0010', '0005']
SLOT_TO_RGB = ["001", "002", "003", "004", "005"]

VIEW_INFO = {
    "0012": "2021-02-09 near-nadir 0.9°",
    "0018": "2022-05-12 oblique ~10°",
    "0010": "2020-03-08 cam d2",
    "0005": "2020-03-08 cam d1 (held-out)",
    "0006": "2020-03-08 cam d1, same strip as 0005 (held-out)",
}

BBOX_PADDING = 80

# Discover surviving crops from view 0's image dir
CROPS = sorted(int(re.search(r"crop_(\d+)\.tif", p.name).group(1))
               for p in (ROOT / "image/0").glob("*.tif"))
print(f"surviving crops in view 0: {CROPS}")


def get_full_view(slot):
    """Re-extract the full Planet image + build RPCModel from the original TXT."""
    from osgeo import gdal
    gdal.UseExceptions()
    pid = SLOT_TO_PID[slot]
    stems = {
        "0012": "20210209_050530_ssc10d1_0012",
        "0018": "20220512_234256_ss02d3_0018",
        "0010": "20200308_021307_ssc2d2_0010",
        "0005": "20200308_021307_ssc2d1_0005",
        "0006": "20200308_021307_ssc2d1_0006",
    }
    stem = stems[pid]
    src_tif = REPO / "Dataset" / f"{stem}_basic_analytic.tif"
    src_rpc = REPO / "Dataset" / f"{stem}_basic_analytic_RPC.TXT"

    ds = gdal.Open(str(src_tif), gdal.GA_ReadOnly)
    arr = ds.ReadAsArray().astype(np.float32)
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=0)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        b = rgb[c]
        lo, hi = np.percentile(b, [2, 98])
        b = np.clip((b - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
        out[c] = b.astype(np.uint8)
    img_hwc = out.transpose(1, 2, 0)
    h, w = img_hwc.shape[:2]
    ds = None

    raw = {}
    for line in src_rpc.read_text().splitlines():
        k, _, v = line.strip().partition(":")
        if k:
            raw[k.strip()] = v.strip()
    rpc_dict = {
        "rowOff": float(raw["LINE_OFF"]),  "rowScale": float(raw["LINE_SCALE"]),
        "colOff": float(raw["SAMP_OFF"]),  "colScale": float(raw["SAMP_SCALE"]),
        "latOff": float(raw["LAT_OFF"]),   "latScale": float(raw["LAT_SCALE"]),
        "lonOff": float(raw["LONG_OFF"]),  "lonScale": float(raw["LONG_SCALE"]),
        "altOff": float(raw["HEIGHT_OFF"]),"altScale": float(raw["HEIGHT_SCALE"]),
        "rowNum": [float(raw[f"LINE_NUM_COEFF_{i}"]) for i in range(1, 21)],
        "rowDen": [float(raw[f"LINE_DEN_COEFF_{i}"]) for i in range(1, 21)],
        "colNum": [float(raw[f"SAMP_NUM_COEFF_{i}"]) for i in range(1, 21)],
        "colDen": [float(raw[f"SAMP_DEN_COEFF_{i}"]) for i in range(1, 21)],
    }
    rpc = RPCModel({"rpc": rpc_dict, "height": h, "width": w})
    return img_hwc, rpc, h, w


def crop_center_latlon(fb):
    p = ROOT / "cameras_others/0" / f"JAX_Tile_999_RGB_001_crop_{fb}_latlonalt_bbx.json"
    if not p.exists():
        return None
    bbx = json.loads(p.read_text())
    return ((bbx["lat_minmax"][0] + bbx["lat_minmax"][1]) / 2.0,
            (bbx["lon_minmax"][0] + bbx["lon_minmax"][1]) / 2.0)


def load_render(slot, fb):
    rgb = SLOT_TO_RGB[slot]
    candidates = list(RENDERS.rglob(f"tgt_view_{slot}_JAX_Tile_999_RGB_{rgb}_crop_{fb}.png"))
    if not candidates:
        return None
    return np.asarray(Image.open(candidates[0]))


def label_panel(arr, text, font_size=44):
    img = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    bar = bbox[3] - bbox[1] + 16
    d.rectangle([0, 0, img.width, bar], fill=(0, 0, 0))
    d.text((10, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def upscale(arr, f):
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((w * f, h * f), Image.LANCZOS))


def pad_to(arr, h, w):
    out = np.zeros((h, w, 3), dtype=arr.dtype)
    oh, ow = arr.shape[:2]
    out[:oh, :ow] = arr
    return out


per_view = {}
for slot in range(5):
    pid = SLOT_TO_PID[slot]
    print(f"\nslot {slot} ({pid}):")
    gt_full, rpc, fh, fw = get_full_view(slot)
    canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
    placed = np.zeros((fh, fw), dtype=np.uint8)
    centers = []
    placed_n = 0
    for fb in CROPS:
        ll = crop_center_latlon(fb)
        if ll is None:
            continue
        lat, lon = ll
        col, row = rpc.projection(np.array([lat]), np.array([lon]), np.array([50.0]))
        cx, cy = float(col[0]), float(row[0])
        centers.append((cx, cy))
        rend = load_render(slot, fb)
        if rend is None:
            continue
        x0, y0 = int(round(cx - 128)), int(round(cy - 128))
        x1, y1 = x0 + 256, y0 + 256
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        dx0 = max(0, x0); dy0 = max(0, y0)
        dx1 = min(fw, x1); dy1 = min(fh, y1)
        sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        canvas[dy0:dy1, dx0:dx1] = rend[sy0:sy1, sx0:sx1]
        placed[dy0:dy1, dx0:dx1] = 1
        placed_n += 1

    if not centers:
        print("  no centers projected — skipping")
        continue
    xs = [c for c, _ in centers]; ys = [r for _, r in centers]
    bx0 = max(0, int(min(xs)) - 128 - BBOX_PADDING)
    by0 = max(0, int(min(ys)) - 128 - BBOX_PADDING)
    bx1 = min(fw, int(max(xs)) + 128 + BBOX_PADDING)
    by1 = min(fh, int(max(ys)) + 128 + BBOX_PADDING)
    if bx1 <= bx0 or by1 <= by0:
        print("  invalid bbox — skipping")
        continue
    gt_t = gt_full[by0:by1, bx0:bx1]
    rd_t = canvas[by0:by1, bx0:bx1]
    print(f"  placed {placed_n}/{len(CROPS)} crops, bbox {bx1-bx0}×{by1-by0}, coverage {int(placed[by0:by1, bx0:bx1].mean()*100)}%")
    per_view[slot] = (gt_t, rd_t, pid)
    Image.fromarray(gt_t).save(OUT / f"view{slot}_{pid}_gt.png")
    Image.fromarray(rd_t).save(OUT / f"view{slot}_{pid}_render.png")

# 5-view grid
if per_view:
    max_h = max(t[0].shape[0] for t in per_view.values())
    max_w = max(t[0].shape[1] for t in per_view.values())
    F = 2
    cols = []
    for slot in sorted(per_view):
        gt, rd, pid = per_view[slot]
        gt_p = pad_to(gt, max_h, max_w); rd_p = pad_to(rd, max_h, max_w)
        col = np.concatenate([
            label_panel(upscale(gt_p, F), f"GT  {pid}  ({VIEW_INFO[pid]})"),
            label_panel(upscale(rd_p, F), f"Render @ {pid}"),
        ], axis=0)
        cols.append(col)
    grid = np.concatenate(cols, axis=1)
    Image.fromarray(grid).save(OUT / "all5_views_lotte_wide_21crops.png")
    print(f"\nsaved {OUT / 'all5_views_lotte_wide_21crops.png'} ({grid.shape[1]}×{grid.shape[0]})")
