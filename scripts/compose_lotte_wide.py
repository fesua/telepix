"""Compose a WIDE view of each run's renders, padded to cover Lotte World mall + Lotte
World Tower area visible together. Each crop is placed at its target-view pixel position
(via RPC), then the canvas is tight-cropped to ALL crops' bounding box (not just placed ones)
so the LOTTE WORLD mall and tower are framed together even if some crops are out of view."""
import json, sys, glob
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")
from preprocess.rpc_model import RPCModel
from preprocess.parse_tif_image import parse_tif_image

REPO = Path("/home/imlab-p6000/js/SkySplat-main")
INPUT2048 = REPO / "CreatDataset/example/input2048/test/image"

VIEW_INFO = {
    "0012": ("2021-02-09 near-nadir 0.9°", "ssc10d1"),
    "0018": ("2022-05-12 oblique ~10° from 0012", "ss02d3"),
    "0010": ("2020-03-08 cam d2", "ssc2d2"),
    "0005": ("2020-03-08 cam d1", "ssc2d1"),
    "0006": ("2020-03-08 cam d1 (same strip as 0005)", "ssc2d1"),
}

# Runs to process — value is the third (varying) context view ID only.
RUNS = [
    ('0010', Path('/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-00-07_ctx_0012-0018-0010_21crops')),
]

CROPS = [12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38]
ALT_FOR_PROJ = 50.0
# Pad the tight bbox outward by this many pixels in target-view coords, so the surrounding
# context (Lotte mall + tower) stays visible even if some crops drop out.
BBOX_PADDING = 80


def load_full_view_from_path(p: Path):
    img, meta = parse_tif_image(str(p))
    return img, RPCModel(meta), meta["height"], meta["width"]


def crop_center_latlon(output_root: Path, fb: int):
    bbx_path = output_root / "cameras_others/0" / f"JAX_Tile_999_RGB_001_crop_{fb}_latlonalt_bbx.json"
    bbx = json.loads(bbx_path.read_text())
    return ((bbx["lat_minmax"][0] + bbx["lat_minmax"][1]) / 2.0,
            (bbx["lon_minmax"][0] + bbx["lon_minmax"][1]) / 2.0)


def load_render(run_dir: Path, view_slot: int, fb: int):
    """Find renders saved as tgt_view_<slot>_<filename>.png."""
    rgb_no = ["001", "002", "003", "004", "005"][view_slot]
    candidates = list((run_dir / "renders").rglob(f"tgt_view_{view_slot}_JAX_Tile_999_RGB_{rgb_no}_crop_{fb}.png"))
    if not candidates:
        return None
    return np.asarray(Image.open(candidates[0]))


def compose_one(run_dir: Path, target_view_slot: int, planet_id: str):
    """Return GT and rendered canvases tight-cropped to ALL crops' bounding box
    (padded by BBOX_PADDING), regardless of which crops actually got placed."""
    rgb_no = ["001", "002", "003", "004", "005"][target_view_slot]
    # Need to know which Planet ID is at this slot for this run — but we don't track that here.
    # Use the input2048 file the run actually used (it's the same input2048 across runs because
    # we rewrote it for each).  Actually after the sweep the LAST configuration is what's on
    # disk. So we can't reliably read the per-run-input2048. Instead read the saved output256
    # cropped views which DO live under run_dir's preprocessing snapshot — but we didn't snapshot.
    #
    # Practical workaround: for each run, we know which Planet ID is at each slot from the
    # third_id in the run folder name. Reconstruct that mapping.
    raise NotImplementedError  # see compose_run_aware below


def planet_id_at_slot(third_id: str, slot: int) -> str:
    """Slot mapping used by the sweep:
       0: 0012, 1: 0018, 2: third, 3 and 4: the two leftovers."""
    fixed = ["0012", "0018"]
    used = set(fixed + [third_id])
    rem = [v for v in ["0012", "0018", "0010", "0005", "0006"] if v not in used]
    return [fixed[0], fixed[1], third_id, rem[0], rem[1]][slot]


def get_full_view_path_from_planet_id(planet_id: str) -> Path:
    """Locate the original Planet-format TIF in the Dataset/ dir (full-res, with RPC)."""
    stem_lookup = {
        "0012": "20210209_050530_ssc10d1_0012",
        "0018": "20220512_234256_ss02d3_0018",
        "0010": "20200308_021307_ssc2d2_0010",
        "0005": "20200308_021307_ssc2d1_0005",
        "0006": "20200308_021307_ssc2d1_0006",
    }
    # The original Planet tif is 4-band UInt16 with embedded RPC — but our parse_tif_image
    # only accepts 3-band uint8. So instead use the most recently written input2048 image
    # for that slot.  After the sweep the last config's input2048 is still on disk.
    #
    # Better: re-convert the Planet tif specifically and load it. But that requires re-running
    # convert_planet for each. To stay simple, use the SAME image content (which is what the
    # model rendered against) — read the saved cropped GT instead from output256.
    raise NotImplementedError


def get_full_view_image(third_id: str, slot: int):
    """Load the FULL Planet image (1080x2560) by re-extracting from the Dataset folder.

    We need RPC + image. The cleanest path: read the Planet TIFF with our converter logic.
    """
    # Inline the conversion to avoid disk dependence on the current input2048 state.
    from osgeo import gdal, gdalconst
    gdal.UseExceptions()
    stem_lookup = {
        "0012": "20210209_050530_ssc10d1_0012",
        "0018": "20220512_234256_ss02d3_0018",
        "0010": "20200308_021307_ssc2d2_0010",
        "0005": "20200308_021307_ssc2d1_0005",
        "0006": "20200308_021307_ssc2d1_0006",
    }
    pid = planet_id_at_slot(third_id, slot)
    stem = stem_lookup[pid]
    src_tif = REPO / "Dataset" / f"{stem}_basic_analytic.tif"
    src_rpc = REPO / "Dataset" / f"{stem}_basic_analytic_RPC.TXT"

    # Load image
    ds = gdal.Open(str(src_tif), gdal.GA_ReadOnly)
    arr = ds.ReadAsArray()  # (4, h, w) UInt16
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=0).astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        b = rgb[c]
        lo, hi = np.percentile(b, [2, 98])
        b = np.clip((b - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
        out[c] = b.astype(np.uint8)
    img_hwc = out.transpose(1, 2, 0)
    h, w = img_hwc.shape[:2]
    ds = None

    # Build the RPCModel from the TXT directly
    raw = {}
    for line in src_rpc.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        k, _, v = line.partition(":")
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


def compose_run_aware(run_dir: Path, third_id: str, output_root: Path):
    """Returns dict view_slot → (gt_tight, render_tight, planet_id).
    Bounding box = tight box around ALL crops in the target's image space, padded."""
    out_per_view = {}
    for slot in range(5):
        pid = planet_id_at_slot(third_id, slot)
        gt_full, rpc, full_h, full_w = get_full_view_image(third_id, slot)

        rendered = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        placed = np.zeros((full_h, full_w), dtype=np.uint8)
        # Collect ALL crop centers to size the bbox (not just placed)
        all_centers = []
        for fb in CROPS:
            lat, lon = crop_center_latlon(output_root, fb)
            col, row = rpc.projection(np.array([lat]), np.array([lon]), np.array([ALT_FOR_PROJ]))
            cx, cy = float(col[0]), float(row[0])
            all_centers.append((cx, cy))
            # place render
            rend = load_render(run_dir, slot, fb)
            if rend is None:
                continue
            x0, y0 = int(round(cx - 128)), int(round(cy - 128))
            x1, y1 = x0 + 256, y0 + 256
            sx0 = max(0, -x0); sy0 = max(0, -y0)
            dx0 = max(0, x0); dy0 = max(0, y0)
            dx1 = min(full_w, x1); dy1 = min(full_h, y1)
            sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            rendered[dy0:dy1, dx0:dx1] = rend[sy0:sy1, sx0:sx1]
            placed[dy0:dy1, dx0:dx1] = 1

        # Bounding box = around all_centers ±128 plus padding
        xs = [c for c, _ in all_centers]
        ys = [r for _, r in all_centers]
        bb_x0 = int(min(xs)) - 128 - BBOX_PADDING
        bb_y0 = int(min(ys)) - 128 - BBOX_PADDING
        bb_x1 = int(max(xs)) + 128 + BBOX_PADDING
        bb_y1 = int(max(ys)) + 128 + BBOX_PADDING
        bb_x0 = max(0, bb_x0); bb_y0 = max(0, bb_y0)
        bb_x1 = min(full_w, bb_x1); bb_y1 = min(full_h, bb_y1)
        if bb_x1 <= bb_x0 or bb_y1 <= bb_y0:
            continue
        gt_t = gt_full[bb_y0:bb_y1, bb_x0:bb_x1]
        rend_t = rendered[bb_y0:bb_y1, bb_x0:bb_x1]
        out_per_view[slot] = (gt_t, rend_t, pid)
        print(f"  slot {slot} ({pid}): bbox {bb_x1 - bb_x0}×{bb_y1 - bb_y0}, "
              f"render coverage {int(placed[bb_y0:bb_y1, bb_x0:bb_x1].mean() * 100)}%")
    return out_per_view


def label_panel(arr, text, font_size=36):
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


def upscale(arr, factor):
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((w * factor, h * factor), Image.LANCZOS))


def pad_to(arr, h, w):
    out = np.zeros((h, w, 3), dtype=arr.dtype)
    oh, ow = arr.shape[:2]
    out[:oh, :ow] = arr
    return out


for third_id, run_dir in RUNS:
    print(f"\n=== Run: ctx = 0012 + 0018 + {third_id} ===")
    output_root = REPO / "CreatDataset/example/output256/test"
    # NOTE: output256 currently reflects only the LAST run. For the earlier runs we need to
    # rebuild output256 to match that run's third_id. Otherwise crop latlonalt_bbx files differ.
    # Workaround: redo preprocessing for each run before composing.
    import subprocess
    subprocess.run(
        f"rm -rf CreatDataset/example/input2048/test/image/*/*.tif "
        f"CreatDataset/example/input2048/test/height/*/*.tif "
        f"CreatDataset/example/output256/test/* 2>/dev/null",
        shell=True, cwd=str(REPO))
    # Patch convert_planet.py to this run's mapping
    fixed = ["0012", "0018"]
    used = set(fixed + [third_id])
    rem = [v for v in ["0012", "0018", "0010", "0005", "0006"] if v not in used]
    ALL_STEMS = {
        "0012": "20210209_050530_ssc10d1_0012",
        "0018": "20220512_234256_ss02d3_0018",
        "0010": "20200308_021307_ssc2d2_0010",
        "0005": "20200308_021307_ssc2d1_0005",
        "0006": "20200308_021307_ssc2d1_0006",
    }
    slots = [(ALL_STEMS[fixed[0]], 0, "001"),
             (ALL_STEMS[fixed[1]], 1, "002"),
             (ALL_STEMS[third_id], 2, "003"),
             (ALL_STEMS[rem[0]],   3, "004"),
             (ALL_STEMS[rem[1]],   4, "005")]
    import re
    src = (Path("/tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad/convert_planet.py")).read_text()
    new_map = "VIEW_MAP = [\n"
    for stem, slot, rgb in slots:
        new_map += f'    ("{stem}", {slot}, "{rgb}"),\n'
    new_map += "]\n"
    src = re.sub(r"VIEW_MAP = \[.*?\n\]\n", new_map, src, count=1, flags=re.DOTALL)
    Path("/tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad/convert_planet.py").write_text(src)
    subprocess.run("python /tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad/convert_planet.py", shell=True, cwd=str(REPO))
    subprocess.run("python satellite_sfm_crop2048to256.py --input_folder ./example/input2048 --output_folder ./example/output256 --splits test --disable_srtm4 --view-mode fixed >/dev/null 2>&1", shell=True, cwd=str(REPO / "CreatDataset"))
    subprocess.run("python /tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad/keep_common_crops.py >/dev/null 2>&1", shell=True, cwd=str(REPO))

    per_view = compose_run_aware(run_dir, third_id, output_root)
    if not per_view:
        print("  no views composed — skipping")
        continue

    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Save individual per-view standalone files
    for slot, (gt, rend, pid) in per_view.items():
        Image.fromarray(rend).save(visuals_dir / f"view{slot}_{pid}_render.png")
        Image.fromarray(gt).save(visuals_dir / f"view{slot}_{pid}_gt.png")

    # 5-view grid: top row GTs, bottom row renders
    max_h = max(t[0].shape[0] for t in per_view.values())
    max_w = max(t[0].shape[1] for t in per_view.values())
    F = 2
    cols = []
    for slot in sorted(per_view):
        gt, rend, pid = per_view[slot]
        desc, _ = VIEW_INFO[pid]
        gt_p = pad_to(gt, max_h, max_w)
        rd_p = pad_to(rend, max_h, max_w)
        col = np.concatenate([
            label_panel(upscale(gt_p, F), f"GT  {pid}  ({desc})"),
            label_panel(upscale(rd_p, F), f"Render @ {pid}"),
        ], axis=0)
        cols.append(col)
    grid = np.concatenate(cols, axis=1)
    Image.fromarray(grid).save(visuals_dir / "all5_views_lotte_wide.png")
    print(f"  → {visuals_dir / 'all5_views_lotte_wide.png'}  ({grid.shape[1]}×{grid.shape[0]})")
