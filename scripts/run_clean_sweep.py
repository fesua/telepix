"""Sweep 3 ctx configurations and produce the clean output layout the user requested:

run_<TS>_ctx_0012-0018-<third>/
├── cam_0012/ {gt.png, render.png, depth.png}
├── cam_0018/ {gt.png, render.png, depth.png}
├── cam_<third>/ {gt.png, render.png, depth.png}
├── full.ply
└── heights.json   # {cam_id: {camera_max_height_m, gaussians_max_height_m}}
"""
import json, re, subprocess, sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")

REPO = Path("/home/imlab-p6000/js/SkySplat-main")
SCRATCH = Path("/tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad")
ALL_STEMS = {
    "0012": "20210209_050530_ssc10d1_0012",
    "0018": "20220512_234256_ss02d3_0018",
    "0010": "20200308_021307_ssc2d2_0010",
    "0005": "20200308_021307_ssc2d1_0005",
    "0006": "20200308_021307_ssc2d1_0006",
}
FIXED = ["0012", "0018"]
THIRD_CTX = ["0010", "0006", "0005"]


def run(cmd, cwd=None, capture=False):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=capture, text=True)
    return r.stdout if capture else r.returncode


def patch_view_map(third: str) -> list[tuple[str, int, str]]:
    used = set(FIXED + [third])
    rem = [v for v in ALL_STEMS if v not in used]
    slots = [(ALL_STEMS[FIXED[0]], 0, "001"),
             (ALL_STEMS[FIXED[1]], 1, "002"),
             (ALL_STEMS[third],    2, "003"),
             (ALL_STEMS[rem[0]],   3, "004"),
             (ALL_STEMS[rem[1]],   5 - 5 + 4, "005")]
    new_map = "VIEW_MAP = [\n"
    for stem, slot, rgb in slots:
        new_map += f'    ("{stem}", {slot}, "{rgb}"),\n'
    new_map += "]\n"
    src = (SCRATCH / "convert_planet.py").read_text()
    src = re.sub(r"VIEW_MAP = \[.*?\n\]\n", new_map, src, count=1, flags=re.DOTALL)
    (SCRATCH / "convert_planet.py").write_text(src)
    return slots


def get_full_view_and_rpc(planet_id: str):
    """Re-extract full Planet image + RPC from Dataset/."""
    from osgeo import gdal
    from preprocess.rpc_model import RPCModel
    gdal.UseExceptions()
    stem = ALL_STEMS[planet_id]
    src_tif = REPO / "Dataset" / f"{stem}_basic_analytic.tif"
    src_rpc_txt = REPO / "Dataset" / f"{stem}_basic_analytic_RPC.TXT"
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
    for line in src_rpc_txt.read_text().splitlines():
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


def crop_center_latlon(output_root: Path, fb: int):
    p = output_root / "cameras_others/0" / f"JAX_Tile_999_RGB_001_crop_{fb}_latlonalt_bbx.json"
    if not p.exists():
        return None
    bbx = json.loads(p.read_text())
    return ((bbx["lat_minmax"][0] + bbx["lat_minmax"][1]) / 2.0,
            (bbx["lon_minmax"][0] + bbx["lon_minmax"][1]) / 2.0)


def stitch_into_view(planet_id: str, slot: int, run_dir: Path, output_root: Path):
    """Build GT, render, and depth mosaics in the planet_id camera's pixel coords."""
    from matplotlib import cm
    gt_full, rpc, fh, fw = get_full_view_and_rpc(planet_id)

    render_canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
    depth_canvas = np.full((fh, fw), np.nan, dtype=np.float32)
    placed = np.zeros((fh, fw), dtype=np.uint8)

    rgb_no = ["001", "002", "003", "004", "005"][slot]
    color_dir = run_dir / "renders/re10k/JAX_Tile_999/color"
    height_dir = run_dir / "renders/re10k/JAX_Tile_999/height"

    crops = sorted(int(re.search(r"crop_(\d+)\.tif$", p.name).group(1))
                   for p in (output_root / "image/0").glob("*.tif"))
    centers = []
    for fb in crops:
        ll = crop_center_latlon(output_root, fb)
        if ll is None:
            continue
        lat, lon = ll
        col, row = rpc.projection(np.array([lat]), np.array([lon]), np.array([50.0]))
        cx, cy = float(col[0]), float(row[0])
        centers.append((cx, cy))
        # Render
        rend_p = color_dir / f"tgt_view_{slot}_JAX_Tile_999_RGB_{rgb_no}_crop_{fb}.png"
        if rend_p.exists():
            rend = np.asarray(Image.open(rend_p))
            x0, y0 = int(round(cx - 128)), int(round(cy - 128))
            x1, y1 = x0 + 256, y0 + 256
            sx0 = max(0, -x0); sy0 = max(0, -y0)
            dx0 = max(0, x0); dy0 = max(0, y0)
            dx1 = min(fw, x1); dy1 = min(fh, y1)
            sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
            if dx1 > dx0 and dy1 > dy0:
                render_canvas[dy0:dy1, dx0:dx1] = rend[sy0:sy1, sx0:sx1]
                placed[dy0:dy1, dx0:dx1] = 1
        # Depth (height) — only present for context-view slots (0, 1, 2)
        height_p = height_dir / f"ctx_view_{slot}_JAX_Tile_999_RGB_{rgb_no}_crop_{fb}.npy"
        if height_p.exists():
            h_arr = np.load(height_p)
            x0, y0 = int(round(cx - 128)), int(round(cy - 128))
            x1, y1 = x0 + 256, y0 + 256
            sx0 = max(0, -x0); sy0 = max(0, -y0)
            dx0 = max(0, x0); dy0 = max(0, y0)
            dx1 = min(fw, x1); dy1 = min(fh, y1)
            sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
            if dx1 > dx0 and dy1 > dy0:
                depth_canvas[dy0:dy1, dx0:dx1] = h_arr[sy0:sy1, sx0:sx1]

    # Tight-crop around the placed bbox + padding
    if not centers:
        return None
    xs = [c for c, _ in centers]; ys = [r for _, r in centers]
    pad = 80
    bx0 = max(0, int(min(xs)) - 128 - pad); by0 = max(0, int(min(ys)) - 128 - pad)
    bx1 = min(fw, int(max(xs)) + 128 + pad); by1 = min(fh, int(max(ys)) + 128 + pad)
    gt = gt_full[by0:by1, bx0:bx1]
    rend = render_canvas[by0:by1, bx0:bx1]
    dpt = depth_canvas[by0:by1, bx0:bx1]
    return gt, rend, dpt


def colormap_depth(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    from matplotlib import cm
    mask = np.isfinite(depth)
    if mask.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8), {"min": None, "max": None}
    vmin = float(np.nanmin(depth))
    vmax = float(np.nanmax(depth))
    norm = (depth - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(norm, 0, 1)
    colored = (cm.plasma(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~mask] = 0
    return colored, {"min": vmin, "max": vmax}


def build_full_ply(run_dir: Path, out_path: Path) -> dict:
    """Concatenate every chunk's .npz into a single 3DGS PLY and return stats."""
    from plyfile import PlyData, PlyElement
    npz_files = sorted((run_dir / "renders/re10k/JAX_Tile_999/gaussians").glob("*.npz"))
    if not npz_files:
        return {"gaussians_count": 0, "z_max": None}
    all_means, all_scales, all_rots, all_opac, all_sh = [], [], [], [], []
    for nz in npz_files:
        d = np.load(nz)
        all_means.append(d["means"].reshape(-1, 3))
        all_scales.append(d["scales"].reshape(-1, 3))
        all_rots.append(d["rotations"].reshape(-1, 4))
        all_opac.append(d["opacities"].reshape(-1))
        all_sh.append(d["harmonics"].reshape(-1, d["harmonics"].shape[-2], d["harmonics"].shape[-1]))
    means = np.concatenate(all_means)
    scales = np.concatenate(all_scales)
    rots = np.concatenate(all_rots)
    opac = np.concatenate(all_opac)
    sh = np.concatenate(all_sh)  # (N, d_sh, 3)

    # Truncate SH to degree 3 (16 coefs) and filter low opacity
    d_sh_out = 16
    if sh.shape[1] > d_sh_out:
        sh = sh[:, :d_sh_out, :]
    keep = opac > 0.05
    means = means[keep]; scales = scales[keep]; rots = rots[keep]
    opac = opac[keep]; sh = sh[keep]

    eps = 1e-7
    logit_opac = np.log(np.clip(opac, eps, 1 - eps) / (1 - np.clip(opac, eps, 1 - eps))).astype(np.float32)
    log_scales = np.log(np.clip(scales, eps, None)).astype(np.float32)
    f_dc = sh[:, 0, :].astype(np.float32)
    f_rest = np.concatenate([sh[:, 1:, c] for c in range(3)], axis=1).astype(np.float32)
    normals = np.zeros((means.shape[0], 3), dtype=np.float32)
    N = means.shape[0]
    n_rest = f_rest.shape[1]
    rec = np.concatenate([
        means.astype(np.float32), normals, f_dc, f_rest,
        logit_opac.reshape(-1, 1), log_scales, rots.astype(np.float32),
    ], axis=1)

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
              ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    fields += [(f"f_rest_{i}", "f4") for i in range(n_rest)]
    fields += [("opacity", "f4"),
               ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
               ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    structured = np.empty(N, dtype=fields)
    for i, (name, _) in enumerate(fields):
        structured[name] = rec[:, i]
    el = PlyElement.describe(structured, "vertex")
    PlyData([el], text=False, byte_order="<").write(str(out_path))
    return {"gaussians_count": int(N), "z_max_m": float(means[:, 2].max())}


# ============= per-config sweep =============

for third in THIRD_CTX:
    print(f"\n{'='*70}\nCONFIG ctx = 0012, 0018, {third}\n{'='*70}")
    slots = patch_view_map(third)
    # Clean + convert + crop + cleanup
    run("rm -rf CreatDataset/example/input2048/test/image/*/*.tif "
        "CreatDataset/example/input2048/test/height/*/*.tif "
        "CreatDataset/example/output256/test/* 2>/dev/null", cwd=str(REPO))
    run(f"python {SCRATCH/'convert_planet.py'}", cwd=str(REPO))
    run("python satellite_sfm_crop2048to256.py --input_folder ./example/input2048 "
        "--output_folder ./example/output256 --splits test --disable_srtm4 "
        "--view-mode fixed >/dev/null 2>&1", cwd=str(REPO/"CreatDataset"))
    run(f"python {SCRATCH/'keep_common_3views.py'} >/dev/null 2>&1", cwd=str(REPO))
    run(f"python {SCRATCH/'fixup_planet.py'} >/dev/null 2>&1", cwd=str(REPO))
    run('python -c "import json; from pathlib import Path; '
        '[p.write_text(json.dumps({\\"min_height\\": 25.0, \\"max_height\\": 600.0})) '
        "for p in Path('CreatDataset/example/output256/test/height').glob('*/*_height_minmax.json')]\"",
        cwd=str(REPO))

    # Inference (heavy run dir at temp loc, we'll repackage)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_dir = REPO / f"outputs/_raw_{ts}_ctx_0012-0018-{third}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = (f"CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k "
           f"checkpointing.load=./checkpoints/SkySplat_baseline.ckpt mode=test "
           f"test.compute_scores=true test.output_path={raw_dir}/renders")
    out = run(cmd, cwd=str(REPO), capture=True)
    for line in out.splitlines():
        if line.startswith(("psnr ", "ssim ", "lpips ")):
            print(f"  {line}")

    # Clean output layout
    clean_dir = REPO / f"outputs/run_{ts}_ctx_0012-0018-{third}"
    clean_dir.mkdir(parents=True, exist_ok=True)

    output_root = REPO / "CreatDataset/example/output256/test"
    heights_info = {}
    # ALL 5 cameras (slots 0..4). Slots 0/1/2 are context (have depth);
    # slots 3/4 are held-out (only gt + render — model doesn't predict heights there).
    used = set(FIXED + [third])
    rem = [v for v in ALL_STEMS if v not in used]
    all_planet_ids = [FIXED[0], FIXED[1], third, rem[0], rem[1]]
    for slot, pid in enumerate(all_planet_ids):
        is_context = slot < 3
        cam_dir = clean_dir / f"cam_{pid}"
        cam_dir.mkdir(exist_ok=True)
        result = stitch_into_view(pid, slot, raw_dir, output_root)
        if result is None:
            continue
        gt, rend, dpt = result
        Image.fromarray(gt).save(cam_dir / "gt.png")
        Image.fromarray(rend).save(cam_dir / "render.png")
        entry = {"role": "context" if is_context else "held_out"}
        if is_context and np.isfinite(dpt).any():
            depth_rgb, dstats = colormap_depth(dpt)
            Image.fromarray(depth_rgb).save(cam_dir / "depth.png")
            entry["camera_max_height_m"] = dstats["max"]
            entry["camera_min_height_m"] = dstats["min"]
        else:
            # No depth available for held-out views (model doesn't predict heights
            # outside context views). Skip depth.png.
            entry["camera_max_height_m"] = None
            entry["camera_min_height_m"] = None
        heights_info[f"cam_{pid}"] = entry

    # Full PLY + gaussian max height
    ply_path = clean_dir / "full.ply"
    ply_stats = build_full_ply(raw_dir, ply_path)
    for cam in heights_info:
        heights_info[cam]["gaussians_max_height_m"] = ply_stats.get("z_max_m")
    heights_info["_gaussian_total_count"] = ply_stats.get("gaussians_count")

    (clean_dir / "heights.json").write_text(json.dumps(heights_info, indent=2))

    print(f"  → {clean_dir}")
    for p in sorted(clean_dir.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(clean_dir)}  ({p.stat().st_size/1e6:.1f} MB)")
