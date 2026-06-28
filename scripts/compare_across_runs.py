"""For each target viewpoint, show how the render changes when the THIRD context view is
varied across {0010, 0006, 0005}, with 0012 and 0018 held fixed.

Layout per target view:
  Row 0: GT (one image, same across runs)
  Row 1: Render ctx=0012+0018+0010
  Row 2: Render ctx=0012+0018+0006
  Row 3: Render ctx=0012+0018+0005
"""
import re, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/imlab-p6000/js/SkySplat-main")
RUNS = [
    ("0010", REPO / "outputs/run_2026-06-28_11-34-21_ctx_0012-0018-0010"),
    ("0006", REPO / "outputs/run_2026-06-28_11-34-47_ctx_0012-0018-0006"),
    ("0005", REPO / "outputs/run_2026-06-28_11-35-12_ctx_0012-0018-0005"),
]
OUT = REPO / "outputs/across_runs_compare"
OUT.mkdir(parents=True, exist_ok=True)

# Each run's slot→Planet ID mapping (must match the sweep)
def slot_to_pid(third_id):
    fixed = ["0012", "0018"]
    used = set(fixed + [third_id])
    rem = [v for v in ["0012", "0018", "0010", "0005", "0006"] if v not in used]
    return [fixed[0], fixed[1], third_id, rem[0], rem[1]]


def find_gt(run_dir, pid):
    cands = sorted(run_dir.glob(f"visuals/view*_{pid}_gt.png"))
    return np.asarray(Image.open(cands[0])) if cands else None

def find_render(run_dir, pid):
    cands = sorted(run_dir.glob(f"visuals/view*_{pid}_render.png"))
    return np.asarray(Image.open(cands[0])) if cands else None


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


for target_pid in ["0012", "0018", "0010", "0005", "0006"]:
    rows = []
    # Top: GT (use the run where this view exists)
    gt_arr = None
    for _, run_dir in RUNS:
        g = find_gt(run_dir, target_pid)
        if g is not None:
            gt_arr = g; break
    if gt_arr is None:
        print(f"skip {target_pid} — no GT found in any run")
        continue
    max_h, max_w = gt_arr.shape[:2]
    # Renders per run (some may be None if pid not in slots? actually all pids are in all runs)
    renders = []
    for third_id, run_dir in RUNS:
        r = find_render(run_dir, target_pid)
        if r is not None:
            max_h = max(max_h, r.shape[0])
            max_w = max(max_w, r.shape[1])
        renders.append((third_id, r))

    F = 2
    panels = [label_panel(upscale(pad_to(gt_arr, max_h, max_w), F),
                          f"GT  view {target_pid}")]
    for third_id, r in renders:
        if r is None:
            r = np.zeros_like(gt_arr)
        in_ctx = target_pid in {"0012", "0018", third_id}
        tag = "context (train view)" if in_ctx else "held-out (novel)"
        panels.append(
            label_panel(upscale(pad_to(r, max_h, max_w), F),
                        f"Render ctx=0012+0018+{third_id}  ·  {tag}"))
    grid = np.concatenate(panels, axis=0)
    out_path = OUT / f"target_{target_pid}_across_runs.png"
    Image.fromarray(grid).save(out_path)
    print(f"saved {out_path.name}  ({grid.shape[1]}×{grid.shape[0]})")

print(f"\nAll under: {OUT}")
