"""Run preprocessing + inference for three context configurations:
  ctx = {0012, 0018, X} where X ∈ {0010, 0006, 0005}.
Reuses the existing 0012/0018 fixed slots; only the 3rd context view changes.
"""
import json
import os
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path("/home/imlab-p6000/js/SkySplat-main")
SCRATCH = Path("/tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad")
CONVERT = SCRATCH / "convert_planet.py"
KEEP = SCRATCH / "keep_common_3views.py"
FIXUP = SCRATCH / "fixup_planet.py"

# Full list of all 5 candidate views (ID → planet stem)
ALL = {
    "0012": "20210209_050530_ssc10d1_0012",
    "0018": "20220512_234256_ss02d3_0018",
    "0010": "20200308_021307_ssc2d2_0010",
    "0005": "20200308_021307_ssc2d1_0005",
    "0006": "20200308_021307_ssc2d1_0006",
}

# Fixed two context views (slots 0 and 1)
FIXED_CTX = ["0012", "0018"]
# We'll sweep the third context view through these:
THIRD_CTX = ["0006", "0005"]   # 0010 already done in run_2026-06-28_12-02-52_*


def write_view_map(third_ctx_id: str):
    """Mutate convert_planet.py VIEW_MAP for the given configuration.

    Slot mapping:
      0: 0012  (ref / context)
      1: 0018  (context)
      2: <third_ctx>  (context, varying)
      3, 4: the two remaining IDs from the unused set (held-out / extra render targets)
    """
    used = set(FIXED_CTX + [third_ctx_id])
    remaining = [v for v in ALL if v not in used]
    assert len(remaining) == 2
    slots = [
        (ALL[FIXED_CTX[0]], 0, "001", "ref ctx 0012"),
        (ALL[FIXED_CTX[1]], 1, "002", "ctx 0018"),
        (ALL[third_ctx_id], 2, "003", f"ctx {third_ctx_id} (varying)"),
        (ALL[remaining[0]], 3, "004", f"held-out / extra target {remaining[0]}"),
        (ALL[remaining[1]], 4, "005", f"held-out / extra target {remaining[1]}"),
    ]
    lines = ['VIEW_MAP = [\n']
    for stem, slot, rgb, note in slots:
        lines.append(f'    ("{stem}", {slot}, "{rgb}"),  # {note}\n')
    lines.append(']\n')
    new_map = "".join(lines)

    src = CONVERT.read_text()
    import re
    src = re.sub(r"VIEW_MAP = \[.*?\n\]\n", new_map, src, count=1, flags=re.DOTALL)
    CONVERT.write_text(src)
    return slots


def run(cmd, cwd=None, capture=False):
    if capture:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=True)
        return r.stdout, r.stderr, r.returncode
    return subprocess.run(cmd, cwd=cwd, shell=True).returncode


for third_id in THIRD_CTX:
    print(f"\n{'=' * 70}\nConfiguration: ctx = 0012, 0018, {third_id}\n{'=' * 70}")
    slots = write_view_map(third_id)
    for stem, slot, rgb, note in slots:
        print(f"  slot {slot}: {stem}  ({note})")

    # Clean + re-convert + re-preprocess
    run("rm -rf CreatDataset/example/input2048/test/image/*/*.tif "
        "CreatDataset/example/input2048/test/height/*/*.tif "
        "CreatDataset/example/output256/test/* 2>/dev/null", cwd=str(REPO))
    run(f"python {CONVERT}", cwd=str(REPO))
    run(f"python satellite_sfm_crop2048to256.py "
        f"--input_folder ./example/input2048 "
        f"--output_folder ./example/output256 "
        f"--splits test --disable_srtm4 --view-mode fixed",
        cwd=str(REPO / "CreatDataset"))
    run(f"python {KEEP}", cwd=str(REPO))
    run(f"python {FIXUP}", cwd=str(REPO))
    # Overwrite minmax
    for p in Path(REPO / "CreatDataset/example/output256/test/height").glob("*/*_height_minmax.json"):
        p.write_text(json.dumps({"min_height": 25.0, "max_height": 600.0}))

    # Timestamped output dir with explicit context view IDs in the name
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = REPO / f"outputs/run_{ts}_ctx_0012-0018-{third_id}_3vfilter"
    (run_dir / "visuals").mkdir(parents=True, exist_ok=True)
    Path("/tmp/run_ts.txt").write_text(f"RUN_DIR={run_dir}\n")

    # Inference
    cmd = (f"CUDA_VISIBLE_DEVICES=0 python -m src.main "
           f"+experiment=re10k "
           f"checkpointing.load=./checkpoints/SkySplat_baseline.ckpt "
           f"mode=test test.compute_scores=true "
           f"test.output_path={run_dir}/renders")
    out, err, rc = run(cmd, cwd=str(REPO), capture=True)
    # Print metrics
    for line in out.splitlines():
        if any(line.startswith(k) for k in ("psnr ", "ssim ", "lpips ", "mae ", "rmse ")):
            print(f"  {line}")
    print(f"  → {run_dir}")
