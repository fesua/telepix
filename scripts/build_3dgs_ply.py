"""Build canonical 3DGS-format PLY from saved Gaussian .npz dumps.

Output is loadable in:
  * SuperSplat (https://playcanvas.com/supersplat/editor) — web viewer, drag the .ply
  * antimatter15's viewer (https://antimatter15.com/splat/)
  * gaussian-splatting reference INRIA code
  * Polycam, Luma AI, etc.

Convention:
  * x, y, z       — Gaussian centers (we keep ENU; viewer treats it as world coords)
  * nx, ny, nz    — zeros (not used by 3DGS but required by PLY parsers)
  * f_dc_0..2     — DC SH (RGB), raw
  * f_rest_*      — higher-order SH, channel-major (R coefs, then G, then B)
  * opacity       — logit(opacity)        (rasterizer applies sigmoid)
  * scale_0..2    — log(scale)            (rasterizer applies exp)
  * rot_0..3      — quaternion (w, x, y, z)
"""
import sys
from pathlib import Path

import numpy as np

# === inputs ===
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-34-50_ctx_0012-0018-0006_3dgs_ply"
)
SH_DEGREE_OUT = 3            # truncate sh_degree=4 → 3 (SuperSplat supports up to 3)
OPACITY_THRESH = 0.05        # drop very transparent gaussians
SUBSAMPLE_PER_CHUNK = None   # set e.g. 32_000 to thin out for SuperSplat

GAUSS_DIR = RUN / "renders/re10k/JAX_Tile_999/gaussians"
OUT_PER_CHUNK = RUN / "renders/re10k/JAX_Tile_999/gaussians_3dgs"
OUT_PER_CHUNK.mkdir(parents=True, exist_ok=True)
OUT_MERGED = RUN / "visuals/merged_3dgs.ply"
OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)


def build_3dgs_ply(npz_path: Path) -> tuple[np.ndarray, int]:
    """Return packed-attribute array + num_gaussians."""
    d = np.load(npz_path)
    # Shapes
    #   means:      (1, V, N, 3)
    #   scales:     (1, V, N, 3)
    #   rotations:  (1, V, N, 4)        # quat (w,x,y,z) per gaussian_adapter
    #   opacities:  (1, V, N, 1)
    #   harmonics:  (1, V, N, d_sh, 3)
    means = d["means"].reshape(-1, 3)
    scales = d["scales"].reshape(-1, 3)
    rots = d["rotations"].reshape(-1, 4)
    opac = d["opacities"].reshape(-1)
    sh = d["harmonics"]              # (1, V, N, d_sh, 3)
    sh = sh.reshape(-1, sh.shape[-2], sh.shape[-1])   # (N_total, d_sh, 3)
    N = means.shape[0]
    assert scales.shape[0] == N == rots.shape[0] == opac.shape[0] == sh.shape[0], \
        f"shape mismatch: {means.shape}, {scales.shape}, {rots.shape}, {opac.shape}, {sh.shape}"
    d_sh_in = sh.shape[1]              # 25 for sh_degree=4
    d_sh_out = (SH_DEGREE_OUT + 1) ** 2
    if d_sh_in > d_sh_out:
        sh = sh[:, :d_sh_out, :]       # truncate higher orders

    # Opacity filter
    mask = opac > OPACITY_THRESH
    if mask.sum() == 0:
        mask[:] = True
    means = means[mask]; scales = scales[mask]; rots = rots[mask]
    opac = opac[mask]; sh = sh[mask]
    N = means.shape[0]

    # Optional subsample
    if SUBSAMPLE_PER_CHUNK is not None and N > SUBSAMPLE_PER_CHUNK:
        idx = np.random.default_rng(0).choice(N, size=SUBSAMPLE_PER_CHUNK, replace=False)
        means = means[idx]; scales = scales[idx]; rots = rots[idx]
        opac = opac[idx]; sh = sh[idx]
        N = means.shape[0]

    # 3DGS convention: opacity = logit, scales = log
    eps = 1e-7
    opac_c = np.clip(opac, eps, 1 - eps)
    logit_opac = np.log(opac_c / (1 - opac_c)).astype(np.float32)
    log_scales = np.log(np.clip(scales, eps, None)).astype(np.float32)

    # DC and rest SH
    f_dc = sh[:, 0, :].astype(np.float32)           # (N, 3) — channel order R,G,B
    # f_rest: channel-major flatten, INRIA convention
    # sh[:, 1:, c] for c in 0..2 → concatenate → (N, (d_sh_out-1)*3)
    f_rest_channels = []
    for c in range(3):
        f_rest_channels.append(sh[:, 1:, c])         # (N, d_sh_out - 1)
    f_rest = np.concatenate(f_rest_channels, axis=1).astype(np.float32)
    n_rest = f_rest.shape[1]

    # Pack into structured-like array.  We'll just emit a binary float32 blob
    # in the property order declared in the header.
    normals = np.zeros((N, 3), dtype=np.float32)
    rec = np.concatenate([
        means.astype(np.float32),     # 3
        normals,                       # 3
        f_dc,                          # 3
        f_rest,                        # n_rest
        logit_opac[:, None],           # 1
        log_scales,                    # 3
        rots.astype(np.float32),       # 4
    ], axis=1)
    return rec, N, n_rest


def write_ply(out_path: Path, rec: np.ndarray, n_rest: int):
    N = rec.shape[0]
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
    ]
    for i in range(n_rest):
        lines.append(f"property float f_rest_{i}")
    lines.append("property float opacity")
    for i in range(3):
        lines.append(f"property float scale_{i}")
    for i in range(4):
        lines.append(f"property float rot_{i}")
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")
    with open(out_path, "wb") as fp:
        fp.write(header)
        fp.write(rec.astype(np.float32).tobytes())


def main():
    npz_files = sorted(GAUSS_DIR.glob("*.npz"))
    print(f"Found {len(npz_files)} .npz files in {GAUSS_DIR}")
    if not npz_files:
        sys.exit(1)

    merged_chunks = []
    total = 0
    n_rest_global = None
    for nz in npz_files:
        rec, n, n_rest = build_3dgs_ply(nz)
        per_path = OUT_PER_CHUNK / (nz.stem + "_3dgs.ply")
        write_ply(per_path, rec, n_rest)
        size_mb = per_path.stat().st_size / 1e6
        print(f"  {nz.stem}: {n:>7,} gaussians, n_rest={n_rest}, {size_mb:.1f} MB → {per_path.name}")
        merged_chunks.append(rec)
        total += n
        n_rest_global = n_rest

    merged = np.concatenate(merged_chunks, axis=0)
    write_ply(OUT_MERGED, merged, n_rest_global)
    print(f"\nMerged: {total:,} gaussians → {OUT_MERGED}  ({OUT_MERGED.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
