"""TIF-level histogram matching variant of convert_planet.py.

For each Planet TIF, apply skimage.match_histograms against 0010 (2020-03-08) in the
raw UInt16 (h, w, 3) RGB space BEFORE percentile/min-max conversion. The reference
TIF (0010) is passed through unchanged.

Output goes to a separate input2048_histmatch_tif/ tree so the standard input2048/
isn't clobbered.
"""
import json, sys, re
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal, gdalconst
from skimage.exposure import match_histograms

sys.path.insert(0, "/home/imlab-p6000/js/SkySplat-main/CreatDataset")

gdal.UseExceptions()

DATASET_DIR = Path("/home/imlab-p6000/js/SkySplat-main/Dataset")
OUT_ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/input2048_histmatch_tif/test")
TILE_ID = "999"

# Reference Planet ID — 2020-03-08 d2 — the user's calibration reference
REFERENCE_PLANET_ID = "20200308_021307_ssc2d2_0010"

# VIEW_MAP gets patched in place by the sweep driver. Default mapping (ctx=0010).
VIEW_MAP = [
    ("20210209_050530_ssc10d1_0012", 0, "001"),
    ("20220512_234256_ss02d3_0018", 1, "002"),
    ("20200308_021307_ssc2d1_0005", 2, "003"),
    ("20200308_021307_ssc2d2_0010", 3, "004"),
    ("20200308_021307_ssc2d1_0006", 4, "005"),
]


def read_planet_rgb_uint16(stem: str):
    """Returns (h, w, 3) float32 RGB (from Planet bands [R, G, B] = [2, 1, 0])."""
    p = DATASET_DIR / f"{stem}_basic_analytic.tif"
    ds = gdal.Open(str(p), gdal.GA_ReadOnly)
    arr = ds.ReadAsArray()  # (4, h, w) UInt16
    assert arr.shape[0] == 4, f"unexpected band count: {arr.shape}"
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1).astype(np.float32)  # (h, w, 3) R,G,B
    ds = None
    return rgb


def read_rpc_txt(path: Path) -> dict:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        raw[key.strip()] = val.strip()
    out: dict[str, str] = {}
    for k in ["LINE_OFF", "SAMP_OFF", "LAT_OFF", "LONG_OFF", "HEIGHT_OFF",
              "LINE_SCALE", "SAMP_SCALE", "LAT_SCALE", "LONG_SCALE", "HEIGHT_SCALE"]:
        out[k] = raw[k]
    for coeff_key in ["LINE_NUM_COEFF", "LINE_DEN_COEFF", "SAMP_NUM_COEFF", "SAMP_DEN_COEFF"]:
        out[coeff_key] = " ".join(raw[f"{coeff_key}_{i}"] for i in range(1, 21))
    return out


def make_nitf_idatim(iso: str) -> str:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d%H%M%S")


def to_uint8_minmax(rgb_f32: np.ndarray) -> np.ndarray:
    """Convert (h, w, 3) float32 → uint8 with per-channel 2-98 percentile stretch.
    Single global min-max produces 17/255-mean tinted output because Planet UInt16
    has very few values near 65535 (saturated highlights); using percentile gives
    proper dynamic range. Channel-wise stretch matches baseline convert_planet.py."""
    out = np.zeros(rgb_f32.shape, dtype=np.uint8)
    for c in range(rgb_f32.shape[-1]):
        b = rgb_f32[..., c]
        lo, hi = np.percentile(b, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        b = np.clip((b - lo) / (hi - lo) * 255.0, 0, 255)
        out[..., c] = b.astype(np.uint8)
    return out


def write_tif(out_path: Path, img_uint8_rgb: np.ndarray, rpc_meta: dict, nitf_idatim: str):
    """img_uint8_rgb shape (h, w, 3) → write as 3-band GeoTIFF with RPC + NITF_IDATIM."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GDAL wants (bands, h, w) — split.
    bands = img_uint8_rgb.transpose(2, 0, 1)
    h, w = bands.shape[1:]
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(out_path), w, h, 3, gdalconst.GDT_Byte, options=["COMPRESS=LZW"])
    ds.SetMetadata(rpc_meta, "RPC")
    ds.SetMetadataItem("NITF_IDATIM", nitf_idatim)
    for i in range(3):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(bands[i])
        band.FlushCache()
    ds = None


def main():
    print(f"Reference image (UInt16 RGB float32) = {REFERENCE_PLANET_ID}")
    ref_rgb = read_planet_rgb_uint16(REFERENCE_PLANET_ID)
    print(f"  shape={ref_rgb.shape}  range=[{ref_rgb.min():.0f}, {ref_rgb.max():.0f}]")

    for planet_id, view_slot, rgb_no in VIEW_MAP:
        src_rpc_path = DATASET_DIR / f"{planet_id}_basic_analytic_RPC.TXT"
        src_meta_path = DATASET_DIR / f"{planet_id}_metadata.json"

        rgb = read_planet_rgb_uint16(planet_id)
        if planet_id != REFERENCE_PLANET_ID:
            matched = match_histograms(rgb, ref_rgb, channel_axis=-1)
            print(f"  view{view_slot} ({planet_id}): MATCHED to ref. src range [{rgb.min():.0f}, {rgb.max():.0f}]  matched range [{matched.min():.0f}, {matched.max():.0f}]")
        else:
            matched = rgb
            print(f"  view{view_slot} ({planet_id}): REFERENCE, no match. range [{rgb.min():.0f}, {rgb.max():.0f}]")
        u8 = to_uint8_minmax(matched)

        rpc_meta = read_rpc_txt(src_rpc_path)
        nitf = make_nitf_idatim(json.loads(src_meta_path.read_text())["properties"]["acquired"])
        out_img = OUT_ROOT / "image" / str(view_slot) / f"JAX_Tile_{TILE_ID}_RGB_{rgb_no}.tif"
        write_tif(out_img, u8, rpc_meta, nitf)

        # Dummy zero XYZ tif (preprocessor reads it, model doesn't use)
        h, w = u8.shape[:2]
        out_hei = OUT_ROOT / "height" / str(view_slot) / f"JAX_Tile_{TILE_ID}_XYZ_{rgb_no}.tif"
        out_hei.parent.mkdir(parents=True, exist_ok=True)
        hdrv = gdal.GetDriverByName("GTiff")
        hds = hdrv.Create(str(out_hei), w, h, 1, gdalconst.GDT_Float32)
        hds.GetRasterBand(1).WriteArray(np.zeros((h, w), dtype=np.float32))
        hds.GetRasterBand(1).FlushCache()
        hds = None


if __name__ == "__main__":
    main()
