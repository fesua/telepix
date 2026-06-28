"""Convert 5 Planet SkySat tiles into SkySplat-compatible input2048 layout.

Each Planet tile is 2560x1080, 4-band (B,G,R,NIR) UInt16 with RPC in a sibling .TXT.
We:
  * pick bands 3,2,1 (R,G,B), apply percentile stretch -> uint8
  * embed RPC into TIFF metadata (GDAL 'RPC' domain)
  * embed a dummy NITF_IDATIM stamp (the dataset's parser expects it)
  * rename to JAX_Tile_999_RGB_00{N}.tif
  * also write a zero-filled XYZ height tif (preprocessor & dataset both expect it;
    model never uses it at inference when compute_scores=False)

View mapping:
  view 0/1/2 (context):  0005, 0006, 0010   (all 2020-03-08)
  view 3/4   (target):   0012, 0018         (held out for PSNR if enabled)
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal, gdalconst

gdal.UseExceptions()

DATASET_DIR = Path("/home/imlab-p6000/js/SkySplat-main/Dataset")
OUT_ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/input2048/test")

VIEW_MAP = [
    ("20210209_050530_ssc10d1_0012", 0, "001"),
    ("20220512_234256_ss02d3_0018", 1, "002"),
    ("20200308_021307_ssc2d1_0005", 2, "003"),
    ("20200308_021307_ssc2d2_0010", 3, "004"),
    ("20200308_021307_ssc2d1_0006", 4, "005"),
]
TILE_ID = "999"


def read_rpc_txt(path: Path) -> dict:
    """Parse Planet's RPC TXT into the dict GDAL accepts via SetMetadata(..., 'RPC').
    Planet's file stores each polynomial coefficient on its own line (LINE_NUM_COEFF_1..20)
    but GDAL expects a single key (LINE_NUM_COEFF) with 20 space-separated floats."""
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        raw[key.strip()] = val.strip()

    out: dict[str, str] = {}
    scalar_keys = [
        "LINE_OFF", "SAMP_OFF", "LAT_OFF", "LONG_OFF", "HEIGHT_OFF",
        "LINE_SCALE", "SAMP_SCALE", "LAT_SCALE", "LONG_SCALE", "HEIGHT_SCALE",
    ]
    for k in scalar_keys:
        out[k] = raw[k]

    for coeff_key in ["LINE_NUM_COEFF", "LINE_DEN_COEFF", "SAMP_NUM_COEFF", "SAMP_DEN_COEFF"]:
        parts = [raw[f"{coeff_key}_{i}"] for i in range(1, 21)]
        out[coeff_key] = " ".join(parts)
    return out


def stretch_to_uint8(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-band percentile stretch -> uint8."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for c in range(arr.shape[0]):
        band = arr[c].astype(np.float32)
        p_lo, p_hi = np.percentile(band, [lo, hi])
        if p_hi <= p_lo:
            p_hi = p_lo + 1.0
        band = np.clip((band - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
        out[c] = band.astype(np.uint8)
    return out


def make_nitf_idatim(acquired_iso: str) -> str:
    dt = datetime.fromisoformat(acquired_iso.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d%H%M%S")


def write_tif(out_path: Path, img_uint8_rgb: np.ndarray, rpc_meta: dict, nitf_idatim: str):
    """Write 3-band uint8 GeoTIFF with RPC + NITF_IDATIM metadata embedded."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img_uint8_rgb.shape[1:]
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(out_path), w, h, 3, gdalconst.GDT_Byte, options=["COMPRESS=LZW"])
    ds.SetMetadata(rpc_meta, "RPC")
    ds.SetMetadataItem("NITF_IDATIM", nitf_idatim)
    for i in range(3):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(img_uint8_rgb[i])
        band.FlushCache()
    ds = None


def main():
    for planet_id, view_slot, rgb_no in VIEW_MAP:
        src_tif = DATASET_DIR / f"{planet_id}_basic_analytic.tif"
        src_rpc = DATASET_DIR / f"{planet_id}_basic_analytic_RPC.TXT"
        src_meta = DATASET_DIR / f"{planet_id}_metadata.json"

        ds = gdal.Open(str(src_tif), gdal.GA_ReadOnly)
        arr = ds.ReadAsArray()  # (bands, h, w), UInt16
        assert arr.shape[0] == 4, f"unexpected band count: {arr.shape}"
        rgb = np.stack([arr[2], arr[1], arr[0]], axis=0)  # band3=R, band2=G, band1=B
        rgb_u8 = stretch_to_uint8(rgb)
        ds = None

        rpc_meta = read_rpc_txt(src_rpc)
        meta = json.loads(src_meta.read_text())
        nitf = make_nitf_idatim(meta["properties"]["acquired"])

        out_img = OUT_ROOT / "image" / str(view_slot) / f"JAX_Tile_{TILE_ID}_RGB_{rgb_no}.tif"
        write_tif(out_img, rgb_u8, rpc_meta, nitf)

        # Dummy XYZ height map: zeros, same h/w, single-band Float32.
        # Preprocessor reads it but the model only consumes it when compute_scores=True.
        h, w = rgb_u8.shape[1:]
        out_hei = OUT_ROOT / "height" / str(view_slot) / f"JAX_Tile_{TILE_ID}_XYZ_{rgb_no}.tif"
        out_hei.parent.mkdir(parents=True, exist_ok=True)
        hdrv = gdal.GetDriverByName("GTiff")
        hds = hdrv.Create(str(out_hei), w, h, 1, gdalconst.GDT_Float32)
        hds.GetRasterBand(1).WriteArray(np.zeros((h, w), dtype=np.float32))
        hds.GetRasterBand(1).FlushCache()
        hds = None

        print(f"view{view_slot}  <-  {planet_id}  ->  {out_img.name}  (h={h}, w={w})")


if __name__ == "__main__":
    main()
