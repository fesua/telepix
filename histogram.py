import numpy as np
import tifffile
from skimage.exposure import match_histograms
from PIL import Image
import os

ref_img = tifffile.imread("/home/imlab/js/Dataset/20200308_021307_ssc2d2_0010_basic_analytic.tif")[:, :, :3].astype(np.float32)
# 2. 색감을 변경할 대상(Source)인 2021년 붉은 이미지를 읽어옵니다.
# src_img = tifffile.imread("/home/imlab/js/Dataset/20210209_050530_ssc10d1_0012_basic_analytic.tif")[:, :, :3].astype(np.float32)
src_img = tifffile.imread("/home/imlab/js/Dataset/20220512_234256_ss02d3_0018_basic_analytic.tif")[:, :, :3].astype(np.float32)


matched_img = match_histograms(src_img, ref_img, channel_axis=-1)

# 4. 이후 0~255 스케일링 후 PNG로 저장
matched_png = ((matched_img - matched_img.min()) / (matched_img.max() - matched_img.min()) * 255).astype(np.uint8)
Image.fromarray(matched_png).save("./stella/2022_matched.png")
