# SkySplat

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub stars](https://img.shields.io/github/stars/<用户名>/<仓库名>?style=social)

Thank you for your attention to and interest in the **SkySplat** series of papers. This repository provides the implementation of SkySplat, a 3D Gaussian Splatting framework for sparse-view satellite image reconstruction.

---

## ✨ Highlights

SkySplat addresses the challenges of multi-temporal sparse-view satellite reconstruction by integrating the RPC camera model into a generalizable 3D Gaussian Splatting pipeline.

<p align="center"> <img src="./fig1_top.png" width="70%" alt="Algorithm Overview"> </p>
RPC-aware generalizable 3D Gaussian Splatting, enabling effective geometric reasoning for satellite imagery
Self-supervised learning with radiometric-robust relative height supervision, without requiring ground-truth DSMs
Efficient reconstruction, achieving up to 86× speedup over per-scene optimization methods such as EOGS

SkySplat consistently outperforms existing generalizable 3DGS baselines, reducing MAE from 13.18 m to 1.80 m on DFC19, and demonstrating strong cross-dataset generalization on MVS3D.

---



---

## 💳 Citation

If your work uses all or part of this code, please cite:
```
@inproceedings{huang2026skysplat,
  title={SkySplat: Generalizable 3D Gaussian splatting from multi-temporal sparse satellite images},
  author={Huang, Xuejun and Liu, Xinyi and Wan, Yi and Zheng, Zhi and Zhang, Bin and Xiong, Mingtao and Pei, Yingying and Zhang, Yongjun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={7},
  pages={5158--5166},
  year={2026}
}
```

You can find our [paper on AAAI2026 and arxiv 📄](https://ojs.aaai.org/index.php/AAAI/article/view/37430).
