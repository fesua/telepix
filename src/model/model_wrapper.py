from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import torch
from einops import  rearrange
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn, optim
import numpy as np
import json
from ..dataset.data_module import get_data_shim
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, calculate_mae, calculate_rmse, calculate_less_thre_ratio
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import save_image
from ..misc.step_tracker import StepTracker
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass

def to_json_serializable(value):
    if isinstance(value, torch.Tensor):
        return to_json_serializable(value.detach().cpu())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_serializable(v) for v in value]
    return value
    
class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            self.val_step_outputs = {}
        self.val_step_cnt = 0

    def training_step(self, batch, batch_idx):
        _, _, _, h, w = batch["context"]["image"].shape
        filename = batch["context"]['ref_filename'][0][0].split("/0/")[-1]
        print("Filename: ", filename)
        # Run the model.
        gaussians = self.encoder(batch["context"], self.global_step, False)
        output = self.decoder.forward(gaussians, batch["context"], (h, w), mode = "train")
        target_gt = batch["context"]["image"]
        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output["render_color"], "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        loss_items = {}
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            loss_items[loss_fn.name] = loss.item()
            total_loss = total_loss + loss

        self.log("loss/total", total_loss)
        # Logging to console
        if (
                self.global_rank == 0
                and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            detail = " | ".join([f"{n}: {v:.6f}" for n, v in loss_items.items()])
            print(f"train step {self.global_step}; total: {total_loss:.6f} | {detail}")

        self.log("info/global_step", self.global_step)

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        return total_loss

    def test_step(self, batch, batch_idx):
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(batch["context"], self.global_step, False)
        with self.benchmarker.time("decoder", num_calls=v):
            # output = self.decoder.forward(gaussians, batch["context"], (h, w), mode="test")
            output = self.decoder.forward(gaussians, batch["target"], (h, w), mode="test")

        scene = batch["context"]['ref_filename'][0][0].split("/0/")[-1].split(".tif")[0].split("_RGB_")[0]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        rgb_gt = batch["target"]["image"]
        rgb_gt = rearrange(rgb_gt, "b v c h w -> (b v) c h w")
        render_rgb = rearrange(output["render_color"], "b v c h w -> (b v) c h w")
        height_gt = batch["context"]['gt_height']
        mask_height = ~torch.isnan(height_gt)
        pred_height = gaussians.hei.reshape(b, v, h, w)

        # Save images.
        if self.test_cfg.save_image:
            # Name each rendered target view by its target filename so we can save all v_target
            # renders (not just min(v_context, v_target) as the old zip-based loop did).
            target_fns = batch["target"]['ref_filename'] if 'ref_filename' in batch["target"] else None
            for t_idx in range(render_rgb.shape[0]):
                if target_fns is not None and t_idx < len(target_fns):
                    tname = target_fns[t_idx][0] if isinstance(target_fns[t_idx], (list, tuple)) else target_fns[t_idx]
                    stem = Path(str(tname)).stem  # e.g. JAX_Tile_999_RGB_004_crop_16
                else:
                    stem = f"target{t_idx}"
                save_image(render_rgb[t_idx], path / scene / f"color/tgt_view_{t_idx}_{stem}.png")

            # Save per-context-view height maps as colormapped PNGs (predicted absolute altitude in meters).
            from ..visualization.vis_depth import viz_depth_tensor
            crop_stem = batch["context"]['ref_filename'][0][0].split("/0/")[-1].split(".tif")[0]
            crop_stem_safe = Path(crop_stem).name  # JAX_Tile_999_RGB_001_crop_<id>
            for v_idx in range(pred_height.shape[1]):
                h_map = pred_height[0, v_idx].detach().cpu()  # (h, w) altitude in m
                hmin = float(h_map.min().item())
                hmax = float(h_map.max().item())
                # plasma colormap on the height range (clamp to a sensible Seoul window)
                disp = (h_map - hmin) / max(hmax - hmin, 1e-6)
                colored = viz_depth_tensor(disp, return_numpy=True)  # (h, w, 3) uint8
                from PIL import Image
                out = path / scene / f"height/ctx_view_{v_idx}_{crop_stem_safe.replace('_001_', f'_00{v_idx+1}_')}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(colored).save(out)
                # Also save the raw height grid as .npy (per-pixel meters) for downstream analysis
                np.save(str(out).replace(".png", ".npy"), h_map.numpy())
                # Per-crop height-min/max sidecar so a stitcher can use a consistent colormap
                with open(str(out).replace(".png", ".json"), "w") as fp:
                    json.dump({"min_height_m": hmin, "max_height_m": hmax}, fp)

            # Export raw Gaussian tensors as .npz (debugging) + simple XYZ+RGB PLY (point cloud).
            self._save_points_ply(
                gaussians,
                out_path=path / scene / f"gaussians/{crop_stem_safe}.ply",
            )
            self._save_gaussians_npz(
                gaussians,
                out_path=path / scene / f"gaussians/{crop_stem_safe}.npz",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = render_rgb

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
            if f"mae" not in self.test_step_outputs:
                self.test_step_outputs[f"mae"] = []
            if f"rmse" not in self.test_step_outputs:
                self.test_step_outputs[f"rmse"] = []
            if f"pag10" not in self.test_step_outputs:
                self.test_step_outputs[f"pag10"] = []
            if f"pag25" not in self.test_step_outputs:
                self.test_step_outputs[f"pag25"] = []
            if f"pag75" not in self.test_step_outputs:
                self.test_step_outputs[f"pag75"] = []
            self.test_step_outputs[f"psnr"].append(compute_psnr(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"ssim"].append(compute_ssim(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"lpips"].append(compute_lpips(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"mae"].append(calculate_mae(height_gt, pred_height, mask_height))
            self.test_step_outputs[f"rmse"].append(calculate_rmse(height_gt, pred_height, mask_height))
            self.test_step_outputs[f"pag10"].append(calculate_less_thre_ratio(height_gt, pred_height, mask_height, 1.0))
            self.test_step_outputs[f"pag25"].append(calculate_less_thre_ratio(height_gt, pred_height, mask_height, 2.5))
            self.test_step_outputs[f"pag75"].append(calculate_less_thre_ratio(height_gt, pred_height, mask_height, 7.5))

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                metric_scores = to_json_serializable(metric_scores)
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                avg_time = float(np.mean(times))
                saved_scores[tag] = [len(times), avg_time]
                print(f"{tag}: {len(times)} calls, avg. {avg_time} seconds per call")
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(to_json_serializable(saved_scores), f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    def _save_points_ply(self, gaussians, out_path):
        """Export Gaussian means + DC-color as a simple XYZRGB binary PLY.
        Skips low-opacity Gaussians so the cloud isn't dominated by transparent splats.
        Viewable in MeshLab, CloudCompare, Blender, ParaView, etc."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Reshape everything to (N, ...) regardless of leading dims.
        means = gaussians.means.detach().cpu().float().numpy().reshape(-1, 3)
        opac = gaussians.opacities.detach().cpu().float().numpy().reshape(-1)
        # harmonics has shape (..., 3, d_sh); pull DC band → (N, 3)
        sh = gaussians.harmonics.detach().cpu().float().numpy()
        sh = sh.reshape(-1, sh.shape[-2], sh.shape[-1])  # (N_chunks_x_per_view, 3, d_sh)
        dc = sh[..., 0]                                  # (N_chunks_x_per_view, 3)
        # If N (means) != N (sh), broadcast/tile to align — they should match in practice.
        if dc.shape[0] != means.shape[0]:
            # tile DC if it's one DC per source pixel and means has more entries (e.g. per gaussian-per-pixel)
            n_per = means.shape[0] // dc.shape[0]
            if n_per * dc.shape[0] == means.shape[0] and n_per >= 1:
                dc = np.repeat(dc, n_per, axis=0)
            else:
                # fallback: gray
                dc = np.full((means.shape[0], 3), 0.5, dtype=np.float32)

        # 3DGS DC band → linear RGB via sigmoid-like; spherical harmonics DC: c = 0.5 + sh_0 * 0.28209
        SH_DC = 0.28209479177387814
        rgb_lin = np.clip(0.5 + dc * SH_DC, 0.0, 1.0)
        rgb_u8 = (rgb_lin * 255).astype(np.uint8)

        # Filter low-opacity gaussians
        mask = opac > 0.01
        if mask.sum() == 0:
            mask = np.ones_like(mask, dtype=bool)
        means = means[mask]
        rgb_u8 = rgb_u8[mask]
        N = means.shape[0]

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        # Pack as a structured array
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.zeros(N, dtype=dtype)
        rec["x"] = means[:, 0]; rec["y"] = means[:, 1]; rec["z"] = means[:, 2]
        rec["r"] = rgb_u8[:, 0]; rec["g"] = rgb_u8[:, 1]; rec["b"] = rgb_u8[:, 2]
        with open(out_path, "wb") as fp:
            fp.write(header)
            fp.write(rec.tobytes())

    def _save_gaussians_npz(self, gaussians, out_path):
        """Dump every Gaussian tensor verbatim so a downstream script can construct
        the 3DGS-format PLY without re-running inference."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_np = lambda t: t.detach().cpu().float().numpy()
        np.savez(out_path,
                 means=to_np(gaussians.means),
                 scales=to_np(gaussians.scales),
                 rotations=to_np(gaussians.rotations),
                 opacities=to_np(gaussians.opacities),
                 harmonics=to_np(gaussians.harmonics),
                 hei=to_np(gaussians.hei))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
