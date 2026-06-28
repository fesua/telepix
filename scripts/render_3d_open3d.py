"""Render proper 3D views of the merged point cloud using open3d offscreen rendering.
Outputs: top-down, oblique, side-from-south, side-from-east, height-colored variants."""
import sys
import numpy as np
from pathlib import Path

import open3d as o3d
from PIL import Image
from matplotlib import cm

RUN = Path("/home/imlab-p6000/js/SkySplat-main/outputs/run_2026-06-28_12-16-38_ctx_0012-0018-0006_height_ply_v2")
PLY = RUN / "visuals/merged_lotte_pointcloud.ply"
OUT = RUN / "visuals"

# Load PLY
pcd = o3d.io.read_point_cloud(str(PLY))
xyz = np.asarray(pcd.points)
rgb = np.asarray(pcd.colors)  # 0..1
print(f"Loaded {len(xyz):,} points")
print(f"  x: {xyz[:,0].min():.1f} – {xyz[:,0].max():.1f}")
print(f"  y: {xyz[:,1].min():.1f} – {xyz[:,1].max():.1f}")
print(f"  z: {xyz[:,2].min():.1f} – {xyz[:,2].max():.1f}")

# Subsample for faster rendering (still keeps high density)
if len(xyz) > 1_000_000:
    idx = np.random.default_rng(0).choice(len(xyz), size=1_000_000, replace=False)
    xyz = xyz[idx]; rgb = rgb[idx]

# Center the cloud at origin
center = xyz.mean(axis=0)
xyz_c = xyz - center

# Build colored point cloud (RGB version)
pcd_rgb = o3d.geometry.PointCloud()
pcd_rgb.points = o3d.utility.Vector3dVector(xyz_c)
pcd_rgb.colors = o3d.utility.Vector3dVector(rgb)

# Build height-colored point cloud
z = xyz[:, 2]
zn = (z - z.min()) / max(z.max() - z.min(), 1e-6)
hcolors = cm.plasma(zn)[:, :3]
pcd_h = o3d.geometry.PointCloud()
pcd_h.points = o3d.utility.Vector3dVector(xyz_c)
pcd_h.colors = o3d.utility.Vector3dVector(hcolors)


def render(pcd, cam_eye, cam_up, name, w=1600, h=1000):
    """Render an offscreen image of pcd from the given camera."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 1.5
    ctrl = vis.get_view_control()
    # Fit
    vis.poll_events()
    vis.update_renderer()
    # Set camera
    bb = pcd.get_axis_aligned_bounding_box()
    extent = bb.get_extent()
    dist = float(np.linalg.norm(extent)) * 1.2
    cam = np.array(cam_eye, dtype=np.float64) * dist
    front = -cam / np.linalg.norm(cam)
    ctrl.set_front(front.tolist())
    ctrl.set_up(np.array(cam_up, dtype=np.float64).tolist())
    ctrl.set_lookat([0, 0, 0])
    ctrl.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    arr = (np.asarray(img) * 255).astype(np.uint8)
    out_path = OUT / name
    Image.fromarray(arr).save(out_path)
    print(f"  saved {name}")


# Camera viewpoints (front vector: direction camera looks toward origin)
# top-down: looking from +z down
render(pcd_rgb, cam_eye=[0, 0, 1],   cam_up=[0, 1, 0],  name="o3d_topdown_rgb.png")
render(pcd_h,   cam_eye=[0, 0, 1],   cam_up=[0, 1, 0],  name="o3d_topdown_heightcolor.png")

# oblique: 45° from front-right
render(pcd_rgb, cam_eye=[1, -1, 0.7], cam_up=[0, 0, 1],  name="o3d_oblique_rgb.png")
render(pcd_h,   cam_eye=[1, -1, 0.7], cam_up=[0, 0, 1],  name="o3d_oblique_heightcolor.png")

# side from south (look toward +n)
render(pcd_rgb, cam_eye=[0, -1, 0.15], cam_up=[0, 0, 1], name="o3d_side_south_rgb.png")
render(pcd_h,   cam_eye=[0, -1, 0.15], cam_up=[0, 0, 1], name="o3d_side_south_heightcolor.png")

# side from east (look toward -e)
render(pcd_rgb, cam_eye=[-1, 0, 0.15], cam_up=[0, 0, 1], name="o3d_side_east_rgb.png")
render(pcd_h,   cam_eye=[-1, 0, 0.15], cam_up=[0, 0, 1], name="o3d_side_east_heightcolor.png")

print("\nAll 3D views in:", OUT)
