import os
import argparse
import sys
import types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

import smplx
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class _ChumpyArrayShim:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]

def install_chumpy_pickle_shim():
    if "chumpy.ch" in sys.modules:
        return
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module

def make_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def resolve_script_path(path: str):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def set_axes_equal(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def smpl_to_plot_coords(points: np.ndarray):
    return points[:, [0, 2, 1]]

def shade_face_colors(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)
    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded

def draw_mesh_simple(ax, vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray):
    plot_vertices = smpl_to_plot_coords(vertices)
    plot_joints = smpl_to_plot_coords(joints)
    face_colors = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
    face_colors = shade_face_colors(plot_vertices, faces, face_colors)
    mesh = Poly3DCollection(
        plot_vertices[faces], facecolors=face_colors, linewidths=0.02, edgecolors=(0,0,0,0.05)
    )
    ax.add_collection3d(mesh)
    ax.scatter(plot_joints[:,0], plot_joints[:,1], plot_joints[:,2], c="white", s=10, edgecolors="black")
    set_axes_equal(ax, plot_vertices)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=15, azim=90)
    ax.axis("off")

def save_frame(path, vertices, faces, joints):
    fig = plt.figure(figsize=(5,5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh_simple(ax, vertices, faces, joints)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main(args):
    device = torch.device("cpu")
    dtype = torch.float32
    model_dir = resolve_script_path(args.model_dir)
    frame_dir = os.path.join(SCRIPT_DIR, "anim_frames")
    make_out_dir(frame_dir)

    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_dir, model_type="smpl", gender="neutral", ext="pkl", num_betas=10
    ).to(device)
    faces = np.asarray(model.faces, dtype=np.int32)

    betas = torch.zeros((1,10), device=device, dtype=dtype)
    betas[0,0] = 2.0
    betas[0,1] = -1.2
    betas[0,2] = 0.8

    total_frames = 30
    joint_idx = 18
    max_angle = 1.2

    images = []
    for i in range(total_frames):
        angle = max_angle * (i / (total_frames-1))
        global_orient = torch.zeros((1,3), device=device)
        body_pose = torch.zeros((1,69), device=device)
        pos = (joint_idx -1)*3
        body_pose[0, pos:pos+3] = torch.tensor([0.0, -angle, 0.0], dtype=dtype)

        with torch.no_grad():
            out = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
        v = to_numpy(out.vertices[0])
        j = to_numpy(out.joints[0])
        p = os.path.join(frame_dir, f"frame_{i:02d}.png")
        save_frame(p, v, faces, j)
        images.append(Image.open(p))

    gif_path = os.path.join(SCRIPT_DIR, "animation.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=60, loop=0)
    print(f"✅ 动画生成完成！\n帧图片：{frame_dir}\nGIF：{gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./models")
    args = parser.parse_args()
    main(args)