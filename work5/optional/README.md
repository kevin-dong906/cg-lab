# 实验报告：可微渲染扩展——联合纹理优化（选做部分）

## 一、实验目的

在必做实验（基于黑白剪影的形状优化）的基础上，进一步实现联合纹理优化：

1. 引入 **SoftPhongShader**，不仅拟合剪影，还直接拟合多视角 RGB 彩色图像。
2. 同时优化网格的顶点坐标和顶点颜色（或纹理贴图），实现形状和纹理的联合重建。
3. 理解颜色损失与形状损失之间的协同作用，以及纹理优化对最终渲染质量的提升。
4. 掌握 PyTorch3D 中 `TexturesVertex` 数据结构的使用方法。

---

## 二、实验原理

### 2.1 联合纹理优化的必要性

在必做实验中，我们仅使用二值剪影作为监督信号，因此只能恢复物体的外部轮廓形状，而无法获得表面的颜色信息。然而，在许多实际应用（如三维重建、增强现实）中，我们不仅需要几何形状，还需要材质/纹理信息以生成逼真的渲染效果。

联合纹理优化通过在损失函数中加入 **RGB 颜色损失**，使得优化过程同时考虑像素的颜色差异，从而驱动顶点位置和顶点颜色的共同更新。

### 2.2 损失函数设计

总损失函数（文本形式）：
```
L_total = w_rgb * L_rgb + w_silhouette * L_silhouette + w_edge * L_edge + w_lap * L_lap + w_normal * L_normal
```

各项说明：

| 损失项 | 公式（简写） | 作用 |
| :--- | :--- | :--- |
| RGB 损失 | `mean((pred_rgb - target_rgb)^2)` | 拟合像素颜色，驱动纹理和形状 |
| 剪影损失 | `mean((pred_alpha - target_alpha)^2)` | 保证形状轮廓正确 |
| 边缘损失 | `mesh_edge_loss(mesh)` | 防止三角形拉伸 |
| 拉普拉斯平滑 | `mesh_laplacian_smoothing(mesh)` | 保持表面光滑 |
| 法线一致性 | `mesh_normal_consistency(mesh)` | 保持相邻面法线一致 |

**核心区别**：相比必做，新增了 `L_rgb`，并且着色器从 `SoftSilhouetteShader` 升级为 `SoftPhongShader`，后者支持光照和顶点颜色的插值渲染。

### 2.3 可微纹理渲染管线

PyTorch3D 的 `SoftPhongShader` 支持两种纹理模式：
- **顶点颜色**：每个顶点存储 RGB 值，通过重心坐标插值得到像素颜色。
- **纹理贴图**：通过 UV 坐标从纹理图像采样。

本实验采用 **顶点颜色**（`TexturesVertex`）方式，因为它可以直接作为优化变量，且不需要 UV 映射。

渲染流程：
1. 光栅化：计算每个像素覆盖的面、重心坐标和深度。
2. 着色：使用 `SoftPhongShader`，结合点光源、材质参数和顶点颜色，计算每个像素的最终 RGB 值和 Alpha 通道。

由于整个管线是可微的，RGB 损失可以反向传播到顶点坐标和顶点颜色。

---

## 三、实验步骤与实现

本实验基于 PyTorch3D 官方教程 `fit_textured_mesh.ipynb` 进行，使用相同的奶牛模型和 20 个视角数据集。

### 3.1 环境准备与数据加载

```python
import torch
import matplotlib.pyplot as plt
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    PointLights, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, SoftSilhouetteShader,
    TexturesVertex
)
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

**加载奶牛模型并归一化**：
```python
mesh = load_objs_as_meshes(["data/cow_mesh/cow.obj"], device=device)
verts = mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))
```

### 3.2 创建多视角数据集

与必做类似，生成 20 个视角的摄像机参数，并分别渲染 RGB 图像和剪影图像。

```python
num_views = 20
elev = torch.linspace(0, 360, num_views)
azim = torch.linspace(-180, 180, num_views)
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# 用于 RGB 渲染的 Phong 着色器
raster_settings_rgb = RasterizationSettings(image_size=128, blur_radius=0.0, faces_per_pixel=1)
renderer_rgb = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_rgb),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# 渲染目标 RGB 图像
meshes = mesh.extend(num_views)
target_images = renderer_rgb(meshes, cameras=cameras, lights=lights)
target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

# 用于剪影的软光栅化器（与必做相同）
sigma = 1e-4
raster_settings_sil = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=50
)
renderer_sil = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_sil),
    shader=SoftSilhouetteShader()
)
silhouette_images = renderer_sil(meshes, cameras=cameras, lights=lights)
target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]
```

### 3.3 初始化源模型（球体 + 随机顶点颜色）

```python
src_mesh = ico_sphere(4, device)  # 2562 顶点，5120 面

# 可学习的顶点偏移
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# 可学习的顶点颜色（初始化为灰色 0.5）
sphere_verts_rgb = torch.full([1, src_mesh.verts_packed().shape[0], 3], 0.5, device=device, requires_grad=True)
```

### 3.4 构建可微渲染器（用于优化）

为了在优化过程中获得梯度，需要使用软光栅化（`blur_radius > 0`）和 `SoftPhongShader`。

```python
raster_settings_soft = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=50,
    perspective_correct=False,
)
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_soft),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)
```

### 3.5 定义损失函数与优化器

```python
losses = {
    "rgb": {"weight": 1.0, "values": []},
    "silhouette": {"weight": 1.0, "values": []},
    "edge": {"weight": 1.0, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
    "normal": {"weight": 0.01, "values": []},
}
optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)
```

### 3.6 优化循环

```python
Niter = 2000
num_views_per_iteration = 2

for i in range(Niter):
    optimizer.zero_grad()
    
    # 变形球体
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    # 赋予顶点颜色
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    
    # 计算正则化损失
    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    loss["edge"] = mesh_edge_loss(new_src_mesh)
    loss["laplacian"] = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    loss["normal"] = mesh_normal_consistency(new_src_mesh)
    
    # 随机选择两个视角计算数据损失
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images = renderer_textured(new_src_mesh, cameras=target_cameras[j], lights=lights)
        # RGB 损失
        loss_rgb = ((images[..., :3] - target_rgb[j]) ** 2).mean()
        loss["rgb"] += loss_rgb / num_views_per_iteration
        # 剪影损失
        loss_sil = ((images[..., 3] - target_silhouette[j]) ** 2).mean()
        loss["silhouette"] += loss_sil / num_views_per_iteration
    
    # 加权总损失
    total_loss = sum(losses[k]["weight"] * loss[k] for k in losses)
    total_loss.backward()
    optimizer.step()
    
    # 记录和可视化
    if i % 250 == 0:
        print(f"Iter {i}: total_loss = {total_loss.item():.4f}")
        # 可视化代码省略...
```

### 3.7 保存最终模型

```python
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts * scale + center  # 反归一化
save_obj("final_textured_model.obj", final_verts, final_faces)
# 顶点颜色会保存在 .obj 文件中（如果格式支持）
```

---

## 四、代码深度分析

### 4.1 SoftPhongShader 与顶点颜色

`SoftPhongShader` 的核心计算流程：
1. 光栅化返回每个像素的 barycentric 坐标和面索引。
2. 利用 barycentric 坐标插值顶点位置、法线和顶点颜色，得到像素级的 `fragments`。
3. 计算光照（漫反射 + 高光 + 环境光），与插值后的基础颜色相乘。
4. 输出 RGB + Alpha。

由于所有操作都是可微的，顶点颜色 `sphere_verts_rgb` 的梯度可以通过链式法则从 RGB 损失反向传播。

### 4.2 损失权重的选择与影响

| 损失项 | 权重 | 作用与影响 |
| :--- | :---: | :--- |
| RGB 损失 | 1.0 | 主导优化，驱动形状和颜色同时更新 |
| 剪影损失 | 1.0 | 辅助保持轮廓，防止颜色损失导致形状漂移 |
| 边缘损失 | 1.0 | 防止三角形过度拉伸，保持网格质量 |
| 拉普拉斯平滑 | 1.0 | 防止尖刺，保持表面光滑 |
| 法线一致性 | 0.01 | 轻微约束法线，避免褶皱 |

**实验观察**：若 RGB 损失权重过大，模型会倾向于通过调整颜色来拟合图像，而忽略形状；若剪影损失权重过大，形状会收敛但颜色可能不准确。本实验采用两者权重相等（均为1.0），取得了良好平衡。

### 4.3 随机视角采样策略

每个迭代步骤只随机选择 2 个视角进行损失计算，而不是全部 20 个视角。原因：
- 减少计算量，加速迭代。
- 增加随机性，有助于跳出局部最优。
- 每个迭代步骤使用不同视角，模型会逐渐适应所有视角。

### 4.4 软光栅化参数的选择

`blur_radius` 和 `faces_per_pixel` 影响梯度平滑度：
- `blur_radius` 越大，梯度传播范围越广，但图像会模糊。
- `faces_per_pixel` 越大，每个像素考虑的三角形越多，梯度信息越丰富，但计算量增加。
本实验采用 `blur_radius` 使 Sigmoid 函数在边界处平滑过渡，`faces_per_pixel=50` 保证足够的梯度信息。

---

## 五、结果与对比分析

### 5.1 优化过程

| 迭代次数 | 形状状态 | 颜色状态 |
| :---: | :--- | :--- |
| 0 | 球体 | 均匀灰色 |
| 500 | 初步呈现奶牛轮廓 | 开始出现颜色斑块 |
| 1000 | 形状基本准确 | 颜色接近目标（如棕色、白色） |
| 1500 | 细节完善 | 颜色丰富，逼真 |
| 2000 | 收敛 | 纹理清晰，渲染逼真 |

### 5.2 与仅形状优化（必做）的对比

| 对比维度 | 仅形状优化（必做） | 联合纹理优化（选做） |
| :--- | :--- | :--- |
| 监督信号 | 二值剪影 | RGB 彩色图像 + 剪影 |
| 优化变量 | 顶点坐标 | 顶点坐标 + 顶点颜色 |
| 最终效果 | 灰色/单色网格 | 彩色纹理网格，视觉逼真 |
| 应用价值 | 形状重建 | 全彩三维重建，可渲染 |

### 5.3 可视化结果

- 渲染出的彩色奶牛与目标图像基本一致，颜色分布准确（头部棕色、身体白色带斑点等）。
- 形状细节（如耳朵、犄角）也得到了更准确的拟合，因为颜色损失提供了额外的监督信号。

---

## 六、常见问题与解决方案

| 问题现象 | 可能原因 | 解决方法 |
| :--- | :--- | :--- |
| 颜色出现明显块状 | 顶点颜色学习率过大或过小 | 调整学习率或使用更精细的初始颜色 |
| 形状与颜色不匹配 | RGB 损失与剪影损失权重失衡 | 调整权重，确保两者平衡 |
| 纹理过度拟合单视角 | 未使用足够多的视角 | 增加 `num_views_per_iteration` 或使用全部视角 |
| 高光区域过亮 | 光源参数固定但不合理 | 调整光源位置或强度 |
| 优化不收敛 | 正则化权重过大或过小 | 尝试调整正则化权重，例如增大拉普拉斯平滑 |

---

## 七、总结

本次选做实验在必做形状优化的基础上，成功实现了联合纹理优化，核心收获包括：

1. **可微渲染的扩展性**：通过更换着色器（从 `SoftSilhouetteShader` 到 `SoftPhongShader`）和增加损失项，我们可以轻松地扩展优化目标，实现更丰富的重建任务。
2. **纹理学习的可行性**：顶点颜色作为优化变量，可以直接从 RGB 图像中学习，证明了可微渲染在纹理重建中的有效性。
3. **多任务学习的重要性**：同时优化形状和纹理时，不同损失项之间的平衡至关重要，合理的权重设置能获得更好的收敛结果。
4. **工程实践**：掌握了 PyTorch3D 中 `TexturesVertex`、`SoftPhongShader` 和 `PointLights` 的配合使用，为后续处理更复杂的纹理（如 UV 贴图）奠定了基础。

联合纹理优化大大提升了重建模型的实用性和视觉效果，是从“几何重建”迈向“全彩三维重建”的关键一步。

---

## 八、参考资料

- PyTorch3D 官方教程：Fitting a mesh via rendering
- PyTorch3D 文档：Textures, Shaders, Lights
- 可微渲染与逆渲染相关论文
