# 实验报告：可微渲染

## 一、实验目的

1. **理解可微渲染原理**：掌握可微光栅化的核心思想，特别是处理离散几何体边界时的软光栅化（Soft Rasterization）数学近似方法。
2. **掌握逆渲染流程**：学会如何通过多视角二维图像（剪影）反推并优化三维空间中的网格顶点坐标。
3. **理解正则化的关键作用**：深刻认识在网格优化过程中，拉普拉斯平滑、边长一致性和法线一致性等正则化项对于防止拓扑崩坏和陷入局部最优的决定性作用。
4. **工程实践**：掌握 PyTorch3D 库的使用，完成从球体到奶牛形状的“形状优化”全流程。

---

## 二、实验原理

### 2.1 可微渲染概述

可微渲染是计算机视觉与图形学交叉领域的前沿方向，它通过将二维图像像素的损失反向传播回三维场景的属性（如顶点位置、颜色、纹理等），实现了在没有显式三维监督的情况下学习三维形状。

传统渲染管线中的光栅化步骤是不可微的：像素要么在三角形内部，要么在外部，这种阶跃变化导致边界处的梯度为零（梯度消失），无法指导顶点向正确方向移动。

### 2.2 软光栅化（Soft Rasterization）

软光栅化通过计算像素到三角形边缘的距离，并利用 Sigmoid 函数在边界处产生平滑的概率过渡，解决了传统硬光栅化的梯度消失问题。

**概率过渡公式（文本形式）**：
```
A(d) = sigmoid(d / sigma)
```
其中：
- `d`：像素到三角形边缘的距离
- `sigma`：控制边缘模糊程度的参数

**核心思想**：软光栅化将渲染视为一个可微的聚合过程，融合所有三角形对像素的概率贡献。即使顶点在像素外部，也能提供微小但非零的梯度，引导顶点向正确方向移动。

**PyTorch3D 的实现**：PyTorch3D 的光栅化器返回多个中间变量，包括 `pix_to_face`（每个像素重叠的面索引）、`zbuf`（深度值）、`bary_coords`（重心坐标）和 `pix_dists`（像素到面的距离）。这些中间变量使得整个渲染管线完全可微。

### 2.3 网格正则化（Mesh Regularization）

如果仅依靠图像差异（Loss）去移动顶点，顶点会为了迎合某个视角的投影而疯狂交叉、重叠，最终变成“刺猬”形状，彻底陷入局部最优。因此，必须引入正则化损失来保持网格的光滑和物理合理性。

**三种核心正则化项**：

| 正则化项 | 公式（文本形式） | 作用 |
| :--- | :--- | :--- |
| 拉普拉斯平滑 | `mesh_laplacian_smoothing(mesh)` | 约束相邻顶点，防止表面出现尖锐突起 |
| 边长一致性 | `mesh_edge_loss(mesh)` | 惩罚过长或过短的边，防止三角形严重拉伸 |
| 法线一致性 | `mesh_normal_consistency(mesh)` | 约束相邻三角形面的法线方向接近，保持表面平滑 |

**总损失函数（文本形式）**：
```
L_total = L_silhouette + w_lap * L_lap + w_edge * L_edge + w_normal * L_normal
```

---

## 三、实验任务与实现

### 3.1 任务概览

| 任务 | 描述 | 关键实现 |
| :---: | :--- | :--- |
| 1 | 环境配置 | 安装 PyTorch、PyTorch3D 及依赖 |
| 2 | 加载目标模型并生成参考图 | 加载 `cow.obj`，多视角渲染剪影 |
| 3 | 初始化源模型与渲染管线 | 初始化二十面体球，构建软剪影光栅化器 |
| 4 | 执行可微优化循环 | 梯度下降优化顶点偏移，加入正则化 |
| 5 | 可视化与输出 | 定期保存中间模型，展示优化过程 |

### 3.2 环境配置（ModelScope 平台）

实验采用阿里云魔搭社区（ModelScope）提供的免费 GPU 算力，避免本地环境配置的困难。

**关键安装步骤**：

```python
# 1. 升级包管理工具
!pip install --upgrade pip

# 2. 安装前置依赖项
!pip install fvcore iopath matplotlib ninja

# 3. 源码编译安装 PyTorch3D（从 Gitee 镜像）
!pip install "git+https://gitee.com/hongwenzhang/pytorch3d.git" --no-build-isolation
```

> **说明**：`ninja` 用于多核加速编译，`--no-build-isolation` 可防止某些云平台的严格沙箱隔离导致的编译报错。

### 3.3 加载目标模型并生成参考图

**加载奶牛模型**：

```python
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

verts, faces, _ = load_obj("cow.obj")
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# 归一化：中心对齐并缩放到单位范围
verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
cow_mesh = Meshes(verts=[verts], faces=[faces_idx])
```

**多视角摄像机配置**：

```python
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

num_views = 20
R, T = look_at_view_transform(
    dist=2.7, 
    elev=torch.zeros(num_views), 
    azim=torch.linspace(-180, 180, num_views)
)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
```

**生成目标剪影**：

```python
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer, 
    SoftSilhouetteShader, BlendParams
)

rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=RasterizationSettings(
        image_size=256,
        blur_radius=np.log(1./1e-4 - 1.) * 1e-4,
        faces_per_pixel=50
    )
)
shader = SoftSilhouetteShader(
    blend_params=BlendParams(sigma=1e-4, gamma=1e-4)
)

target_silhouette = shader(
    rasterizer(cow_mesh.extend(num_views)), 
    cow_mesh.extend(num_views)
)[..., 3]  # 提取 Alpha 通道作为剪影
```

### 3.4 初始化源模型（球体）

使用 `ico_sphere` 生成一个细分等级为 4 的二十面体球，作为优化的起点：

```python
from pytorch3d.utils import ico_sphere

src_mesh = ico_sphere(4, device)
deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
```

**关键参数**：
- `ico_sphere(4)`：生成包含 2562 个顶点和 5120 个面的球体网格，细分程度适中，既能表达复杂形状又不过度增加计算量。

### 3.5 构建可微渲染管线

**软剪影渲染器**由光栅化器和着色器两部分组成：

```python
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1./1e-4 - 1.) * 1e-4,
    faces_per_pixel=50
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
)
```

**参数说明**：
- `blur_radius`：控制边缘模糊半径，影响梯度的传播范围。
- `faces_per_pixel=50`：每个像素保留最多 50 个面的信息，确保梯度信息充足。
- `sigma` 和 `gamma`：控制 Sigmoid 函数的陡峭程度，影响软光栅化的平滑度。

### 3.6 优化循环

**核心优化代码**：

```python
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

for i in range(epochs):
    optimizer.zero_grad()
    
    # 形变生成新网格
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # 渲染当前网格的剪影
    pred_silhouette = renderer(new_src_mesh.extend(num_views))[..., 3]
    
    # 计算损失
    loss_silhouette = ((pred_silhouette - target_silhouette) ** 2).mean()
    loss = loss_silhouette + \
           1.0 * mesh_laplacian_smoothing(new_src_mesh) + \
           0.1 * mesh_edge_loss(new_src_mesh) + \
           0.01 * mesh_normal_consistency(new_src_mesh)
    
    loss.backward()
    optimizer.step()
```

**损失函数权重说明**：

| 损失项 | 权重 | 作用 |
| :--- | :---: | :--- |
| 剪影损失 | 1.0 | 驱动顶点向目标形状移动 |
| 拉普拉斯平滑 | 1.0 | 保持表面光滑，防止尖刺 |
| 边长损失 | 0.1 | 防止三角形过度拉伸或压缩 |
| 法线一致性 | 0.01 | 保持相邻面法线方向一致 |

### 3.7 可视化与模型保存

```python
if i % 20 == 0:
    # 保存当前网格为 .obj 文件
    current_verts = new_src_mesh.verts_list()[0]
    current_faces = new_src_mesh.faces_list()[0]
    save_obj(f"output_meshes/mesh_epoch_{i:03d}.obj", current_verts, current_faces)
    
    # 可视化对比
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(target_silhouette[0].cpu().numpy(), cmap='gray')
    ax[0].set_title("Ground Truth")
    ax[1].imshow(pred_silhouette[0].detach().cpu().numpy(), cmap='gray')
    ax[1].set_title(f"Epoch {i}")
    plt.show()
```

---

## 四、结果分析与讨论

### 4.1 优化过程

| 迭代次数 | 总 Loss | 剪影 Loss | 网格状态 |
| :---: | :---: | :---: | :--- |
| 0 | 较高 | 较高 | 初始球体 |
| 20 | 下降明显 | 下降明显 | 开始出现凹凸 |
| 60 | 继续下降 | 继续下降 | 初步呈现奶牛轮廓 |
| 100 | 较低 | 较低 | 形状基本成型 |
| 200+ | 收敛 | 收敛 | 接近目标形状 |

### 4.2 正则化的效果对比

| 正则化配置 | 结果 |
| :--- | :--- |
| 仅剪影损失（无正则化） | 网格出现大量尖刺和自交，形状崩坏 |
| 加入拉普拉斯平滑 | 表面光滑，但细节可能被过度平滑 |
| 加入边长损失 | 三角形分布均匀，无过度拉伸 |
| 加入法线一致性 | 表面法线过渡自然，无突兀褶皱 |
| **三者结合** | 形状准确且表面光滑、拓扑良好 |

### 4.3 软光栅化 vs 硬光栅化

| 对比维度 | 硬光栅化 | 软光栅化 |
| :--- | :--- | :--- |
| 梯度可用性 | 边界处梯度为零 | 边界处提供平滑梯度 |
| 优化稳定性 | 容易陷入局部最优 | 梯度引导稳定收敛 |
| 边缘表现 | 锐利、二值化 | 模糊、概率化 |
| 适用场景 | 传统渲染 | 可微渲染/逆渲染 |

### 4.4 视角数量的影响

- **视角过少（如 4-6 个）**：形状在未观测角度出现偏差，容易产生不对称。
- **视角适中（如 20 个）**：形状各向同性收敛，效果最佳。
- **视角过多（如 50+ 个）**：计算量增大，收益递减。

---

## 五、关键参数调优指南

| 参数 | 推荐值 | 调优建议 |
| :--- | :---: | :--- |
| 学习率 | 1.0 | 过大导致震荡，过小收敛慢 |
| 优化器 | SGD + momentum 0.9 | Adam 也可，但 SGD 更稳定 |
| 网格细分等级 | 4 | 等级越高细节越丰富，计算量越大 |
| 图像分辨率 | 256 | 分辨率影响精度和计算量 |
| 剪影损失权重 | 1.0 | 主导优化方向 |
| 拉普拉斯权重 | 0.5-1.0 | 过大过度平滑，过小产生尖刺 |
| 边长损失权重 | 0.05-0.2 | 过大限制形变能力 |
| 法线损失权重 | 0.005-0.02 | 辅助光滑，不宜过大 |

---

## 六、常见问题与解决方案

| 问题现象 | 可能原因 | 解决方法 |
| :--- | :--- | :--- |
| 模型不收敛 | 学习率过大或过小 | 调整学习率，尝试 0.1-2.0 |
| 网格出现尖刺 | 正则化权重不足 | 增大拉普拉斯平滑权重 |
| 三角形严重变形 | 边长损失权重不足 | 增大 `mesh_edge_loss` 权重 |
| 形状细节丢失 | 正则化权重过大 | 适当降低正则化权重 |
| 剪影模糊 | `sigma` 过大 | 减小 `sigma` 值 |
| 梯度消失 | `blur_radius` 过小 | 增大 `blur_radius` |
| 内存不足 | 网格太密或 batch 太大 | 降低细分等级或减少视角数 |
| 编译失败 | PyTorch3D 版本不兼容 | 检查 PyTorch 版本，使用匹配的 PyTorch3D |

---

## 七、总结

本次实验完整实现了基于可微渲染的形状优化流程，从二十面体球出发，通过多视角剪影监督和正则化约束，成功将球体“捏”成奶牛形状。核心收获包括：

1. **软光栅化的必要性**：传统硬光栅化在边界处梯度消失，无法驱动顶点优化；软光栅化通过概率化渲染提供了平滑的梯度信号，使得逆渲染成为可能。

2. **正则化的决定性作用**：纯剪影损失会导致网格拓扑崩坏，必须配合拉普拉斯平滑、边长一致性和法线一致性三种正则化项，才能在保持形状准确的同时维持网格质量。

3. **PyTorch3D 的模块化设计**：光栅化与着色解耦的设计使得渲染管线高度可定制，支持异构批处理和梯度回传。

4. **多视角监督的优势**：通过 20 个均匀分布的视角提供剪影监督，实现了三维形状的各向同性收敛，避免了单视角优化带来的遮挡和不对称问题。

本实验为后续更复杂的可微渲染应用（如纹理优化、神经渲染、三维重建等）奠定了坚实的基础。

---

## 八、选做内容思路（联合纹理优化）

在完成形状优化的基础上，可进一步扩展至纹理优化：

1. **使用 `SoftPhongShader`**：替换剪影着色器为 Phong 着色器，支持 RGB 颜色渲染。
2. **定义可优化纹理**：将顶点颜色或纹理图设为可微参数（`requires_grad=True`）。
3. **多任务损失**：同时优化剪影损失和 RGB 图像损失。
4. **参考实现**：PyTorch3D 官方 `fit_textured_mesh.ipynb` 教程提供了完整的实现参考。

---

## 九、参考资料

- PyTorch3D 官方文档：渲染器架构
- PyTorch3D 技术报告
- Soft Rasterizer (SoftRas) 论文
- PyTorch3D 网格变形教程
- PyTorch3D 纹理网格拟合教程
