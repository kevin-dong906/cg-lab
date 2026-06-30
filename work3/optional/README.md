# 实验报告：Phong光照模型扩展——Blinn-Phong与硬阴影

## 一、实验目的

在基础 Phong 光照模型实验之上，进一步实现两种增强技术：

1. **Blinn-Phong 模型**：用半程向量替代反射向量计算镜面高光，理解其对高光形状和边缘表现的改善。
2. **硬阴影**：通过阴影射线检测遮挡关系，实现物体间的简单阴影效果，增强场景真实感。

通过对比分析，深入理解不同光照模型在视觉上的差异以及阴影算法的基本原理。

---

## 二、实验原理

### 2.1 Blinn-Phong 模型

Phong 模型使用反射向量 R 和视线向量 V 的点积来计算镜面分量，但在大入射角（光源与法线夹角较大）时，R 与 V 的夹角容易急剧增大，导致高光边缘出现不自然的“撕裂”或过度衰减。

Blinn-Phong 模型引入**半程向量 H**（即光线方向 L 和视线方向 V 的中间方向），用 H 与法线 N 的点积来代替 R·V：

```
H = normalize(L + V)
specular = Ks * max(0, N·H)^n * C_light
```

当视线方向与反射方向完全重合时，H 与 N 重合；当偏离时，N·H 的衰减比 R·V 更平滑，且物理上更接近微表面理论。

### 2.2 硬阴影（Shadow Ray）

硬阴影的基本思想：对于每个着色点，从该点向光源方向发射一条检测射线，检查射线在到达光源前是否与其他物体相交。若相交，则该点处于阴影中，仅显示环境光分量（或者环境光加漫反射部分，为简单起见可只保留环境光）。

算法流程：
1. 在像素着色阶段，确定最近的交点 P 和法向量 N。
2. 计算从 P 指向光源的方向向量 L。
3. 从 P 出发沿 L 方向发射阴影射线，检测是否与场景中除自身以外的物体相交。
4. 若存在交点且距离小于 P 到光源的距离，则该点在阴影内，最终颜色 = 环境光分量（或环境光+一个微弱的漫反射）。
5. 否则正常计算 Phong 颜色。

---

## 三、实现步骤

### 3.1 环境准备

本次选做基于必做的 Phong 光照模型代码进行扩展，场景不变（红球+紫圆锥），光源位置和材质参数 UI 保留。

### 3.2 Blinn-Phong 实现

在 `render` 内核中，替换镜面高光计算部分：

原代码（Phong）：
```
R = normalize(reflect(-L, N))
spec = ti.max(0.0, R.dot(V)) ** shininess[None]
specular = Ks[None] * spec * light_color
```

改为 Blinn-Phong：
```
H = normalize(L + V)
spec = ti.max(0.0, N.dot(H)) ** shininess[None]
specular = Ks[None] * spec * light_color
```

其他部分（环境光、漫反射）保持不变。

### 3.3 硬阴影实现

需要新增一个求交函数，用于检测从点 P 沿方向 L 是否被遮挡。可使用已有的 `intersect_sphere` 和 `intersect_cone`，只需判断是否存在正的交点且交点距离小于光源距离。

在着色阶段，增加阴影检测代码（伪代码）：
```
in_shadow = False
dist_to_light = (light_pos - P).norm()
shadow_eps = 1e-4   # 避免与自身相交

# 检测球体遮挡
t_sph, _ = intersect_sphere(P, L, sphere_center, sphere_radius)
if t_sph > shadow_eps and t_sph < dist_to_light:
    in_shadow = True

# 检测圆锥遮挡
t_cone, _ = intersect_cone(P, L, cone_apex, cone_base_y, cone_radius)
if t_cone > shadow_eps and t_cone < dist_to_light:
    in_shadow = True

if in_shadow:
    color = ambient   # 只保留环境光
else:
    color = ambient + diffuse + specular  # 完整光照
```

注意：阴影射线起点应沿法向量方向微微偏移（加小量 epsilon），避免与自身表面相交产生自阴影。

---

## 四、代码关键修改（完整函数片段）

以下展示在 `render` 内核中整合两项扩展的简化代码（仅显示改动部分）：

```python
@ti.kernel
def render():
    for i, j in pixels:
        # 光线生成、求交、获取 min_t, hit_normal, hit_color 与必做相同 ...
        # （省略前期求交代码，得到最近的交点 P、法线 N、物体颜色 object_color）

        if min_t < 1e9:
            P = ro + rd * min_t
            N = hit_normal
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - P)
            V = normalize(ro - P)

            # 计算环境光与漫反射（与 Phong 相同）
            ambient = Ka[None] * light_color * hit_color
            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color

            # --- Blinn-Phong 镜面高光 ---
            H = normalize(L + V)
            spec = ti.max(0.0, N.dot(H)) ** shininess[None]
            specular = Ks[None] * spec * light_color

            # --- 硬阴影检测 ---
            in_shadow = False
            dist_to_light = (light_pos - P).norm()
            eps = 1e-4
            # 从 P 沿 L 方向偏移一点，避免自交
            shadow_origin = P + L * eps

            # 检测球体
            t_sph, _ = intersect_sphere(shadow_origin, L, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
            if 0 < t_sph < dist_to_light:
                in_shadow = True

            # 检测圆锥（注意圆锥可能被自身遮挡，需排除自身交点，但阴影起点已偏移，一般不会）
            t_cone, _ = intersect_cone(shadow_origin, L, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
            if 0 < t_cone < dist_to_light:
                in_shadow = True

            # 最终颜色
            if in_shadow:
                color = ambient   # 仅环境光
            else:
                color = ambient + diffuse + specular

            pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)
        else:
            pixels[i, j] = background_color
```

---

## 五、结果对比与分析

### 5.1 Phong vs Blinn-Phong 高光表现

| 特征 | Phong 模型 | Blinn-Phong 模型 |
| :--- | :--- | :--- |
| 高光形状 | 沿反射方向聚焦，圆形高光 | 更自然的椭圆形高光，与观察角度关系更平滑 |
| 大入射角表现 | 高光边缘可能突然消失（因 R·V 快速衰减） | 高光边缘过渡更柔和，更符合真实物理 |
| 计算复杂度 | 需要计算反射向量 R | 只需计算 H = normalize(L+V)，略简单 |
| 适用场景 | 老式实时渲染 | 现代渲染（更接近微表面模型） |

**直观观测**：在调整光源位置或观察角度时，Blinn-Phong 的高光区域在物体边缘移动更平滑，不会出现 Phong 模型在特定角度下高光“瞬移”或“爆裂”的现象。

### 5.2 硬阴影效果

- 实现了物体之间的简单遮挡关系：当红球位于光源和紫圆锥之间时，圆锥部分区域会出现阴影。
- 阴影区域颜色仅保留环境光，与周围明亮区形成鲜明对比，增强了立体感和层次感。
- 由于未做阴影柔化，阴影边缘锐利（硬阴影），适合简单场景，但真实感略逊于软阴影。

### 5.3 综合效果

同时启用两项扩展后，场景既具有更自然的镜面高光，又具有基本的阴影关系，视觉效果显著提升。

---

## 六、性能与实现注意事项

- **阴影射线数量**：每个像素的着色点都发射一条额外的阴影射线，求交次数翻倍（原来两个物体，现在每个像素最多做 4 次求交）。对于 480k 像素，仍可在 GPU 上实时运行，性能下降不明显。
- **自阴影问题**：阴影射线起点必须沿法线方向偏移一个小量，否则可能与自己所在表面相交，形成错误的阴影。
- **数值稳定性**：使用 `eps = 1e-4` 避免浮点误差导致的错误遮挡。
- **同一物体阴影**：本例中，球体的阴影射线可能与圆锥相交，反之亦然，从而实现互阴影。

---

## 七、总结

通过实现 Blinn-Phong 模型和硬阴影，我获得了以下经验和认识：

1. **光照模型的选择**：Blinn-Phong 在大多数情况下优于原始 Phong，高光更稳定、更自然，且计算略微简化，是现代实时渲染的常用选择。
2. **阴影的重要性**：阴影是提升场景真实感的关键因素，即便简单的硬阴影也能极大改善空间层次感。
3. **性能权衡**：阴影射线增加了计算开销，但通过 GPU 并行可以有效应对；对于更复杂的场景，可考虑阴影缓存或降低阴影采样率。
4. **调试技巧**：阴影问题容易出现，使用颜色标记（如将阴影区域染成红色）可帮助快速定位 bug。

本次选做完整实现了两项扩展，代码运行稳定，效果明显，加深了对光照模型和阴影算法的理解，为后续学习全局光照、软阴影等技术奠定了坚实基础。

---

## 八、参考资料

- 实验四教程（基础 Phong 模型）
- Taichi 官方文档
- 计算机图形学教材：Blinn-Phong 模型、阴影映射基础
