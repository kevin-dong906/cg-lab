# 实验报告：Phong光照模型扩展——Blinn-Phong与硬阴影（选做部分）

## 一、实验目的

在必做 Phong 光照模型的基础上，进一步实现以下两种增强技术：

1. **Blinn-Phong 模型升级**：用半程向量替代反射向量计算镜面高光，理解两者在高光形状和边缘过渡上的差异，特别是大入射角情况下的表现。
2. **硬阴影（Hard Shadow）**：通过阴影射线检测遮挡关系，为场景添加简单阴影，增强真实感。

通过本次选做，深入理解光照模型的改进方向以及阴影算法的基本原理。

---

## 二、实验原理

### 2.1 Blinn-Phong 模型

原 Phong 模型使用反射向量 R 和视线向量 V 的点积来控制高光分布：
```
spec = (R·V)^n
```
其中 R = 2(L·N)N - L。

Blinn-Phong 模型引入**半程向量 H**，定义为光线方向 L 与视线方向 V 的中间方向：
```
H = normalize(L + V)
```
高光分量改为：
```
spec = (N·H)^n
```
即用法线与半程向量的夹角代替反射方向与视线的夹角。当视线与反射方向重合时，H 与 N 重合；当偏离时，N·H 的衰减比 R·V 更平滑。

优点：
- 物理上更接近微表面理论。
- 高光边缘在大入射角时过渡更自然，不会出现突然截断。
- 计算量略小（省去反射向量计算）。

### 2.2 硬阴影（Shadow Ray）

硬阴影的原理简单：对于每个着色点 P，向光源方向 L 发射一条检测射线，检查在到达光源之前是否与其他物体相交。若相交，则 P 处于阴影中，此时仅保留环境光分量（或加少量漫反射），否则计算完整 Phong 颜色。

算法步骤：
1. 在正常求交后得到最近的交点 P 和法线 N。
2. 计算从 P 指向光源的方向 L。
3. 从 P 沿 L 方向偏移一个小量 eps（防止自阴影）发射阴影射线。
4. 测试该射线是否与场景中的其他物体相交，且交点距离小于 P 到光源的距离。
5. 若被遮挡，则颜色 = ambient；否则颜色 = ambient + diffuse + specular。

---

## 三、实现步骤

本次选做基于必做 Phong 光照模型代码，仅修改 `render` 内核中的着色部分，并新增阴影检测逻辑。

### 3.1 环境准备
- 沿用必做实验的场景（红球+紫圆锥）、光源位置、摄像机位置、背景色。
- 保留四个材质参数 Ka, Kd, Ks, Shininess 及其 UI 控件。
- 为了对比，可以在 UI 中添加一个切换模式按钮，或直接注释切换，本报告采用分别实现两种模式。

### 3.2 Blinn-Phong 替换

在 `render` 内核中，将镜面高光计算部分从：
```
R = normalize(reflect(-L, N))
spec = ti.max(0.0, R.dot(V)) ** shininess[None]
specular = Ks[None] * spec * light_color
```
改为：
```
H = normalize(L + V)
spec = ti.max(0.0, N.dot(H)) ** shininess[None]
specular = Ks[None] * spec * light_color
```
其余不变。

### 3.3 硬阴影实现

在着色阶段，增加阴影检测代码。因为场景中只有两个几何体，我们只需检测从 P 发出的阴影射线是否击中了另一个物体。

核心伪代码：
```
in_shadow = False
dist_to_light = (light_pos - P).norm()
eps = 1e-4
shadow_origin = P + L * eps

# 检测球体遮挡
t_sph, _ = intersect_sphere(shadow_origin, L, sphere_center, sphere_radius)
if 0 < t_sph < dist_to_light:
    in_shadow = True

# 检测圆锥遮挡
t_cone, _ = intersect_cone(shadow_origin, L, cone_apex, cone_base_y, cone_radius)
if 0 < t_cone < dist_to_light:
    in_shadow = True

if in_shadow:
    color = ambient
else:
    color = ambient + diffuse + specular
```

**注意**：
- 阴影射线起点需要偏移，防止与自身相交。
- 需要排除被检测物体自身的交点，但因为我们检查的是不同物体，且偏移后一般不会与自身相交，可以安全使用。
- 如果场景中有多个物体，需要循环检测所有物体。

---

## 四、代码修改详情（关键部分）

以下展示在 `render` 内核中应用 Blinn-Phong 和硬阴影的完整修改（仅包含着色部分，求交部分不变）：

```python
@ti.kernel
def render():
    for i, j in pixels:
        # 光线生成和求交（与必做相同）...
        # 得到 min_t, hit_normal, hit_color, 以及交点 P

        if min_t < 1e9:
            P = ro + rd * min_t
            N = hit_normal
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - P)
            V = normalize(ro - P)

            # 环境光和漫反射（与必做相同）
            ambient = Ka[None] * light_color * hit_color
            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color

            # --- Blinn-Phong 高光 ---
            H = normalize(L + V)
            spec = ti.max(0.0, N.dot(H)) ** shininess[None]
            specular = Ks[None] * spec * light_color

            # --- 硬阴影检测 ---
            in_shadow = False
            dist_to_light = (light_pos - P).norm()
            eps = 1e-4
            shadow_origin = P + L * eps

            # 检测球体遮挡（如果当前击中球体，则检测圆锥；反之检测球体）
            # 为简化，这里检测两个物体，但注意避免自遮挡（因为起点偏移了）
            # 球体检测
            t_sph, _ = intersect_sphere(shadow_origin, L,
                                        ti.Vector([-1.2, -0.2, 0.0]), 1.2)
            if 0 < t_sph < dist_to_light:
                in_shadow = True

            # 圆锥检测
            t_cone, _ = intersect_cone(shadow_origin, L,
                                       ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
            if 0 < t_cone < dist_to_light:
                in_shadow = True

            # 最终颜色
            if in_shadow:
                color = ambient
            else:
                color = ambient + diffuse + specular

            pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)
        else:
            pixels[i, j] = background_color
```

---

## 五、结果分析与对比

### 5.1 Phong vs Blinn-Phong 高光对比

通过并排比较（可切换模式），观察以下差异：

| 观察角度 | Phong 模型表现 | Blinn-Phong 模型表现 |
| :--- | :--- | :--- |
| 正对光源（高光在中心） | 两者高光大小相近，亮度相似 | 无明显差异 |
| 斜向观察，大入射角（光源与法线夹角大） | 高光边缘可能突然消失，出现“撕裂” | 高光过渡平滑，边缘渐变自然 |
| 旋转物体或光源 | 高光区域可能跳动 | 高光移动连续、稳定 |

**理论原因**：在大入射角时，反射方向 R 与视线 V 的夹角变化剧烈，导致 R·V 快速下降，而 N·H 的变化更为平滑。Blinn-Phong 更符合微表面模型，因此视觉上更真实。

### 5.2 硬阴影效果

- 当红球位于光源和圆锥之间时，圆锥部分区域被红球遮挡，呈现暗色（仅环境光），与周围形成鲜明对比。
- 阴影边缘锐利（硬阴影），未作柔化处理。
- 效果直观，增强了场景立体感和物体之间的空间关系。

### 5.3 综合影响

同时启用两项扩展后，场景既具有更自然的镜面高光，又包含物体间的阴影，视觉效果明显优于纯 Phong 模型。计算开销虽有所增加，但在 GPU 并行下仍实时运行。

---

## 六、性能与实现注意事项

- **阴影射线额外开销**：每个像素多两次求交（球体和圆锥），共 4 次求交，但 GPU 并行下性能影响不大。
- **自阴影问题**：起点必须偏移 eps，否则会与自己表面相交导致错误阴影。
- **数值稳定性**：使用 `eps = 1e-4` 避免浮点误差。
- **多物体阴影**：若场景物体增多，需循环检测所有物体，可用统一列表管理。

---

## 七、总结

通过完成两项选做内容，我获得以下认知与经验：

1. **光照模型的演进**：Blinn-Phong 模型在计算简便性和视觉效果上均优于 Phong，是现代实时渲染的主流选择，其高光边缘的自然过渡在大入射角场景下尤为明显。
2. **阴影的加入**：即便是简单的硬阴影，也能显著提升场景真实感和层次感，是光照渲染中不可或缺的一环。
3. **调试技巧**：阴影问题容易出现，建议先用颜色标记（如将阴影区域染红）来快速定位错误。
4. **性能权衡**：阴影射线增加计算量，但通过合理设计（如限定射线长度、减少采样）可保证实时性。

本次选做实现了完整的 Blinn-Phong 光照模型和硬阴影效果，代码运行稳定，为后续学习更高级的渲染技术（如软阴影、全局光照）奠定了良好基础。

---

## 八、参考资料

- 实验四教程（基础 Phong 模型）
- Taichi 官方文档（向量操作、GUI）
- 计算机图形学教材：Blinn-Phong 模型、阴影映射基础
