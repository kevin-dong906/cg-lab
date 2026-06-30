姓名：何佳林    
学号：202411081001   
专业：计算机科学与技术（公费师范）

# 实验报告：Phong光照模型

## 一、实验目的

1. 理论层面：理解 Phong 局部光照模型的三个分量（环境光、漫反射、镜面高光）及其物理意义，掌握向量运算（法向量、光线方向、视线方向、反射向量）在光照计算中的应用。
2. 工程层面：利用 Taichi 实现基于光线投射的交互式渲染系统，通过 GPU 并行计算为每个像素着色。
3. 实践层面：通过 UI 滑动条实时调节材质参数（Ka, Kd, Ks, Shininess），直观感受各参数对渲染效果的影响。

---

## 二、实验原理：Phong 光照模型

Phong 光照模型是一种经验模型，它将物体表面反射的光分解为三个独立分量，最终叠加得到像素颜色：

**总颜色 = 环境光分量 + 漫反射分量 + 镜面高光分量**

### 2.1 环境光（Ambient）
模拟间接照明，均匀照亮所有表面，与观察方向无关。
```
I_ambient = K_a * C_light * C_object
```
- K_a：环境光反射系数（0~1）
- C_light：光源颜色（此处为纯白 (1,1,1)）
- C_object：物体基础颜色

### 2.2 漫反射（Diffuse）
模拟粗糙表面的散射，遵循 Lambert 余弦定律，光照强度与入射角余弦成正比。
```
I_diffuse = K_d * max(0, N·L) * C_light * C_object
```
- K_d：漫反射系数
- N：表面单位法向量
- L：指向光源的单位向量
- 点积为负说明光源在背面，应截断为0

### 2.3 镜面高光（Specular）
模拟光滑表面的高光反射，依赖视线方向与理想反射方向的接近程度。
```
I_specular = K_s * max(0, R·V)^n * C_light
```
- K_s：镜面反射系数
- R：光线反射方向（入射光关于法线的反射）
- V：指向观察者的单位向量
- n：高光指数（Shininess），越大高光越集中

反射向量 R 可由公式计算：
```
R = 2 * (L·N) * N - L
```
（L 为入射光方向，指向光源；注意符号）

所有向量必须在计算前归一化，最终颜色需限制在 [0,1] 区间以避免过曝。

---

## 三、实验任务与实现

### 3.1 任务概览

| 任务 | 描述 | 关键实现 |
| :---: | :--- | :--- |
| 1 | 构建三维场景（球体+圆锥） | 隐式几何相交函数 |
| 2 | 光线求交与深度测试 | `min_t` 深度竞争 |
| 3 | Phong 着色器 | 计算 N, L, V, R 并合成颜色 |
| 4 | UI 交互面板 | 滑动条调整材质参数 |

### 3.2 场景与几何体定义

- **摄像机**：固定在 (0, 0, 5)，向 -Z 方向观察。
- **光源**：点光源位于 (2, 3, 4)，颜色为纯白 (1,1,1)。
- **红色球体**：中心 (-1.2, -0.2, 0)，半径 1.2，基础颜色 (0.8, 0.1, 0.1)。
- **紫色圆锥**：顶点 (1.2, 1.2, 0)，底面 y = -1.4，底面半径 1.2，基础颜色 (0.6, 0.2, 0.8)。
- **背景**：深青色 (0.05, 0.15, 0.15)。

### 3.3 光线与球体求交

球体方程：|p - C|^2 = r^2，代入光线 p(t) = ro + t * rd，求解一元二次方程：
```
A = rd·rd = 1 (rd归一化)
B = 2 * (ro - C)·rd
C = (ro - C)·(ro - C) - r^2
判别式 Δ = B^2 - 4AC
```
若 Δ ≥ 0，取正根 t = (-B - sqrt(Δ))/2（最近交点），并计算法向量 N = normalize(p - C)。

**代码片段（球体相交）**：
```python
@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal
```

### 3.4 光线与圆锥求交

圆锥竖直放置，顶点在 apex，底面 y = base_y（base_y < apex.y），底面半径为 radius。在局部坐标系中，圆锥表面满足：
```
x^2 + z^2 = k * (y_apex - y)^2
```
其中 k = (radius / height)^2，height = apex.y - base_y。

将光线代入，得到关于 t 的二次方程，解出正根后还需验证交点是否在圆锥高度范围内（局部 y 坐标在 [-height, 0] 之间）。法向量通过梯度求得。

**代码片段（圆锥相交）**：
```python
@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    ro_local = ro - apex
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    # 求解并检查范围...
    return t, normal
```

### 3.5 深度测试（Z-buffer 思想）

在 `render` 内核中，对每个像素发射光线，依次测试球体和圆锥，记录最近的交点（最小正 t）及其法向量和颜色。实现逻辑：
```python
min_t = 1e10
hit_normal = ti.Vector([0.0, 0.0, 0.0])
hit_color = ti.Vector([0.0, 0.0, 0.0])

t_sph, n_sph = intersect_sphere(...)
if 0 < t_sph < min_t:
    min_t = t_sph; hit_normal = n_sph; hit_color = red

t_cone, n_cone = intersect_cone(...)
if 0 < t_cone < min_t:
    min_t = t_cone; hit_normal = n_cone; hit_color = purple
```

若 `min_t` 仍为初始值，则未击中任何物体，显示背景色。

### 3.6 Phong 着色器实现

在最近交点处，计算：
- 交点位置 `p = ro + rd * min_t`
- 法向量 `N = hit_normal`（已归一化）
- 光线方向 `L = normalize(light_pos - p)`
- 视线方向 `V = normalize(ro - p)`
- 反射方向 `R = normalize(reflect(-L, N))`（注意 reflect 函数通常输入入射光方向，指向光源的反方向）

然后计算三个分量并累加，最后 clamp 到 [0,1]。

**关键代码**：
```python
ambient = Ka[None] * light_color * hit_color
diff = ti.max(0.0, N.dot(L))
diffuse = Kd[None] * diff * light_color * hit_color
R = normalize(reflect(-L, N))
spec = ti.max(0.0, R.dot(V)) ** shininess[None]
specular = Ks[None] * spec * light_color
color = ambient + diffuse + specular
pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)
```

### 3.7 UI 交互面板

使用 `ti.ui.Window` 的 GUI 子窗口，添加四个滑动条：
- Ka: 0.0 ~ 1.0，默认 0.2
- Kd: 0.0 ~ 1.0，默认 0.7
- Ks: 0.0 ~ 1.0，默认 0.5
- Shininess: 1.0 ~ 128.0，默认 32.0

每个滑动条绑定到对应的 `ti.field`，修改后 `render` 内核自动读取最新值。

**代码**：
```python
with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.22):
    Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
    Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
    Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
    shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)
```

---

## 四、代码深度分析

### 4.1 数据结构

| 变量 | 类型 | 用途 |
| :--- | :--- | :--- |
| `pixels` | `ti.Vector.field(3, ti.f32, shape=(res_x, res_y))` | 帧缓冲区，存储 RGB |
| `Ka, Kd, Ks, shininess` | `ti.field(ti.f32, shape=())` | 材质参数，标量，可被 GUI 修改 |
| 光源位置、颜色 | 硬编码常量 | 场景固定 |

### 4.2 并行渲染策略

- `render()` 是 `@ti.kernel`，其外层循环 `for i, j in pixels` 会被 Taichi 自动并行化到 GPU 线程。
- 每个线程独立计算一个像素的颜色，彼此无依赖，适合大规模并行。
- 光线求交和着色计算全部在 GPU 内完成，无需 CPU 干预，效率极高。

### 4.3 向量归一化与防除零

- `normalize(v)` 函数内部使用 `v / (v.norm() + 1e-5)` 防止零向量。
- 所有参与点乘的向量（N, L, V, R）都需归一化，否则光照强度错误。

### 4.4 参数实时更新机制

- `Ka[None]` 等标量 field 的 `None` 索引表示访问标量值。
- 滑动条修改后，`Ka[None]` 的值会更新，下一次 `render()` 执行时自动生效，无需额外同步。

### 4.5 渲染循环流程

```
while window.running:
    render()                         # GPU 并行渲染
    canvas.set_image(pixels)         # 显示帧缓冲
    with gui.sub_window(...):        # 绘制 UI
        Ka[None] = gui.slider_float(...)
        ...                          # 其他滑块
    window.show()                    # 提交显示
```

---

## 五、运行与交互效果

### 5.1 运行方式

确保已安装 Taichi，终端执行：
```bash
python phong_demo.py
```

### 5.2 初始效果

- 窗口 800×600，背景深青色。
- 左侧红色球体，右侧紫色圆锥，二者在中心区域有重叠。
- 高光区域明显，尤其是球体表面和圆锥尖端。

### 5.3 UI 交互

拖动滑块可实时调整材质外观：

| 参数 | 增大效果 | 减小效果 |
| :--- | :--- | :--- |
| Ka | 整体变亮，暗部提升 | 整体变暗，暗部更黑 |
| Kd | 漫反射增强，颜色更饱满 | 漫反射减弱，颜色暗淡 |
| Ks | 高光亮度增加，更亮 | 高光减弱，表面更粗糙 |
| Shininess | 高光更集中、更小 | 高光扩散、更大 |

### 5.4 性能

- 800×600 = 480,000 像素，每个像素进行两次求交和着色计算，在 GPU 上实时运行，帧率稳定在 60 FPS。

---

## 六、常见问题与解决方案

| 问题现象 | 可能原因 | 解决方法 |
| :--- | :--- | :--- |
| 全黑画面 | 向量未归一化 | 检查 N, L, V 是否调用 `.normalized()` |
| 画面有黑色噪点 | 漫反射或镜面点积为负未截断 | 使用 `ti.max(0.0, dot)` |
| 颜色过曝发白 | RGB 累加超过 1.0 | 输出前使用 `ti.math.clamp(color, 0.0, 1.0)` |
| 圆锥显示不全 | 求交范围判断有误 | 确认局部 Y 坐标范围 `[-H, 0]`，以及 t 的符号 |
| 物体重叠闪烁 | 深度测试未正确更新 `min_t` | 确保每次求交后比较 `t > 0 and t < min_t` |
| UI 无响应 | 滑块未绑定到 field | 确认 `Ka[None]` 与 `gui.slider_float` 返回值赋值 |

---

## 七、实验总结

本次实验完整实现了基于 Phong 模型的交互式渲染器，核心收获包括：

1. **光照模型理解**：通过手动实现三个光照分量，清晰体会到环境光提供基础亮度，漫反射决定表面明暗过渡，镜面高光带来光泽感。
2. **向量运算实战**：熟练使用归一化、点积、反射向量等操作，并认识到归一化对正确光照的关键作用。
3. **光线与隐式曲面求交**：掌握了球体和圆锥的数学求交方法，并实现了深度测试解决遮挡问题。
4. **Taichi 并行渲染**：利用 `@ti.kernel` 实现像素级并行，代码简洁且性能优异，为后续复杂渲染器奠定基础。
5. **实时参数调优**：通过 UI 交互直观体验材质参数对视觉效果的影响，增强了图形学直觉。

---

## 八、选做内容思路（简要）

### 8.1 Blinn-Phong 模型
- 计算半程向量 H = normalize(L + V)，用 N·H 代替 R·V。
- Blinn-Phong 高光更柔和，尤其是在观察方向与光线方向夹角较大时，边缘高光更自然。

### 8.2 硬阴影
- 在交点 p 处向光源方向发射阴影射线，检测是否被其他物体阻挡。
- 若被阻挡，仅保留环境光分量，否则计算完整 Phong 颜色。

---

## 九、参考资料

- 实验教程文档（课堂提供）
- Taichi 官方文档：Field、GUI、向量运算
- 计算机图形学（教材）Phong 光照模型章节
