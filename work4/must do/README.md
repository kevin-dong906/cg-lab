# 实验报告：Phong光照模型（必做部分）

## 一、实验目的

1. **理论理解**：理解并掌握局部光照的基本原理，区分环境光（Ambient）、漫反射（Diffuse）和镜面高光（Specular）三个分量的物理意义和计算方式。
2. **数学基础**：熟练掌握三维空间中的向量运算，包括法向量计算、光线方向、视线方向以及反射向量的求解与归一化。
3. **工程实践**：掌握利用 Taichi 实现交互式渲染的方法，通过 UI 控件实时调节材质参数，直观感受各参数对渲染结果的影响。

---

## 二、实验原理（Phong 光照模型）

Phong 光照模型是一种经验模型，将物体表面反射的光分解为三个分量，最终颜色为三者之和：

**总颜色 = 环境光 + 漫反射 + 镜面高光**

### 2.1 环境光（Ambient）
模拟场景中经过多次反射后均匀分布的光，与光源方向、视线方向无关，仅取决于物体颜色和环境光系数。

公式（文本形式）：
```
I_ambient = K_a * C_light * C_object
```
- K_a：环境光反射系数（0~1）
- C_light：光源颜色（本实验中为纯白 (1,1,1)）
- C_object：物体基础颜色

### 2.2 漫反射（Diffuse）
模拟粗糙表面向各个方向均匀散射光，遵循 Lambert 余弦定律，强度与光线入射角（光源方向与法线夹角）的余弦成正比。

公式：
```
I_diffuse = K_d * max(0, N·L) * C_light * C_object
```
- K_d：漫反射系数
- N：表面单位法向量
- L：指向光源的单位向量
- max(0, N·L)确保背面不受光

### 2.3 镜面高光（Specular）
模拟光滑表面反射的强光，集中在反射方向附近，强度与视线方向和理想反射方向的夹角有关。

公式：
```
I_specular = K_s * max(0, R·V)^n * C_light
```
- K_s：镜面反射系数
- R：理想反射方向（入射光关于法线的反射）
- V：指向观察者的单位向量
- n：高光指数（Shininess），越大高光越集中

反射向量 R 的计算：
```
R = 2 * (L·N) * N - L
```
注意 L 是指向光源的方向，反射公式中入射光方向为 -L。

---

## 三、实验任务与实现

本实验共包含四个必做任务，具体实现如下。

### 3.1 任务1：构建代码驱动的三维场景

不使用外部模型，通过光线投射（Ray Casting）在 GPU 内核中隐式定义两个几何体：

- **红色球体**：中心 (-1.2, -0.2, 0)，半径 1.2，基础颜色 (0.8, 0.1, 0.1)
- **紫色圆锥**：顶点 (1.2, 1.2, 0)，底面 y = -1.4，底面半径 1.2，基础颜色 (0.6, 0.2, 0.8)

**摄像机**：固定在 (0, 0, 5)，向 -Z 方向观察。  
**光源**：点光源位于 (2, 3, 4)，颜色为纯白 (1,1,1)。  
**背景色**：深青色 (0.05, 0.15, 0.15)。

### 3.2 任务2：光线求交与深度测试

为每个像素发射光线，分别测试与球体和圆锥的交点，记录最近的交点（最小正 t）及其法向量和颜色。通过类似 Z-buffer 的竞争逻辑实现遮挡。

**球体求交**：代入光线方程 p(t)=ro+t*rd 到球面方程 |p-C|^2=r^2，解一元二次方程取正根。

**圆锥求交**：将圆锥用局部坐标表示，代入光线方程，解二次方程并验证交点是否在高度范围内。

关键变量：
- `min_t`：记录最近交点距离，初始为极大值。
- `hit_normal` 和 `hit_color` 分别存储对应法向量和颜色。

### 3.3 任务3：编写 Phong 着色器

在最近交点 P 处计算：
- 法向量 N（已归一化）
- 光源方向 L = normalize(light_pos - P)
- 视线方向 V = normalize(ro - P)
- 反射方向 R = normalize(reflect(-L, N))

然后按公式分别计算 ambient、diffuse、specular，求和后 clamp 到 [0,1]。

### 3.4 任务4：UI 交互面板

使用 `ti.ui.Window` 的子窗口提供四个滑动条，实时调整材质参数：

| 参数 | 含义 | 范围 | 默认值 |
| :---: | :--- | :---: | :---: |
| Ka | 环境光系数 | 0.0 ~ 1.0 | 0.2 |
| Kd | 漫反射系数 | 0.0 ~ 1.0 | 0.7 |
| Ks | 镜面高光系数 | 0.0 ~ 1.0 | 0.5 |
| Shininess | 高光指数 | 1.0 ~ 128.0 | 32.0 |

---

## 四、核心代码分析

### 4.1 数据结构和参数定义

```python
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())
```
- `pixels` 是帧缓冲区，每个像素为 RGB 三元向量。
- 四个材质参数定义为标量 field，便于在 GPU 内核中读取并在主线程更新。

### 4.2 向量工具函数

```python
@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N
```
- `normalize` 加入微小量防除零。
- `reflect` 输入入射方向 I，输出反射方向。

### 4.3 球体相交函数

```python
@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b*b - 4.0*c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal
```
求解二次方程，取较小正根，法向量从球心指向交点。

### 4.4 圆锥相交函数

```python
@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    H = apex.y - base_y
    k = (radius / H)**2
    ro_local = ro - apex
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x*rd.x + ro_local.z*rd.z - k*ro_local.y*rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    # 解二次方程，并验证交点高度范围
    # 法线计算：normal = (p_local.x, -k*p_local.y, p_local.z)
```
圆锥在局部坐标系中满足 x^2+z^2 = k*y^2（顶点为原点，向下延伸）。求根后检查局部 y 是否在 [-H, 0] 内。

### 4.5 渲染内核（Phong 着色）

```python
@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x/2.0) / res_y * 2.0
        v = (j - res_y/2.0) / res_y * 2.0
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))
        # 求交深度测试...
        # 着色
        L = normalize(light_pos - p)
        V = normalize(ro - p)
        ambient = Ka[None] * light_color * hit_color
        diff = ti.max(0.0, N.dot(L))
        diffuse = Kd[None] * diff * light_color * hit_color
        R = normalize(reflect(-L, N))
        spec = ti.max(0.0, R.dot(V)) ** shininess[None]
        specular = Ks[None] * spec * light_color
        color = ambient + diffuse + specular
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)
```

- 外层循环自动并行到 GPU 线程。
- 注意所有向量均调用 `normalize` 归一化。
- 使用 `ti.max` 截断负值，防止背面光。
- 最终 clamp 防止过曝。

### 4.6 主循环与 UI

```python
def main():
    window = ti.ui.Window("Phong Shading Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    # 初始化参数
    while window.running:
        render()
        canvas.set_image(pixels)
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.22):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)
        window.show()
```
- 每帧先渲染再显示，UI 控件绑定到 field，修改后实时生效。

---

## 五、运行与效果

### 5.1 运行方式
确保已安装 Taichi，在终端执行：
```
python phong_demo.py
```

### 5.2 初始画面
- 窗口 800×600，背景深青色。
- 左侧红色球体，右侧紫色圆锥。
- 球体和圆锥表面有明暗变化，带有白色高光。

### 5.3 UI 参数调节效果

| 参数 | 增大后视觉变化 | 减小后视觉变化 |
| :--- | :--- | :--- |
| Ka | 整体变亮，暗部细节显现 | 整体变暗，阴影区域更黑 |
| Kd | 漫反射增强，颜色更鲜艳 | 漫反射减弱，颜色变灰 |
| Ks | 高光更亮、面积更广 | 高光变暗，表面更粗糙 |
| Shininess | 高光区域缩小，更集中 | 高光扩散，更柔和 |

### 5.4 遮挡验证
球体与圆锥在场景中有重叠区域，由于实现了深度测试，远距离物体被正确遮挡，不会出现前后颠倒。

---

## 六、常见问题与解决办法

| 现象 | 原因 | 解决方法 |
| :--- | :--- | :--- |
| 全黑画面 | 向量未归一化 | 确保所有向量用 `normalize` 处理 |
| 出现黑色噪点 | N·L 为负未截断 | 使用 `ti.max(0.0, dot)` |
| 颜色发白 | 累计值超过1.0 | 输出前 `ti.math.clamp(color, 0.0, 1.0)` |
| 圆锥显示不完整 | 求交范围判断错误 | 检查局部 y 范围 `[-H, 0]` 和 t 的正负 |
| 物体被错误遮挡 | 深度更新逻辑错误 | 确保 `if 0 < t_sph < min_t` 条件正确 |
| UI 无响应 | 滑块未正确绑定 | 确认使用 `Ka[None] = gui.slider_float(...)` |

---

## 七、总结

本次实验完整实现了基于光线投射的 Phong 光照模型渲染器，主要收获包括：

- **光照模型理解**：通过手动实现三个分量，清晰区分了环境光提供基础亮度、漫反射决定明暗过渡、镜面高光带来光泽感的作用。
- **向量运算实战**：熟练运用归一化、点积、反射向量等操作，并认识到归一化对正确光照的关键作用。
- **光线求交与深度测试**：实现了球体和圆锥的隐式求交，并通过深度竞争处理遮挡，为后续复杂场景渲染打下基础。
- **Taichi 并行编程**：利用 `@ti.kernel` 实现像素级并行，代码简洁高效，实时交互流畅。
- **参数调优体验**：通过 UI 滑块实时调整材质参数，直观感受每个系数对视觉效果的影响，增强了图形直觉。

本实验为后续学习更高级的渲染技术（如纹理映射、阴影、全局光照）奠定了坚实基础。所有必做功能均已正确实现并通过测试。
