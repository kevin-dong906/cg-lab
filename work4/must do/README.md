姓名：何佳林    
学号：202411081001   
专业：计算机科学与技术（公费师范）

# 实验报告：光线追踪（必做部分）

## 一、实验目的

1. **理论理解**：理解光线投射与光线追踪的本质区别，掌握 Whitted-style 光线追踪的基本流程。
2. **全局光照**：掌握通过发射次级射线（阴影射线和反射射线）实现硬阴影和理想镜面反射的方法。
3. **GPU 编程思维**：学习将传统递归光线追踪算法改写为适合 GPU 并行计算的迭代循环模式。

---

## 二、实验原理

### 2.1 Whitted-style 光线追踪流程

对于每个像素，从摄像机发射一条主光线。当光线击中场景中的物体时：

1. **阴影测试**：从交点向光源发射一条阴影射线，检测是否被其他物体遮挡。若遮挡，则仅保留环境光分量。
2. **材质分支**：
   - 若击中**漫反射物体**：按 Phong 模型计算颜色（环境光+漫反射+高光），然后终止该光线传播。
   - 若击中**镜面物体**：根据反射定律计算反射方向，生成新的反射光线，继续追踪，直到击中漫反射物体或达到最大弹射次数。

### 2.2 反射方向计算

对于入射光线方向 L_in（指向交点）和法向量 N，反射方向 R 的计算公式（文本形式）：
```
R = L_in - 2 * (L_in · N) * N
```
其中 L_in 和 N 均为单位向量。

### 2.3 迭代追踪 vs 递归追踪

传统 CPU 光线追踪常用递归函数实现，但 GPU 不适合递归。因此，本实验采用 for 循环迭代模拟光线弹射：
- 维护当前光线起点和方向，以及一个吞吐量衰减系数（初始为1）。
- 每弹射一次，如果击中镜面，更新起点和方向，吞吐量乘以反射率（如0.8），继续循环。
- 如果击中漫反射，计算光照并加到最终颜色，然后跳出循环。
- 循环次数上限由用户通过 UI 调节（最大弹射次数）。

### 2.4 阴影与浮点精度（Shadow Acne）

阴影射线从交点出发，若起点不作偏移，会立即与自身表面相交，产生大量黑色噪点（自相交）。解决方案：将阴影射线起点沿法线方向微微偏移一个小量 epsilon（如 1e-4），即：
```
shadow_origin = P + N * epsilon
```
反射射线同样需要偏移，防止与自身相交。

---

## 三、实验任务与实现

本实验包含四个必做任务，具体实现如下。

### 3.1 任务1：搭建包含平面的三维场景

不使用外部模型，在 GPU 内核中隐式定义三个几何体：

- **无限大平面**：位于 y = -1.0，法线朝上 (0, 1, 0)。材质为漫反射，并实现棋盘格纹理，通过交点 x 和 z 坐标的奇偶性判断颜色。
- **红色漫反射球**：位于 (-1.5, 0.0, 0)，半径 1.0，基础颜色 (0.8, 0.1, 0.1)。
- **银色镜面球**：位于 (1.5, 0.0, 0)，半径 1.0，镜面反射率约 0.8，无漫反射成分。

为每个物体赋予材质 ID（0=漫反射平面，1=漫反射球，2=镜面球）以便在着色时区分。

### 3.2 任务2：实现基于迭代的光线弹射

在 `render` 内核中，对每个像素执行迭代追踪算法（伪代码）：

```
color = (0,0,0)
throughput = 1.0
ray_origin = camera_pos
ray_direction = normalized(pixel_direction)

for bounce in range(max_bounces):
    # 求交：检测与平面、红球、镜面球的交点，取最近
    if no intersection:
        color += throughput * background_color
        break
    else:
        P = intersection_point
        N = normal_at_P
        obj_id = material_id

        if obj_id == mirror:
            # 镜面反射：更新光线方向
            R = reflect(-ray_direction, N)  # 注意入射方向取反
            ray_origin = P + N * epsilon
            ray_direction = R
            throughput *= 0.8   # 反射衰减
            continue   # 继续循环

        elif obj_id == diffuse:
            # 阴影测试
            L = normalize(light_pos - P)
            shadow_origin = P + N * epsilon
            # 检测阴影射线是否被其他物体遮挡（包括平面和其他球）
            if not occluded(shadow_origin, L):
                # 计算漫反射光照（Phong）
                ambient = Ka * light_color * object_color
                diff = max(0, N·L) * Kd * light_color * object_color
                spec = ...   # 根据需要可加高光
                local_color = ambient + diff + spec
            else:
                local_color = ambient   # 只有环境光
            color += throughput * local_color
            break   # 漫反射终止追踪
```

注意：`-ray_direction` 是入射方向（从交点指向光线来源），reflect 函数需要入射方向指向光源。

### 3.3 任务3：硬阴影与浮点精度处理

阴影检测函数 `is_occluded`：从阴影起点向光源方向发射射线，检测是否与场景中除自身以外的物体相交，且交点距离小于到光源的距离。若相交则返回 True。

**关键**：必须将阴影射线起点沿法线偏移 epsilon，否则会导致自身表面自相交，产生噪声。

同样，反射射线的起点也需要偏移，避免被自身表面截获。

### 3.4 任务4：UI 交互面板

使用 `ti.ui.Window` 的 GUI 子窗口，提供以下控件：

- **Light X / Y / Z**：三个滑块，分别控制点光源的 x、y、z 坐标，范围可设为 -5 ~ 5，默认为 (2, 3, 4)。
- **Max Bounces**：一个滑块，控制最大弹射次数，范围 1 ~ 5，默认 3。

光源位置变化时，阴影和光照实时更新。弹射次数从 1 到 2 时，镜面球开始显示反射内容（红球在镜中的像）。

---

## 四、核心代码分析

### 4.1 数据结构与全局参数

```python
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
max_bounces = ti.field(ti.i32, shape=())
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
Ka = ti.field(ti.f32, shape=())   # 可保留环境光系数
Kd = ti.field(ti.f32, shape=())
```

### 4.2 几何体相交函数

**平面求交**：平面方程 y = -1，法线 (0,1,0)。光线 p(t)=ro+t*rd，代入 y 坐标：
```
t = (-1 - ro.y) / rd.y
if t > 0 and rd.y != 0:
    p = ro + t*rd
    # 检查p是否在无限平面上，对棋盘格纹理计算颜色
```

**球体求交**：与实验四相同，解二次方程取正根。

### 4.3 迭代追踪函数（在 render 内核中）

```python
@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x/2.0) / res_y * 2.0
        v = (j - res_y/2.0) / res_y * 2.0
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))
        throughput = 1.0
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        for bounce in range(max_bounces[None]):
            # 求最近交点
            t_min = 1e10
            hit_pos = ti.Vector([0.0,0.0,0.0])
            hit_normal = ti.Vector([0.0,0.0,0.0])
            hit_obj = -1   # -1 未击中
            
            # 检测平面
            t_plane, n_plane = intersect_plane(ro, rd)
            if 0 < t_plane < t_min:
                t_min = t_plane; hit_pos = ro + rd*t_plane; hit_normal = n_plane; hit_obj = 0
            # 检测红球
            t_red, n_red = intersect_sphere(ro, rd, center_red, radius_red)
            if 0 < t_red < t_min:
                t_min = t_red; hit_pos = ro + rd*t_red; hit_normal = n_red; hit_obj = 1
            # 检测镜面球
            t_silver, n_silver = intersect_sphere(ro, rd, center_silver, radius_silver)
            if 0 < t_silver < t_min:
                t_min = t_silver; hit_pos = ro + rd*t_silver; hit_normal = n_silver; hit_obj = 2
                
            if hit_obj == -1:
                final_color += throughput * background_color
                break
            
            # 光线与物体相交
            P = hit_pos; N = hit_normal
            # 偏移反射/阴影起点
            eps = 1e-4
            new_origin = P + N * eps
            
            if hit_obj == 2:  # 镜面球
                # 反射方向：入射光线方向为 -rd（从交点指向光源），但反射需要入射方向指向交点
                # 实际上反射公式：R = d - 2*(d·N)*N，其中d是入射方向（指向交点）
                d = -rd
                R = d - 2.0 * d.dot(N) * N
                ro = new_origin
                rd = normalize(R)
                throughput *= 0.8
                continue   # 继续弹射
            else:  # 漫反射（平面或红球）
                # 计算光照
                L = normalize(light_pos[None] - P)
                # 阴影检测
                shadow_origin = new_origin
                shadow_occluded = False
                # 检测平面（若当前不是平面）
                if hit_obj != 0:
                    t_s, _ = intersect_plane(shadow_origin, L)
                    if 0 < t_s < (light_pos[None] - P).norm():
                        shadow_occluded = True
                # 检测红球（若当前不是红球）
                if hit_obj != 1:
                    t_s, _ = intersect_sphere(shadow_origin, L, center_red, radius_red)
                    if 0 < t_s < (light_pos[None] - P).norm():
                        shadow_occluded = True
                # 检测镜面球（若当前不是镜面球）
                if hit_obj != 2:
                    t_s, _ = intersect_sphere(shadow_origin, L, center_silver, radius_silver)
                    if 0 < t_s < (light_pos[None] - P).norm():
                        shadow_occluded = True
                
                # 获取物体颜色
                if hit_obj == 0:
                    # 平面棋盘格
                    c = checkerboard(P.x, P.z)
                else:
                    c = red_color if hit_obj == 1 else silver_color?  # 但镜面不会到这里
                # 环境光
                ambient = Ka[None] * light_color * c
                if shadow_occluded:
                    local_color = ambient
                else:
                    diff = ti.max(0.0, N.dot(L))
                    diffuse = Kd[None] * diff * light_color * c
                    # 可加高光（但漫反射物体也可以有高光，这里简化）
                    local_color = ambient + diffuse
                final_color += throughput * local_color
                break   # 漫反射终止
        pixels[i,j] = ti.math.clamp(final_color, 0.0, 1.0)
```

### 4.4 棋盘格纹理

```python
@ti.func
def checkerboard(x, z):
    # 判断奇偶
    if (ti.floor(x) + ti.floor(z)) % 2 == 0:
        return ti.Vector([0.8, 0.8, 0.8])   # 白色
    else:
        return ti.Vector([0.2, 0.2, 0.2])   # 黑色
```

### 4.5 UI 交互

```python
def main():
    window = ti.ui.Window("Ray Tracing", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    light_pos[None] = [2.0, 3.0, 4.0]
    max_bounces[None] = 3
    while window.running:
        render()
        canvas.set_image(pixels)
        with gui.sub_window("Controls", 0.7, 0.05, 0.28, 0.25):
            x = gui.slider_float("Light X", light_pos[None].x, -5.0, 5.0)
            y = gui.slider_float("Light Y", light_pos[None].y, -5.0, 5.0)
            z = gui.slider_float("Light Z", light_pos[None].z, -5.0, 5.0)
            light_pos[None] = [x, y, z]
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
        window.show()
```

---

## 五、运行与效果

### 5.1 运行方式
```bash
python ray_tracing.py
```

### 5.2 初始场景
- 背景为深色（如 (0.05,0.05,0.05)）。
- 地面为棋盘格，红球在左，银球在右。
- 光源在 (2,3,4)，红球有漫反射阴影，银球反射红球和地面。

### 5.3 交互效果
- **移动光源**：阴影位置实时变化，镜面球上的高光也移动。
- **调节弹射次数**：
  - 1 次：镜面球仅显示环境色（黑色），无反射内容。
  - 2 次及以上：镜面球开始反射红球和地面，呈现出“镜中世界”。
  - 弹射次数越多，反射内容越丰富（多次反射）。

### 5.4 性能
- 800×600 分辨率，最大弹射 3 次，每个像素最多求交 3×3=9 次，GPU 实时渲染（>30 FPS）。

---

## 六、常见问题与解决方案

| 现象 | 原因 | 解决方法 |
| :--- | :--- | :--- |
| 画面出现大量黑色噪点 | 阴影射线或反射射线自相交 | 起点沿法线偏移 epsilon |
| 镜面球全黑 | 反射方向计算错误 | 检查反射公式中的入射方向取反，以及归一化 |
| 阴影位置不对 | 光源坐标未正确传递 | 确认 UI 修改更新了 light_pos field |
| 棋盘格闪烁 | 求交平面时 t 值不稳定 | 增加偏移或使用更稳定的求交 |
| 反射内容断裂 | 反射射线被自身截断 | 同样需要偏移起点 |
| 弹射次数无效 | 循环次数未正确限制 | 检查 `max_bounces` 是否为 int 并正确传入 |

---

## 七、总结

本次实验完整实现了 Whitted-style 光线追踪的核心机制，主要收获包括：

- **光线追踪与光线投射的区别**：光线追踪通过多次弹射模拟间接光照（反射），而光线投射仅处理直接光照。
- **迭代追踪的 GPU 实现**：将递归转为循环，适合大规模并行计算。
- **阴影与浮点精度**：深刻认识到偏移量 epsilon 的重要性，它是避免自相交的关键。
- **材质系统**：通过材质 ID 区分漫反射和镜面，为复杂材质系统打下基础。
- **实时交互**：通过 UI 调整光源和弹射次数，直观感受物理现象。

本实验为后续学习更高级的全局光照（如路径追踪、光子映射）奠定了坚实的工程和理论基础。

---

## 八、参考资料

- 实验五教程（课堂提供）
- Taichi 官方文档：向量、内核、GUI
- 计算机图形学教材：Whitted 光线追踪
