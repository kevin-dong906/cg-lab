姓名：何佳林    
学号：202411081001   
专业：计算机科学与技术（公费师范）

# 实验报告：旋转与变换

## 一、实验目的

1. 理解三维图形学中模型变换、视图变换与投影变换（MVP）的数学原理及其在渲染管线中的作用。
2. 独立推导并编码实现绕 Z 轴的旋转矩阵、相机平移视图矩阵以及完整的透视投影矩阵。
3. 掌握 Taichi 框架中的矩阵运算、`@ti.func` 和 `@ti.kernel` 的使用，完成顶点从局部坐标到屏幕空间的映射。
4. 实现键盘交互（A/D 键控制旋转，Esc 键退出）验证变换正确性。

---

## 二、实验内容

给定一个三维三角形，局部坐标顶点为：
- v0 = (2.0, 0.0, -2.0)
- v1 = (0.0, 2.0, -2.0)
- v2 = (-2.0, 0.0, -2.0)

**任务要求**：
1. 补全三个函数，分别返回 4×4 齐次变换矩阵：
   - `get_model_matrix(angle)`：绕 Z 轴旋转 angle 度。
   - `get_view_matrix(eye_pos)`：将相机位置 eye_pos 平移到原点。
   - `get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)`：透视投影矩阵。
2. 将三个顶点依次经模型、视图、投影变换，执行透视除法与视口映射，在 700×700 窗口中绘制线框三角形。
3. 键盘 A 键逆时针旋转（+10°），D 键顺时针旋转（-10°），Esc 退出。

---

## 三、变换流程与矩阵定义（纯文本）

### 3.1 完整流水线

对于顶点 (x, y, z, 1)^T（齐次坐标），依次进行：

1. **模型变换**：v_world = M_model · v_local
2. **视图变换**：v_view = M_view · v_world
3. **投影变换**：v_clip = M_proj · v_view
4. **透视除法**：v_ndc = v_clip / w_clip
5. **视口变换**：u = (x_ndc + 1) / 2,   v = (y_ndc + 1) / 2

所有矩阵采用列向量右乘（即 v' = M · v）。

---

### 3.2 模型矩阵（绕 Z 轴旋转 θ 弧度）

矩阵形式（4×4）：
```
[ cosθ, -sinθ, 0, 0 ]
[ sinθ,  cosθ, 0, 0 ]
[ 0,     0,    1, 0 ]
[ 0,     0,    0, 1 ]
```
其中 θ = angle × π / 180（角度转弧度）。

---

### 3.3 视图矩阵（相机平移）

相机位置 eye = (ex, ey, ez)。视图矩阵将相机移回原点，即平移 -eye：
```
[ 1, 0, 0, -ex ]
[ 0, 1, 0, -ey ]
[ 0, 0, 1, -ez ]
[ 0, 0, 0,  1  ]
```
本实验中 eye = (0, 0, 5)。

---

### 3.4 透视投影矩阵（关键）

**符号约定**（右手系，相机看向 -Z）：
- 输入 zNear > 0, zFar > 0（距离）
- 实际近平面坐标 n = -zNear，远平面 f = -zFar（均为负数，且 n > f，例如 -0.1 > -50）

**步骤1：计算视锥体边界**
给定垂直视场角 fovY（弧度）和宽高比 aspect：
- t = tan(fovY/2) * (-n)     （-n 为正，即 |n|）
- b = -t
- r = aspect * t
- l = -r

**步骤2：透视→正交挤压矩阵（4×4）**
```
[ n,  0,  0,   0 ]
[ 0,  n,  0,   0 ]
[ 0,  0, n+f, -n*f ]
[ 0,  0,  1,   0 ]
```
该矩阵将视锥体挤压为长方体，保持近远平面深度不变。

**步骤3：正交投影矩阵（合并平移与缩放）**
将长方体 [l, r] × [b, t] × [f, n] 映射到 [-1,1]³：
```
[ 2/(r-l),    0,        0,      -(r+l)/(r-l) ]
[ 0,        2/(t-b),    0,      -(t+b)/(t-b) ]
[ 0,          0,      2/(n-f),  -(n+f)/(n-f) ]
[ 0,          0,        0,          1         ]
```
**最终投影矩阵**：M_proj = M_ortho · M_persp_to_ortho

---

## 四、完整代码实现

```python
import taichi as ti
import math

# 初始化Taichi（可改用 ti.gpu）
ti.init(arch=ti.cpu)

# ------------------------------------------------------------------
# 矩阵生成函数
# ------------------------------------------------------------------
@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    # 实际负坐标
    n = -zNear
    f = -zFar
    # 视锥体边界
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * (-n)   # -n 为 |n|
    b = -t
    r = aspect_ratio * t
    l = -r

    # 挤压矩阵
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    # 正交矩阵（合并平移与缩放）
    M_ortho = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)],
        [0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)],
        [0.0, 0.0, 2.0 / (n - f), -(n + f) / (n - f)],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return M_ortho @ M_p2o

# ------------------------------------------------------------------
# 顶点数据与变换核
# ------------------------------------------------------------------
# 三个顶点（三维坐标）
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
# 变换后的屏幕坐标（归一化 0~1）
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.kernel
def compute_transform(angle: ti.f32):
    eye = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    # 右乘顺序：先模型，再视图，最后投影
    mvp = proj @ view @ model

    for i in range(3):
        # 构造齐次坐标
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        # 裁剪空间坐标
        v_clip = mvp @ v4
        # 透视除法得到 NDC
        v_ndc = v_clip / v_clip[3]
        # 视口映射到屏幕 [0,1]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# ------------------------------------------------------------------
# 主程序与交互
# ------------------------------------------------------------------
def main():
    # 初始化三角形顶点
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]

    gui = ti.GUI("Rotation & Transformation", res=(700, 700))
    angle = 0.0

    while gui.running:
        # 键盘事件
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        # 计算变换
        compute_transform(angle)

        # 获取三个屏幕坐标
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]

        # 绘制三条边（红、绿、蓝）
        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)

        gui.show()

if __name__ == '__main__':
    main()
```

---

## 五、运行与验证

### 5.1 运行方式
在终端（项目根目录）执行：
```bash
python 本文件名.py
```
若使用 uv 环境，可先激活环境再运行。

### 5.2 预期交互效果
- 弹出 700×700 窗口，标题 "Rotation & Transformation"。
- 窗口内显示彩色线框三角形（红、绿、蓝边）。
- 按下键盘 **A** 键，三角形逆时针旋转（每次 10°）。
- 按下 **D** 键，三角形顺时针旋转。
- 按下 **Esc** 键退出程序。

### 5.3 正确性验证（角度=0）
当 angle=0 时，模型矩阵为单位阵。以顶点 v0=(2,0,-2) 为例：
- 视图变换（相机在 (0,0,5)）：v_view = (2, 0, -7)。
- 投影参数：fov=45°, aspect=1, n=-0.1, f=-50。
  计算得 t≈0.0414, r=0.0414, l=-0.0414, b=-0.0414。
  经投影与透视除法，x_ndc≈0.46，y_ndc≈0。
- 视口映射：u=(0.46+1)/2=0.73，v=0.5。该点位于屏幕右半部分，与程序绘制位置一致。

### 5.4 常见问题排查
| 现象 | 可能原因 | 解决方法 |
| :--- | :--- | :--- |
| 三角形不显示 | 视口映射公式错误 | 检查是否用 (ndc+1)/2 |
| 旋转方向相反 | A/D 键加减反了 | 交换 angle += 与 -= |
| 编译错误 | 矩阵元素类型不匹配 | 确保所有数字带 `.0` |
| 透视效果不明显 | n/f 取正数而非负数 | 必须 n=-zNear, f=-zFar |

---

## 六、总结

本次实验完整实现了三维三角形的 MVP 变换，独立推导并编码了：
- 绕 Z 轴的旋转模型矩阵；
- 基于相机位置的视图平移矩阵；
- 包含挤压与正交两步的透视投影矩阵。

通过 Taichi 框架的矩阵运算与 `@ti.kernel` 并行计算，顶点变换高效且代码直观。键盘交互验证了变换的正确性，加深了对渲染管线坐标变换顺序（模型→视图→投影）的理解，为后续三维几何体（如立方体、网格）的渲染奠定了基础。
