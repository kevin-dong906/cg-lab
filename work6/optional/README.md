# 计算机图形学实验报告
## 质点-弹簧系统布料仿真（拓展：完整弹簧模型 + 空间碰撞）

## 一、实验拓展目标
1. 完善质点-弹簧模型：在**结构弹簧**基础上，补充**剪切弹簧**与**弯曲弹簧**，构建完整布料物理模型，观察并分析不同弹簧对布料形态的影响。
2. 实现空间碰撞检测：在3D场景中添加球体障碍物，完成布料质点与球体的碰撞判定及物理响应，实现布料自然覆盖球体的真实效果。
3. 保持原有核心功能：保留三种数值积分器、速度钳制、GPU并行计算、GGUI交互面板，保证系统完整性与可对比性。

## 二、拓展实验原理
### 2.1 完整质点-弹簧模型（三弹簧体系）
布料仿真中，仅使用结构弹簧无法模拟真实织物的力学特性，需补充剪切与弯曲弹簧约束形变：
1. **结构弹簧（Structural）**
    连接质点上下左右直接相邻节点，抵抗布料**拉伸/压缩**形变，是网格基础支撑。
2. **剪切弹簧（Shear）**
    连接质点对角线相邻节点，抵抗布料**剪切畸变**，避免网格被拉扯为平行四边形。
3. **弯曲弹簧（Bending）**
    连接间隔一个单位的相邻质点，抵抗布料**过度折叠/弯曲**，赋予布料刚性与厚度感。

三种弹簧均遵循**胡克定律**：
$$f_{a} = -k_s (|x_a-x_b| - l_0) \frac{x_a-x_b}{|x_a-x_b|}$$
系统为三种弹簧设置独立刚度系数，分别控制形变抵抗能力。

### 2.2 布料-球体空间碰撞原理
1. **碰撞检测**：计算布料质点到球心的欧氏距离，判断质点是否进入球体内部（距离 < 球体半径）。
2. **位置修正**：将穿透球体的质点沿法向推送回球体表面，消除穿透。
3. **速度响应**：修正质点速度的法向分量，添加弹性恢复效果，模拟真实物理碰撞。

碰撞修正公式：
$$\vec{x}_i = \vec{c}_{sphere} + \frac{\vec{x}_i-\vec{c}_{sphere}}{|\vec{x}_i-\vec{c}_{sphere}|} \cdot r_{sphere}$$
$$\vec{v}_i = \vec{v}_i - (1+e) \cdot (\vec{v}_i \cdot \vec{n}) \cdot \vec{n}$$
其中 $e$ 为恢复系数，$\vec{n}$ 为碰撞法向量。

## 三、拓展核心代码实现
### 3.1 数据结构扩展
```python
# 三种弹簧独立刚度系数
k_struct = 12000.0      # 结构弹簧
k_shear = 6000.0        # 剪切弹簧
k_bend = 1200.0         # 弯曲弹簧

# 弹簧拓扑扩展：每个质点最多12个邻居(4结构+4剪切+4弯曲)
spring_neighbors = ti.field(int, (N * N, 12))
spring_rest_len = ti.field(float, (N * N, 12))
spring_type = ti.field(int, (N * N, 12))  # 0结构 1剪切 2弯曲

# 球体碰撞参数
sphere_center = ti.Vector([1.0, 0.6, 1.0])
sphere_radius = 0.5
restitution = 0.3
```

### 3.2 完整弹簧初始化（结构+剪切+弯曲）
```python
@ti.kernel
def init_spring():
    for i in range(N * N):
        row = i // N
        col = i % N
        cnt = 0
        # 1. 结构弹簧
        if col + 1 < N:
            spring_neighbors[i, cnt] = row * N + (col + 1)
            spring_rest_len[i, cnt] = 0.1
            spring_type[i, cnt] = 0
            cnt += 1
        if row + 1 < N:
            spring_neighbors[i, cnt] = (row + 1) * N + col
            spring_rest_len[i, cnt] = 0.1
            spring_type[i, cnt] = 0
            cnt += 1
        if col - 1 >= 0:
            spring_neighbors[i, cnt] = row * N + (col - 1)
            spring_rest_len[i, cnt] = 0.1
            spring_type[i, cnt] = 0
            cnt += 1
        if row - 1 >= 0:
            spring_neighbors[i, cnt] = (row - 1) * N + col
            spring_rest_len[i, cnt] = 0.1
            spring_type[i, cnt] = 0
            cnt += 1
        
        # 2. 剪切弹簧（对角线）
        if col + 1 < N and row + 1 < N:
            spring_neighbors[i, cnt] = (row + 1) * N + (col + 1)
            spring_rest_len[i, cnt] = 0.1 * ti.sqrt(2)
            spring_type[i, cnt] = 1
            cnt += 1
        if col + 1 < N and row - 1 >= 0:
            spring_neighbors[i, cnt] = (row - 1) * N + (col + 1)
            spring_rest_len[i, cnt] = 0.1 * ti.sqrt(2)
            spring_type[i, cnt] = 1
            cnt += 1
        if col - 1 >= 0 and row + 1 < N:
            spring_neighbors[i, cnt] = (row + 1) * N + (col - 1)
            spring_rest_len[i, cnt] = 0.1 * ti.sqrt(2)
            spring_type[i, cnt] = 1
            cnt += 1
        if col - 1 >= 0 and row - 1 >= 0:
            spring_neighbors[i, cnt] = (row - 1) * N + (col - 1)
            spring_rest_len[i, cnt] = 0.1 * ti.sqrt(2)
            spring_type[i, cnt] = 1
            cnt += 1

        # 3. 弯曲弹簧（间隔一格）
        if col + 2 < N:
            spring_neighbors[i, cnt] = row * N + (col + 2)
            spring_rest_len[i, cnt] = 0.2
            spring_type[i, cnt] = 2
            cnt += 1
        if row + 2 < N:
            spring_neighbors[i, cnt] = (row + 2) * N + col
            spring_rest_len[i, cnt] = 0.2
            spring_type[i, cnt] = 2
            cnt += 1
        if col - 2 >= 0:
            spring_neighbors[i, cnt] = row * N + (col - 2)
            spring_rest_len[i, cnt] = 0.2
            spring_type[i, cnt] = 2
            cnt += 1
        if row - 2 >= 0:
            spring_neighbors[i, cnt] = (row - 2) * N + col
            spring_rest_len[i, cnt] = 0.2
            spring_type[i, cnt] = 2
            cnt += 1

        for j in range(cnt, 12):
            spring_neighbors[i, j] = -1
```

### 3.3 弹簧力计算（分类型刚度）
```python
@ti.func
def get_spring_stiffness(type):
    if type == 0: return k_struct
    elif type == 1: return k_shear
    elif type == 2: return k_bend
    return 0.0

@ti.func
def compute_forces_on(i: ti.i32):
    force = gravity * mass - k_d * v[i]
    for j in range(12):
        n = spring_neighbors[i, j]
        if n == -1: continue
        diff = x[i] - x[n]
        dist = diff.norm()
        if dist > 1e-6:
            rest = spring_rest_len[i, j]
            k = get_spring_stiffness(spring_type[i, j])
            force += -k * (dist - rest) * diff.normalized()
    return force
```

### 3.4 布料-球体碰撞处理
```python
@ti.kernel
def collide_with_sphere():
    for i in range(N * N):
        if is_fixed[i]: continue
        to_center = x[i] - sphere_center
        dist = to_center.norm()
        if dist < sphere_radius:
            normal = to_center / dist
            # 位置修正
            x[i] = sphere_center + normal * sphere_radius
            # 速度反弹
            vn = v[i].dot(normal)
            if vn < 0:
                v[i] -= (1 + restitution) * vn * normal
```

## 四、实验结果与分析
### 4.1 三种弹簧作用效果对比表
| 弹簧类型 | 核心作用 | 关闭后布料表现 |
| ---- | ---- | ---- |
| 结构弹簧 | 抗拉伸、维持网格结构 | 网格解体、布料断裂 |
| 剪切弹簧 | 抗剪切畸变、保持方形 | 网格严重菱形扭曲、失去真实感 |
| 弯曲弹簧 | 抗过度折叠、提供刚性 | 布料过度褶皱、无厚度、类似软纸 |

### 4.2 碰撞效果分析
1. 布料下落至球体位置时，质点自动检测碰撞并完成修正；
2. 布料**自然覆盖在球体表面**，无穿透、无异常拉伸；
3. 碰撞位置产生合理褶皱，符合真实织物物理特性；
4. 球体采用线框渲染，不遮挡布料网格，可视化效果清晰。

### 4.3 积分器在拓展模型下的表现
1. **显式欧拉**：极易数值爆炸，仅极小步长可短暂运行；
2. **半隐式欧拉**：稳定性优秀，仿真流畅，布料运动自然；
3. **隐式欧拉**：最高稳定性，强阻尼，无抖动，适合高精度仿真。

## 五、实验难点与解决方案
1. **弹簧拓扑冲突**
    问题：三种弹簧同时初始化易出现索引越界、邻居重复。
    解决方案：分阶段初始化，独立计数，无效位统一赋值-1。
2. **碰撞穿透/抖动**
    问题：质点高速运动时穿透球体或反复弹跳。
    解决方案：先执行位置修正，再进行速度法向衰减，限制恢复系数。
3. **渲染遮挡**
    问题：实心球体遮挡布料线框，视觉异常。
    解决方案：使用球体线框渲染，保证布料网格完全可见。

## 六、拓展实验总结
1. 成功实现**结构+剪切+弯曲**完整质点-弹簧模型，布料仿真效果大幅接近真实织物；
2. 完成布料与球体的**空间碰撞检测与响应**，实现无穿透、自然覆盖的碰撞效果；
3. 拓展系统兼容原有三种积分器，可直观对比不同物理条件下的仿真差异；
4. 严格遵循GPU并行编程规范，保持`ti.kernel`初始化、`ti.func`计算的架构要求；
5. 实验验证了弹簧模型对布料形态的决定性作用，以及碰撞算法在物理仿真中的必要性。
