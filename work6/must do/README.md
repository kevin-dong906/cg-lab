# 计算机图形学实验报告
## 布料物理模拟（质点-弹簧系统 + 积分器对比 + 碰撞检测）

**实验环境**：Python 3.13 + Taichi 1.7.4
**实验平台**：Windows + GPU(CUDA)
**核心内容**：质点-弹簧布料仿真、三种数值积分器、结构/剪切/弯曲弹簧、球体碰撞

# 一、实验目标
1. **动态场景渲染**
   掌握 Taichi 框架构建 3D 场景，使用 GGUI 实现交互控制面板。
2. **质点-弹簧模型**
   理解胡克定律、阻尼力、速度钳制，解决数值爆炸问题。
3. **数值积分方法对比**
   独立实现**显式欧拉、半隐式欧拉、隐式欧拉**，观察稳定性差异。
4. **GPU 并行编程**
   掌握 `ti.kernel` / `ti.func`，理解 GPU 同步、Kernel 优化。
5. **高级拓展（选做）**
   实现**剪切弹簧、弯曲弹簧**、**布料-球体碰撞**。

# 二、实验原理
## 2.1 质点-弹簧模型（Mass-Spring Model）
将布料离散为**网格质点**，质点间用弹簧连接，通过受力更新运动状态。

### 受力公式
1. **弹簧力（胡克定律）**
$$f_{a}=-k_s\left(\left|x_a-x_b\right|-l\right)\frac{x_a-x_b}{\left|x_a-x_b\right|}$$
- $k_s$：弹簧刚度
- $l$：弹簧原长
- $x_a,x_b$：质点位置

2. **阻尼力**
$$f_d=-k_d v_a$$

3. **重力**
$$f_g = m \cdot g$$

### 弹簧类型（本实验完整实现）
- **结构弹簧 Structural**：上下左右相邻 → 抗拉伸
- **剪切弹簧 Shear**：对角线相邻 → 抗菱形畸变
- **弯曲弹簧 Bending**：隔一格相邻 → 抗折叠、保持布料硬度

## 2.2 数值积分方法
根据牛顿第二定律：
$$a=F/m$$

在离散时间步 $\Delta t$ 内更新速度与位置：

1. **显式欧拉**
$$x_{t+1}=x_t+v_t \Delta t$$
$$v_{t+1}=v_t+a_t \Delta t$$
特点：计算快、**极不稳定、易爆炸**。

2. **半隐式欧拉（推荐）**
$$v_{t+1}=v_t+a_t \Delta t$$
$$x_{t+1}=x_t+v_{t+1} \Delta t$$
特点：**稳定、高效、适合实时布料**。

3. **隐式欧拉（定点迭代）**
$$v_{t+1}=v_t+a_{t+1}\Delta t$$
$$x_{t+1}=x_t+v_{t+1}\Delta t$$
特点：最稳定、阻尼强、迭代求解、开销大。


# 三、实验环境与架构
## 3.1 开发环境
- 语言：Python 3.13
- 框架：Taichi 1.7.4（GPU 加速）
- 渲染：GGUI 3D 渲染管线

## 3.2 系统架构（严格按实验要求）
1. **多 Kernel 初始化**（保证 GPU 同步）
2. **ti.func 力学计算**（内联优化）
3. **独立积分器 Kernel**
4. **GGUI 交互面板**
5. **碰撞检测与响应**


# 四、核心代码分析（按实验任务）
## 任务1：场景初始化（多 Kernel 保证 GPU 同步）
实验要求：**拆分初始化，避免 GPU 状态不同步**。

```python
@ti.kernel
def init_particle_pos():
    for i in range(N * N):
        row = i // N
        col = i % N
        x[i] = ti.Vector([col * 0.1, 2.0, row * 0.1])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        # 固定两点
        if (row == 0 and col == 0) or (row == 0 and col == N - 1):
            is_fixed[i] = 1
```

```python
@ti.kernel
def init_spring():
    # 结构弹簧（上下左右）
    # 剪切弹簧（对角线）
    # 弯曲弹簧（隔一格）
```

```python
@ti.kernel
def init_line():
    # 生成线框渲染索引
```

---

## 任务2：力学计算 + 防爆（ti.func）
实验要求：`compute_forces_on`、`clamp_velocity` 必须为 `ti.func`。

```python
@ti.func
def compute_forces_on(i: ti.i32):
    force = gravity * mass - k_d * v[i]
    # 遍历12个弹簧（结构+剪切+弯曲）
    for j in range(12):
        n = spring_neighbors[i, j]
        if n == -1: continue
        diff = x[i] - x[n]
        dist = diff.norm()
        if dist > 1e-6:
            rest = spring_rest_len[i, j]
            k = get_spring_stiffness(spring_type[i, j])
            f_spring = -k * (dist - rest) * diff.normalized()
            force += f_spring
    return force

@ti.func
def clamp_velocity(vel):
    spd = vel.norm()
    if spd > max_velocity:
        return vel / spd * max_velocity
    return vel
```


## 任务3：三种积分器（独立 Kernel）
### 3.1 显式欧拉
```python
@ti.kernel
def step_explicit(dt_cur: float):
    for i in range(N * N):
        if is_fixed[i]: continue
        f = compute_forces_on(i)
        a = f / mass
        v[i] += a * dt_cur
        v[i] = clamp_velocity(v[i])
        x[i] += v[i] * dt_cur
```

### 3.2 半隐式欧拉（稳定）
```python
@ti.kernel
def step_semi_implicit(dt_cur: float):
    for i in range(N * N):
        if is_fixed[i]: continue
        f = compute_forces_on(i)
        a = f / mass
        v[i] += a * dt_cur
        v[i] = clamp_velocity(v[i])
        x[i] += v[i] * dt_cur
```

### 3.3 隐式欧拉（定点迭代）
```python
@ti.kernel
def step_implicit_iter(dt_cur: float):
    for _ in range(8):
        for i in range(N * N):
            # 使用下一时刻状态计算受力
            f = gravity * mass - k_d * v_next[i]
            # ... 弹簧力 ...
            a = f / mass
            v_next[i] = v[i] + a * dt_cur
            x_next[i] = x[i] + v_next[i] * dt_cur
    # 回写
    for i in range(N * N):
        x[i] = x_next[i]
        v[i] = v_next[i]
```

## 任务4：碰撞处理（布料 ↔ 球体）
实验要求：判断质点与球体距离，做位置修正 + 速度反弹。

```python
@ti.kernel
def collide_with_sphere():
    for i in range(N * N):
        to_center = x[i] - sphere_center
        dist = to_center.norm()
        if dist < sphere_radius:
            normal = to_center / dist
            # 位置修正：推出球体
            x[i] = sphere_center + normal * sphere_radius
            # 速度反弹
            vn = v[i].dot(normal)
            if vn < 0:
                v[i] -= (1 + restitution) * vn * normal
```

# 五、运行结果与对比
## 5.1 三种积分器效果对比表
| 积分器 | 稳定性 | 速度 | 表现 | 适用场景 |
|---|---|---|---|---|
| 显式欧拉 | 极差（极易爆炸） | 最快 | 抖动、飞散、失真 | 仅教学演示 |
| 半隐式欧拉 | 优秀 | 快 | 自然、稳定、流畅 | **实时布料首选** |
| 隐式欧拉 | 最稳定 | 较慢 | 高阻尼、不抖动、柔软 | 高精度仿真 |

## 5.2 弹簧类型效果对比
| 弹簧类型 | 作用 | 关闭后效果 |
|---|---|---|
| 结构弹簧 | 抗拉伸 | 布料断裂、网格崩解 |
| 剪切弹簧 | 抗剪切畸变 | 网格变菱形、失真 |
| 弯曲弹簧 | 抗折叠、提升硬度 | 过度褶皱、像软纸 |

## 5.3 碰撞效果
- 布料质点**不会穿透球体**
- 自然覆盖、产生褶皱
- 速度反弹符合物理规律
- 线框正确显示，无消失、无遮挡错误

# 六、实验难点与解决方案
## 6.1 数值爆炸（质点飞散）
- 原因：显式欧拉步长过大、弹簧刚度过高
- 解决：**速度钳制**、减小显式步长、使用半隐式

## 6.2 GPU 同步错误
- 原因：初始化在一个 Kernel 中并行冲突
- 解决：**拆分为多个 Kernel 顺序执行**

## 6.3 渲染遮挡（弹簧“消失”）
- 原因：实心球体覆盖线框
- 解决：改为**球体线框渲染**，不遮挡布料

## 6.4 碰撞穿透
- 原因：质点速度过快
- 解决：**位置修正 + 法向反弹**

# 七、实验总结
1. 完整实现**质点-弹簧布料系统**
2. 实现**结构/剪切/弯曲三种弹簧**，布料形态真实自然
3. 实现**三种欧拉积分器**，可实时切换对比稳定性
4. 实现**布料-球体碰撞**，无穿透、无抖动
5. 严格遵循实验规范：
   - 多 Kernel 初始化
   - ti.func 力学计算
   - GGUI 交互面板
   - 速度钳制防爆

本实验完整覆盖计算机图形学**布料仿真、物理引擎、数值积分、GPU 并行**核心知识点。
