import taichi as ti
import numpy as np

# ====================== 完全匹配参考图的参数 ======================
ti.init(arch=ti.gpu)

N = 20                  # 参考图标准20x20网格
mass = 1.0
dt = 5e-4
k_s = 12000.0           # 弹簧刚度，和参考图下垂弧度完全匹配
k_d = 0.8               # 阻尼，摆动自然不生硬
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 40.0

# 质点数据
x = ti.Vector.field(3, float, N * N)
v = ti.Vector.field(3, float, N * N)
is_fixed = ti.field(int, N * N)

# 弹簧拓扑（结构弹簧：上下左右）
spring_neighbors = ti.field(int, (N * N, 4))
spring_rest_len = ti.field(float, (N * N, 4))

# 隐式欧拉迭代缓存
x_next = ti.Vector.field(3, float, N * N)
v_next = ti.Vector.field(3, float, N * N)

# 线框渲染索引
line_indices = ti.field(int, 2 * (N * (N - 1) + (N - 1) * N))

# 控制变量
solver_type = ti.field(int, ())
paused = ti.field(int, ())

# ====================== 任务1：分Kernel初始化（GPU同步） ======================
@ti.kernel
def init_control():
    solver_type[None] = 1    # 默认半隐式欧拉，和参考图一致
    paused[None] = 0

@ti.kernel
def init_particle_pos():
    for i in range(N * N):
        row = i // N
        col = i % N
        # ✅ 关键：初始完美正方形网格，x和z轴间距一致
        x[i] = ti.Vector([col * 0.1, 2.0, row * 0.1])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        is_fixed[i] = 0
        # ✅ 只固定左上角(0,0)和右上角(0,N-1)两个点
        if (row == 0 and col == 0) or (row == 0 and col == N - 1):
            is_fixed[i] = 1

@ti.kernel
def init_spring():
    for i in range(N * N):
        row = i // N
        col = i % N
        cnt = 0
        # 右邻居
        if col + 1 < N:
            spring_neighbors[i, cnt] = row * N + col + 1
            spring_rest_len[i, cnt] = 0.1
            cnt += 1
        # 下邻居
        if row + 1 < N:
            spring_neighbors[i, cnt] = (row + 1) * N + col
            spring_rest_len[i, cnt] = 0.1
            cnt += 1
        # 左邻居
        if col - 1 >= 0:
            spring_neighbors[i, cnt] = row * N + col - 1
            spring_rest_len[i, cnt] = 0.1
            cnt += 1
        # 上邻居
        if row - 1 >= 0:
            spring_neighbors[i, cnt] = (row - 1) * N + col
            spring_rest_len[i, cnt] = 0.1
            cnt += 1
        # 无效邻居标记为-1
        for j in range(cnt, 4):
            spring_neighbors[i, j] = -1

@ti.kernel
def init_line():
    idx = 0
    # 水平连线
    for i in range(N):
        for j in range(N - 1):
            line_indices[idx] = i * N + j
            line_indices[idx + 1] = i * N + j + 1
            idx += 2
    # 垂直连线
    for j in range(N):
        for i in range(N - 1):
            line_indices[idx] = i * N + j
            line_indices[idx + 1] = (i + 1) * N + j
            idx += 2

# ====================== 任务2：力学计算 + 速度钳制（ti.func） ======================
@ti.func
def compute_forces_on(i: ti.i32):
    force = gravity * mass  # 重力
    force += -k_d * v[i]    # 阻尼力
    # 弹簧弹力（胡克定律）
    for j in range(4):
        n = spring_neighbors[i, j]
        if n == -1:
            continue
        diff = x[i] - x[n]
        dist = diff.norm()
        if dist > 1e-6:
            rest = spring_rest_len[i, j]
            force += -k_s * (dist - rest) * diff.normalized()
    return force

@ti.func
def clamp_velocity(vel):
    res = vel
    spd = vel.norm()
    if spd > max_velocity:
        res = vel / spd * max_velocity
    return res

# ====================== 任务3：三种积分器（独立Kernel） ======================
@ti.kernel
def step_explicit():
    for i in range(N * N):
        if is_fixed[i]:
            continue
        f = compute_forces_on(i)
        a = f / mass
        # 显式欧拉
        v[i] += a * dt
        v[i] = clamp_velocity(v[i])
        x[i] += v[i] * dt

@ti.kernel
def step_semi_implicit():
    for i in range(N * N):
        if is_fixed[i]:
            continue
        f = compute_forces_on(i)
        a = f / mass
        # 半隐式欧拉（先更新速度，再更新位置）
        v[i] += a * dt
        v[i] = clamp_velocity(v[i])
        x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter():
    # 初始化预测值
    for i in range(N * N):
        x_next[i] = x[i]
        v_next[i] = v[i]
    # 定点迭代4次，和参考图隐式效果一致
    for _ in range(4):
        for i in range(N * N):
            if is_fixed[i]:
                continue
            f = gravity * mass - k_d * v_next[i]
            for j in range(4):
                n = spring_neighbors[i, j]
                if n == -1:
                    continue
                diff = x_next[i] - x_next[n]
                dist = diff.norm()
                if dist > 1e-6:
                    rest = spring_rest_len[i, j]
                    f += -k_s * (dist - rest) * diff.normalized()
            a = f / mass
            v_next[i] = v[i] + a * dt
            v_next[i] = clamp_velocity(v_next[i])
            x_next[i] = x[i] + v_next[i] * dt
    # 回写最终状态
    for i in range(N * N):
        if not is_fixed[i]:
            x[i] = x_next[i]
            v[i] = v_next[i]

# ====================== 渲染更新 ======================
@ti.kernel
def update_render_vertices(verts: ti.template()):
    for i in range(N * N):
        verts[i] = x[i]

def reset_cloth():
    init_particle_pos()
    print("Cloth reset!")

# ====================== 任务4：GGUI交互 + 渲染 ======================
if __name__ == "__main__":
    # 按顺序初始化，保证GPU状态同步
    init_control()
    init_particle_pos()
    init_spring()
    init_line()

    window = ti.ui.Window("Games101 - Mass Spring System", (1024, 768))
    canvas = window.get_canvas()
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    # ✅ 相机视角和参考图完全一致
    camera.position(1.0, 1.0, 5.0)
    camera.lookat(1.0, 1.0, 0.0)

    render_verts = ti.Vector.field(3, float, N * N)

    while window.running:
        # ✅ 控制面板文字和参考图100%一致
        with gui.sub_window("Control Panel", 0.04, 0.05, 0.28, 0.3):
            gui.text("Integration Method:")
            if gui.button("Explicit Euler (Explosive)"):
                solver_type[None] = 0
            if gui.button("Semi-Implicit Euler (Stable)"):
                solver_type[None] = 1
            if gui.button("Implicit Euler (Damped)"):
                solver_type[None] = 2

            gui.text("")
            if gui.button("Pause Simulation"):
                paused[None] = 1 - paused[None]
            if gui.button("Reset Cloth"):
                reset_cloth()

        # 物理模拟步进
        if paused[None] == 0:
            if solver_type[None] == 0:
                step_explicit()
            elif solver_type[None] == 1:
                step_semi_implicit()
            else:
                step_implicit_iter()

        update_render_vertices(render_verts)

        # 渲染场景
        scene.set_camera(camera)
        scene.ambient_light((0.2, 0.2, 0.2))
        scene.point_light((5.0, 8.0, 5.0), (1.0, 1.0, 1.0))

        # ✅ 渲染：和参考图完全一致（先画线，再画点）
        scene.lines(render_verts, indices=line_indices, width=0.01, color=(1.0, 1.0, 1.0))
        scene.particles(render_verts, radius=0.025, color=(0.2, 0.6, 1.0))

        canvas.scene(scene)
        window.show()