import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# ====================== 全局参数 ======================
N = 20
mass = 1.0
dt = 5e-4

k_struct = 12000.0
k_shear = 6000.0
k_bend = 1200.0

k_d = 0.8
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 40.0
max_force = 5000.0

# 球体参数（更小更紧密）
sphere_center = ti.Vector([1.0, 0.6, 1.0])   # 稍微抬高一点，让布料落下时更明显
sphere_radius = 0.5
restitution = 0.3

x = ti.Vector.field(3, float, N * N)
v = ti.Vector.field(3, float, N * N)
is_fixed = ti.field(int, N * N)

spring_neighbors = ti.field(int, (N * N, 12))
spring_rest_len = ti.field(float, (N * N, 12))
spring_type = ti.field(int, (N * N, 12))

x_next = ti.Vector.field(3, float, N * N)
v_next = ti.Vector.field(3, float, N * N)

line_indices = ti.field(int, 2 * (N * (N - 1) + (N - 1) * N))

solver_type = ti.field(int, ())
paused = ti.field(int, ())

# 球体线框数据
sphere_vertices = None
sphere_line_indices = None

# ====================== 初始化 ======================
@ti.kernel
def init_control():
    solver_type[None] = 1
    paused[None] = 0

@ti.kernel
def init_particle_pos():
    for i in range(N * N):
        row = i // N
        col = i % N
        x[i] = ti.Vector([col * 0.1, 2.0, row * 0.1])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        is_fixed[i] = 0
        if (row == 0 and col == 0) or (row == 0 and col == N - 1):
            is_fixed[i] = 1

@ti.kernel
def init_spring():
    for i in range(N * N):
        row = i // N
        col = i % N
        cnt = 0
        # 结构弹簧
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
        # 剪切弹簧
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
        # 弯曲弹簧
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

@ti.kernel
def init_line():
    idx = 0
    for i in range(N):
        for j in range(N - 1):
            line_indices[idx] = i * N + j
            line_indices[idx + 1] = i * N + j + 1
            idx += 2
    for j in range(N):
        for i in range(N - 1):
            line_indices[idx] = i * N + j
            line_indices[idx + 1] = (i + 1) * N + j
            idx += 2

def init_sphere_wireframe():
    """生成球体线框（动态创建 field），更密更醒目"""
    global sphere_vertices, sphere_line_indices
    stacks = 24          # 增加网格密度，让球体看起来更紧密
    slices = 24
    vertices = []
    indices = []
    for i in range(stacks + 1):
        theta = i * math.pi / stacks
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for j in range(slices + 1):
            phi = j * 2 * math.pi / slices
            x = sphere_center[0] + sphere_radius * sin_theta * math.cos(phi)
            y = sphere_center[1] + sphere_radius * cos_theta
            z = sphere_center[2] + sphere_radius * sin_theta * math.sin(phi)
            vertices.append([x, y, z])
    for i in range(stacks + 1):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = i * (slices + 1) + j + 1
            indices.append([a, b])
    for j in range(slices + 1):
        for i in range(stacks):
            a = i * (slices + 1) + j
            b = (i + 1) * (slices + 1) + j
            indices.append([a, b])
    flat_indices = []
    for pair in indices:
        flat_indices.append(pair[0])
        flat_indices.append(pair[1])
    num_verts = len(vertices)
    num_indices = len(flat_indices)
    sphere_vertices = ti.Vector.field(3, float, shape=num_verts)
    sphere_line_indices = ti.field(int, shape=num_indices)
    sphere_vertices.from_numpy(np.array(vertices, dtype=np.float32))
    sphere_line_indices.from_numpy(np.array(flat_indices, dtype=np.int32))
    print(f"Sphere wireframe: {num_verts} vertices, {num_indices//2} line segments")

# ====================== 力学计算 ======================
@ti.func
def get_spring_stiffness(type):
    k = 0.0
    if type == 0:
        k = k_struct
    elif type == 1:
        k = k_shear
    elif type == 2:
        k = k_bend
    return k

@ti.func
def clamp_velocity(vel):
    spd = vel.norm()
    result = vel
    if spd > max_velocity:
        result = vel / spd * max_velocity
    return result

@ti.func
def compute_forces_on(i: ti.i32):
    force = gravity * mass - k_d * v[i]
    for j in range(12):
        n = spring_neighbors[i, j]
        if n == -1:
            continue
        diff = x[i] - x[n]
        dist = diff.norm()
        if dist > 1e-6:
            rest = spring_rest_len[i, j]
            k = get_spring_stiffness(spring_type[i, j])
            f_spring = -k * (dist - rest) * diff.normalized()
            if f_spring.norm() > max_force:
                f_spring = f_spring.normalized() * max_force
            force += f_spring
    return force

# ====================== 球体碰撞 ======================
@ti.kernel
def collide_with_sphere():
    for i in range(N * N):
        if is_fixed[i]:
            continue
        to_center = x[i] - sphere_center
        dist = to_center.norm()
        if dist < sphere_radius:
            normal = to_center / dist
            x[i] = sphere_center + normal * sphere_radius
            vn = v[i].dot(normal)
            if vn < 0:
                v[i] -= (1 + restitution) * vn * normal
    for i in range(N * N):
        if is_fixed[i]:
            continue
        to_center = x[i] - sphere_center
        dist = to_center.norm()
        if dist < sphere_radius:
            normal = to_center / dist
            x[i] = sphere_center + normal * sphere_radius

# ====================== 三种积分器 ======================
@ti.kernel
def step_explicit(dt_cur: float):
    for i in range(N * N):
        if is_fixed[i]:
            continue
        f = compute_forces_on(i)
        a = f / mass
        v[i] += a * dt_cur
        v[i] = clamp_velocity(v[i])
        x[i] += v[i] * dt_cur

@ti.kernel
def step_semi_implicit(dt_cur: float):
    for i in range(N * N):
        if is_fixed[i]:
            continue
        f = compute_forces_on(i)
        a = f / mass
        v[i] += a * dt_cur
        v[i] = clamp_velocity(v[i])
        x[i] += v[i] * dt_cur

@ti.kernel
def step_implicit_iter(dt_cur: float):
    for i in range(N * N):
        x_next[i] = x[i]
        v_next[i] = v[i]
    for _ in range(8):
        for i in range(N * N):
            if is_fixed[i]:
                continue
            f = gravity * mass - k_d * v_next[i]
            for j in range(12):
                n = spring_neighbors[i, j]
                if n == -1:
                    continue
                diff = x_next[i] - x_next[n]
                dist = diff.norm()
                if dist > 1e-6:
                    rest = spring_rest_len[i, j]
                    k = get_spring_stiffness(spring_type[i, j])
                    f_spring = -k * (dist - rest) * diff.normalized()
                    if f_spring.norm() > max_force:
                        f_spring = f_spring.normalized() * max_force
                    f += f_spring
            a = f / mass
            v_next[i] = v[i] + a * dt_cur
            v_next[i] = clamp_velocity(v_next[i])
            x_next[i] = x[i] + v_next[i] * dt_cur
    for i in range(N * N):
        if not is_fixed[i]:
            x[i] = x_next[i]
            v[i] = v_next[i]

# ====================== 渲染 ======================
@ti.kernel
def update_render_vertices(verts: ti.template()):
    for i in range(N * N):
        verts[i] = x[i]

def reset_cloth():
    init_particle_pos()
    print("Cloth reset!")

# ====================== 主程序 ======================
if __name__ == "__main__":
    init_control()
    init_particle_pos()
    init_spring()
    init_line()
    init_sphere_wireframe()

    window = ti.ui.Window("Mass-Spring Cloth - Visible Ball", (1024, 768))
    canvas = window.get_canvas()
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(2.5, 1.5, 4.0)
    camera.lookat(1.0, 0.8, 1.0)

    render_verts = ti.Vector.field(3, float, N * N)

    explicit_dt = 1e-4
    default_dt = dt

    while window.running:
        with gui.sub_window("Control Panel", 0.04, 0.05, 0.28, 0.35):
            gui.text("Integration Method:")
            if gui.button("Explicit Euler (Unstable)"):
                solver_type[None] = 0
                print("⚠️ Explicit Euler is unstable. Use Semi-Implicit or Implicit.")
            if gui.button("Semi-Implicit Euler (Stable)"):
                solver_type[None] = 1
            if gui.button("Implicit Euler (Damped)"):
                solver_type[None] = 2

            gui.text("")
            if gui.button("Pause Simulation"):
                paused[None] = 1 - paused[None]
            if gui.button("Reset Cloth"):
                reset_cloth()

            gui.text("")
            gui.text(f"Spring Stiffness:")
            gui.text(f"  Structural: {k_struct:.0f}")
            gui.text(f"  Shear:      {k_shear:.0f}")
            gui.text(f"  Bending:    {k_bend:.0f}")

        cur_dt = explicit_dt if solver_type[None] == 0 else default_dt

        if paused[None] == 0:
            if solver_type[None] == 0:
                step_explicit(cur_dt)
            elif solver_type[None] == 1:
                step_semi_implicit(cur_dt)
            else:
                step_implicit_iter(cur_dt)

            collide_with_sphere()

            x_np = x.to_numpy()
            if np.any(np.abs(x_np) > 1e6):
                reset_cloth()

        update_render_vertices(render_verts)

        scene.set_camera(camera)
        scene.ambient_light((0.2, 0.2, 0.2))
        scene.point_light((3.0, 5.0, 3.0), (1.0, 1.0, 1.0))

        # 球体线框（更粗、更亮、更密）
        if sphere_vertices is not None and sphere_line_indices is not None:
            scene.lines(sphere_vertices, indices=sphere_line_indices, width=0.015, color=(1.0, 0.2, 0.2))
        # 布料线框和质点
        scene.lines(render_verts, indices=line_indices, width=0.01, color=(1.0, 1.0, 1.0))
        scene.particles(render_verts, radius=0.025, color=(0.2, 0.6, 1.0))

        canvas.scene(scene)
        window.show()