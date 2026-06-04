import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# ====================== 物理参数（来自你的布料代码） ======================
N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# ====================== 球体参数（来自你的碰撞代码） ======================
sphere_center = ti.Vector([0.0, 0.0, 0.0])
sphere_radius = 0.4
restitution = 0.3

# ====================== 布料场 ======================
x = ti.Vector.field(3, float, N * N)
v = ti.Vector.field(3, float, N * N)
f = ti.Vector.field(3, float, N * N)
is_fixed = ti.field(int, N * N)

x_next = ti.Vector.field(3, float, N * N)
v_next = ti.Vector.field(3, float, N * N)
f_next = ti.Vector.field(3, float, N * N)

max_springs = N * N * 4
spring_indices = ti.field(int, max_springs * 2)
spring_pairs = ti.Vector.field(2, int, max_springs)
spring_lengths = ti.field(float, max_springs)
num_springs = ti.field(int, ())

# ====================== 球体线框 ======================
sphere_vertices = None
sphere_line_indices = None

# ====================== 初始化（完全来自你的布料代码） ======================
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_r = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_r])
            spring_lengths[c] = (x[idx] - x[idx_r]).norm()
        if j < N - 1:
            idx_d = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_d])
            spring_lengths[c] = (x[idx] - x[idx_d]).norm()

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

# ====================== 球体网格生成（来自你的碰撞代码） ======================
def init_sphere_wireframe():
    global sphere_vertices, sphere_line_indices
    stacks = 24
    slices = 24
    vertices = []
    indices = []
    for i in range(stacks + 1):
        theta = i * math.pi / stacks
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for j in range(slices + 1):
            phi = j * 2 * math.pi / slices
            px = sphere_center[0] + sphere_radius * sin_theta * math.cos(phi)
            py = sphere_center[1] + sphere_radius * cos_theta
            pz = sphere_center[2] + sphere_radius * sin_theta * math.sin(phi)
            vertices.append([px, py, pz])
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
    flat = []
    for p in indices:
        flat.append(p[0])
        flat.append(p[1])
    sphere_vertices = ti.Vector.field(3, float, len(vertices))
    sphere_line_indices = ti.field(int, len(flat))
    sphere_vertices.from_numpy(np.array(vertices, dtype=np.float32))
    sphere_line_indices.from_numpy(np.array(flat, dtype=np.int32))

# ====================== 力计算（来自你的布料代码） ======================
@ti.func
def compute_forces_on(pos, vel, force):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        a = spring_pairs[i][0]
        b = spring_pairs[i][1]
        d = pos[a] - pos[b]
        dist = d.norm()
        if dist > 1e-6:
            dn = d / dist
            fs = -k_s * (dist - spring_lengths[i]) * dn
            ti.atomic_add(force[a], fs)
            ti.atomic_add(force[b], -fs)

@ti.func
def clamp_velocity(vel, idx):
    nrm = vel[idx].norm()
    if nrm > max_velocity:
        vel[idx] = vel[idx] / nrm * max_velocity

# ====================== 碰撞 kernel（来自你的碰撞代码） ======================
@ti.kernel
def collide_with_sphere():
    for i in range(N * N):
        if is_fixed[i]: continue
        to = x[i] - sphere_center
        dist = to.norm()
        if dist < sphere_radius:
            normal = to / dist
            x[i] = sphere_center + normal * sphere_radius
            vn = v[i].dot(normal)
            if vn < 0:
                v[i] -= (1 + restitution) * vn * normal

# ====================== 积分器（来自你的布料代码） ======================
@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if not is_fixed[i]:
            x[i] += v[i] * dt
            v[i] += f[i] / mass * dt
            clamp_velocity(v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if not is_fixed[i]:
            v[i] += f[i] / mass * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if not is_fixed[i]:
                v_next[i] = v[i] + f_next[i] / mass * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ====================== 主程序 ======================
def main():
    init_cloth()
    init_sphere_wireframe()

    window = ti.ui.Window("Mass Spring + Sphere Collision", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    method = 1
    paused = False

    while window.running:
        window.GUI.begin("Control", 0.02,0.02,0.38,0.36)
        window.GUI.text("Method:")
        p0 = "[*] " if method==0 else "[ ] "
        p1 = "[*] " if method==1 else "[ ] "
        p2 = "[*] " if method==2 else "[ ] "
        if window.GUI.button(p0+"Explicit"):
            method=0; init_cloth()
        if window.GUI.button(p1+"Semi-Implicit"):
            method=1; init_cloth()
        if window.GUI.button(p2+"Implicit"):
            method=2; init_cloth()

        window.GUI.text("")
        if window.GUI.button("Pause" if not paused else "Resume"):
            paused = not paused
        if window.GUI.button("Reset"):
            init_cloth()
        window.GUI.end()

        if not paused:
            for _ in range(40):
                if method ==0: step_explicit()
                elif method ==1: step_semi_implicit()
                else: step_implicit_iter()
                collide_with_sphere()

        scene.set_camera(camera)
        scene.ambient_light((0.5,0.5,0.5))
        scene.point_light((0.5,1.5,1.5),(1,1,1))

        scene.particles(x, 0.015, (0.2,0.6,1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8,0.8,0.8))
        scene.lines(sphere_vertices, indices=sphere_line_indices, width=2.0, color=(1.0,0.2,0.2))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()