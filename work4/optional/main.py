import taichi as ti
ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 600
pixels = ti.Vector.field(3, dtype=float, shape=(WIDTH, HEIGHT))

light_x = ti.field(ti.f32, shape=())
light_y = ti.field(ti.f32, shape=())
light_z = ti.field(ti.f32, shape=())
bounces = ti.field(ti.i32, shape=())
aa = ti.field(ti.i32, shape=())

DIFFUSE = 0
MIRROR = 1
GLASS = 2

@ti.func
def normalize(v):
    return v.normalized()

@ti.func
def reflect(I, N):
    return I - 2 * I.dot(N) * N

@ti.func
def refract(I, N, ni_over_nt):
    uv = normalize(I)
    dt = uv.dot(N)
    disc = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt*dt)
    refl = ti.Vector([0.0,0.0,0.0])
    ok = False
    if disc > 0:
        refl = ni_over_nt * (uv - N*dt) - N * ti.sqrt(disc)
        ok = True
    return ok, refl

@ti.func
def hit_sphere(ro, rd, center, r):
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - r*r
    d = b*b - 4*a*c
    t = -1.0
    if d > 0:
        t = (-b - ti.sqrt(d)) / (2*a)
    n = normalize(ro + t*rd - center)
    return t, n

@ti.func
def hit_plane(ro, rd, y):
    t = -1.0
    if rd.y != 0:
        t = (y - ro.y) / rd.y
    n = ti.Vector([0,1,0])
    return t, n

@ti.func
def scene(ro, rd):
    t = 1e9
    mat = DIFFUSE
    col = ti.Vector([1.0,1.0,1.0])
    n = ti.Vector([0.0,1.0,0.0])

    t1, n1 = hit_sphere(ro, rd, ti.Vector([-1.5,0,0]), 1.0)
    if t1 > 0 and t1 < t:
        t = t1
        n = n1
        mat = GLASS

    t2, n2 = hit_sphere(ro, rd, ti.Vector([1.5,0,0]), 1.0)
    if t2 > 0 and t2 < t:
        t = t2
        n = n2
        mat = MIRROR
        col = ti.Vector([0.9,0.9,0.9])

    t3, n3 = hit_plane(ro, rd, -1.0)
    if t3 > 0 and t3 < t:
        t = t3
        n = n3
        mat = DIFFUSE
        p = ro + t*rd
        if (int(p.x*2) + int(p.z*2)) % 2 == 0:
            col = ti.Vector([0.25,0.25,0.25])
        else:
            col = ti.Vector([0.85,0.85,0.85])

    return t, n, col, mat

@ti.func
def ray(ro, rd):
    result = ti.Vector([0.0,0.0,0.0])
    atten = ti.Vector([1.0,1.0,1.0])
    light = normalize(ti.Vector([light_x[None], light_y[None], light_z[None]]))

    for _ in range(bounces[None]):
        t, n, col, mat = scene(ro, rd)
        if t > 1e8:
            result += atten * ti.Vector([0.03,0.12,0.18])
            break

        p = ro + rd * t
        ro = p

        if mat == GLASS:
            outward = n
            if rd.dot(n) > 0:
                outward = -n
            ok, refr = refract(rd, outward, 1.0/1.5)
            if ok:
                rd = refr
            else:
                rd = reflect(rd, n)
            atten *= col

        elif mat == MIRROR:
            rd = reflect(rd, n)
            atten *= 0.8 * col

        else:
            shadow = 1.0
            st, _, _, _ = scene(p + n*1e-4, light)
            if st < 1e8:
                shadow = 0.2
            diff = max(n.dot(light), 0.0) * shadow
            result += atten * col * (0.2 + 0.8*diff)
            break

    return result

@ti.kernel
def render():
    for i, j in pixels:
        c = ti.Vector([0.0,0.0,0.0])
        for _ in range(aa[None]):
            u = (i + ti.random() - WIDTH/2) / HEIGHT
            v = (j + ti.random() - HEIGHT/2) / HEIGHT
            ro = ti.Vector([0,1,5])
            rd = normalize(ti.Vector([u, v-0.2, -1]))
            c += ray(ro, rd)
        pixels[i,j] = c / aa[None]

def main():
    light_x[None] = 2.5
    light_y[None] = 4.5
    light_z[None] = 3.5
    bounces[None] = 4
    aa[None] = 4

    render()
    gui = ti.GUI("Glass + AA", res=(WIDTH, HEIGHT))
    while gui.running:
        gui.set_image(pixels)
        gui.show()

if __name__ == "__main__":
    main()