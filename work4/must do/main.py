import taichi as ti

ti.init(arch=ti.gpu)

res_width, res_height = 800, 600
screen_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_width, res_height))

light_x = ti.field(ti.f32, shape=())
light_y = ti.field(ti.f32, shape=())
light_z = ti.field(ti.f32, shape=())
trace_bounces = ti.field(ti.i32, shape=())

DIFFUSE = 0
MIRROR = 1

@ti.func
def unit_vector(v):
    return v / (v.norm() + 1e-6)

@ti.func
def reflect_vector(incident, normal):
    return incident - 2.0 * incident.dot(normal) * normal

@ti.func
def sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    t_result = -1.0
    norm_result = ti.Vector([0.0, 0.0, 0.0])
    oc = ray_origin - sphere_center
    b_coeff = 2.0 * oc.dot(ray_dir)
    c_coeff = oc.dot(oc) - sphere_radius ** 2
    discriminant = b_coeff ** 2 - 4.0 * c_coeff
    
    if discriminant > 0:
        t_val = (-b_coeff - ti.sqrt(discriminant)) / 2.0
        if t_val > 0:
            t_result = t_val
            hit_point = ray_origin + ray_dir * t_val
            norm_result = unit_vector(hit_point - sphere_center)
    return t_result, norm_result

@ti.func
def plane_intersect(ray_origin, ray_dir, plane_height):
    t_result = -1.0
    norm_result = ti.Vector([0.0, 1.0, 0.0])
    if abs(ray_dir.y) > 1e-6:
        t_val = (plane_height - ray_origin.y) / ray_dir.y
        if t_val > 0:
            t_result = t_val
    return t_result, norm_result

@ti.func
def trace_scene(ray_origin, ray_dir):
    closest_t = 1e10
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_color = ti.Vector([0.0, 0.0, 0.0])
    hit_material = DIFFUSE

    t, n = sphere_intersect(ray_origin, ray_dir, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
    if 0 < t < closest_t:
        closest_t = t
        hit_normal = n
        hit_color = ti.Vector([0.9, 0.2, 0.2])
        hit_material = DIFFUSE

    t, n = sphere_intersect(ray_origin, ray_dir, ti.Vector([1.5, 0.0, 0.0]), 1.0)
    if 0 < t < closest_t:
        closest_t = t
        hit_normal = n
        hit_color = ti.Vector([0.95, 0.95, 0.95])
        hit_material = MIRROR

    t, n = plane_intersect(ray_origin, ray_dir, -1.0)
    if 0 < t < closest_t:
        closest_t = t
        hit_normal = n
        hit_material = DIFFUSE
        p = ray_origin + ray_dir * t
        scale = 2.0
        x_cell = ti.floor(p.x * scale)
        z_cell = ti.floor(p.z * scale)
        if (x_cell + z_cell) % 2 == 0:
            hit_color = ti.Vector([0.25, 0.25, 0.25])
        else:
            hit_color = ti.Vector([0.85, 0.85, 0.85])

    return closest_t, hit_normal, hit_color, hit_material

@ti.kernel
def render_frame():
    light_position = ti.Vector([light_x[None], light_y[None], light_z[None]])
    background = ti.Vector([0.03, 0.12, 0.18])

    for x, y in screen_pixels:
        u = (x - res_width / 2.0) / res_height * 2.0
        v = (y - res_height / 2.0) / res_height * 2.0
        
        cam_origin = ti.Vector([0.0, 1.0, 5.0])
        cam_dir = unit_vector(ti.Vector([u, v - 0.2, -1.0]))

        color_accum = ti.Vector([0.0, 0.0, 0.0])
        energy = ti.Vector([1.0, 1.0, 1.0])

        for _ in range(trace_bounces[None]):
            t, normal, obj_color, mat = trace_scene(cam_origin, cam_dir)
            
            if t > 1e9:
                color_accum += energy * background
                break
                
            point = cam_origin + cam_dir * t
            
            if mat == MIRROR:
                cam_origin = point + normal * 1e-4
                cam_dir = unit_vector(reflect_vector(cam_dir, normal))
                energy *= 0.8 * obj_color
                
            elif mat == DIFFUSE:
                light_dir = unit_vector(light_position - point)
                
                shadow_origin = point + normal * 1e-4
                shadow_t, _, _, _ = trace_scene(shadow_origin, light_dir)
                
                light_dist = (light_position - point).norm()
                shadowed = 0.0
                if shadow_t < light_dist:
                    shadowed = 1.0
                
                ambient = 0.2 * obj_color
                direct = ambient
                
                if shadowed == 0.0:
                    diff = ti.max(0.0, normal.dot(light_dir))
                    diffuse = 0.8 * diff * obj_color
                    direct += diffuse
                
                color_accum += energy * direct
                break

        screen_pixels[x, y] = ti.math.clamp(color_accum, 0.0, 1.0)

def main():
    window = ti.ui.Window("Custom Ray Tracer", (res_width, res_height))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_x[None] = 2.5
    light_y[None] = 4.5
    light_z[None] = 3.5
    trace_bounces[None] = 3

    while window.running:
        render_frame()
        canvas.set_image(screen_pixels)
        
        with gui.sub_window("Control Panel", 0.75, 0.05, 0.23, 0.22):
            light_x[None] = gui.slider_float('Light X', light_x[None], -5.0, 5.0)
            light_y[None] = gui.slider_float('Light Y', light_y[None], 1.0, 8.0)
            light_z[None] = gui.slider_float('Light Z', light_z[None], -5.0, 5.0)
            trace_bounces[None] = gui.slider_int('Max Bounces', trace_bounces[None], 1, 5)

        window.show()

if __name__ == '__main__':
    main()