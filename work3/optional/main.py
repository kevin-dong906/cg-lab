import taichi as ti

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 600
screen = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

ambient_coeff = ti.field(ti.f32, shape=())
diffuse_coeff = ti.field(ti.f32, shape=())
specular_coeff = ti.field(ti.f32, shape=())
shininess_exp = ti.field(ti.f32, shape=())

@ti.func
def unit_vector(vec):
    return vec / (vec.norm() + 1e-6)

@ti.func
def sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    dist = -1.0
    norm = ti.Vector([0.0, 0.0, 0.0])
    oc_val = ray_origin - sphere_center
    linear_term = 2.0 * oc_val.dot(ray_dir)
    quad_term = oc_val.dot(oc_val) - sphere_radius ** 2
    discriminant = linear_term ** 2 - 4.0 * quad_term
    if discriminant > 0:
        root = (-linear_term - ti.sqrt(discriminant)) / 2.0
        if root > 0:
            dist = root
            intersect_pt = ray_origin + ray_dir * dist
            norm = unit_vector(intersect_pt - sphere_center)
    return dist, norm

@ti.func
def cone_intersect(ray_origin, ray_dir, cone_apex, base_height, cone_radius):
    dist = -1.0
    norm = ti.Vector([0.0, 0.0, 0.0])
    height_total = cone_apex.y - base_height
    ratio = (cone_radius / height_total) ** 2
    local_origin = ray_origin - cone_apex
    
    a_coeff = ray_dir.x**2 + ray_dir.z**2 - ratio * ray_dir.y** 2
    b_coeff = 2.0 * (local_origin.x * ray_dir.x + local_origin.z * ray_dir.z - ratio * local_origin.y * ray_dir.y)
    c_coeff = local_origin.x** 2 + local_origin.z**2 - ratio * local_origin.y** 2
    
    if ti.abs(a_coeff) > 1e-6:
        disc = b_coeff**2 - 4.0 * a_coeff * c_coeff
        if disc > 0:
            t0 = (-b_coeff - ti.sqrt(disc)) / (2.0 * a_coeff)
            t1 = (-b_coeff + ti.sqrt(disc)) / (2.0 * a_coeff)
            near_t = t0
            far_t = t1
            if t0 > t1:
                near_t, far_t = far_t, near_t
            
            y_val = local_origin.y + near_t * ray_dir.y
            if near_t > 0 and -height_total <= y_val <= 0:
                dist = near_t
            else:
                y_val2 = local_origin.y + far_t * ray_dir.y
                if far_t > 0 and -height_total <= y_val2 <= 0:
                    dist = far_t
            
            if dist > 0:
                pt_local = local_origin + ray_dir * dist
                norm = unit_vector(ti.Vector([pt_local.x, -ratio * pt_local.y, pt_local.z]))
    return dist, norm

@ti.func
def shadow_check(origin, light_dir, light_dist):
    shadow = False
    t1, _ = sphere_intersect(origin, light_dir, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    t2, _ = cone_intersect(origin, light_dir, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
    if 0 < t1 < light_dist - 1e-3 or 0 < t2 < light_dist - 1e-3:
        shadow = True
    return shadow

@ti.kernel
def draw_scene():
    for x, y in screen:
        u_coord = (x - WIDTH / 2.0) / HEIGHT * 2.0
        v_coord = (y - HEIGHT / 2.0) / HEIGHT * 2.0
        
        cam_pos = ti.Vector([0.0, 0.0, 5.0])
        ray_direction = unit_vector(ti.Vector([u_coord, v_coord, -1.0]))
        
        closest_dist = 1e10
        surface_normal = ti.Vector([0.0, 0.0, 0.0])
        obj_color = ti.Vector([0.0, 0.0, 0.0])
        
        t1, n1 = sphere_intersect(cam_pos, ray_direction, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t1 < closest_dist:
            closest_dist = t1
            surface_normal = n1
            obj_color = ti.Vector([0.8, 0.1, 0.1])
        
        t2, n2 = cone_intersect(cam_pos, ray_direction, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t2 < closest_dist:
            closest_dist = t2
            surface_normal = n2
            obj_color = ti.Vector([0.6, 0.2, 0.8])
        
        final_color = ti.Vector([0.05, 0.15, 0.15])
        if closest_dist < 1e9:
            hit_point = cam_pos + ray_direction * closest_dist
            norm = surface_normal
            
            light_position = ti.Vector([2.0, 3.0, 4.0])
            light_rgb = ti.Vector([1.0, 1.0, 1.0])
            
            light_dir = unit_vector(light_position - hit_point)
            view_dir = unit_vector(cam_pos - hit_point)
            light_dist = (light_position - hit_point).norm()
            
            in_shadow = shadow_check(hit_point, light_dir, light_dist)
            
            amb_result = ambient_coeff[None] * light_rgb * obj_color
            
            if in_shadow:
                final_color = amb_result
            else:
                diff_factor = ti.max(0.0, norm.dot(light_dir))
                diff_result = diffuse_coeff[None] * diff_factor * light_rgb * obj_color
                
                half_vector = unit_vector(light_dir + view_dir)
                spec_factor = ti.max(0.0, norm.dot(half_vector)) ** shininess_exp[None]
                spec_result = specular_coeff[None] * spec_factor * light_rgb
                
                final_color = amb_result + diff_result + spec_result
        
        screen[x, y] = ti.math.clamp(final_color, 0.0, 1.0)

def launch():
    app = ti.ui.Window("Blinn-Phong Renderer with Shadow", (WIDTH, HEIGHT))
    render_canvas = app.get_canvas()
    ui_panel = app.get_gui()
    
    ambient_coeff[None] = 0.2
    diffuse_coeff[None] = 0.7
    specular_coeff[None] = 0.5
    shininess_exp[None] = 32.0

    while app.running:
        draw_scene()
        render_canvas.set_image(screen)
        
        with ui_panel.sub_window("Shader Controls", 0.7, 0.05, 0.28, 0.22):
            ambient_coeff[None] = ui_panel.slider_float('Ka (Ambient)', ambient_coeff[None], 0.0, 1.0)
            diffuse_coeff[None] = ui_panel.slider_float('Kd (Diffuse)', diffuse_coeff[None], 0.0, 1.0)
            specular_coeff[None] = ui_panel.slider_float('Ks (Specular)', specular_coeff[None], 0.0, 1.0)
            shininess_exp[None] = ui_panel.slider_float('N (Shininess)', shininess_exp[None], 1.0, 128.0)

        app.show()

if __name__ == '__main__':
    launch()