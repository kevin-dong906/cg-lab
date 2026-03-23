import taichi as ti
import math

ti.init(arch=ti.cpu)

vertex_count = 6
vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=vertex_count)
edge_indices = ti.field(dtype=ti.i32, shape=36)

@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
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
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 7.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(60.0, 1.0, 0.1, 100.0)
    mvp = proj @ view @ model
    
    for i in range(vertex_count):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    s = 2.0
    vertices[0] = [ s,  s, -s]
    vertices[1] = [-s,  s, -s]
    vertices[2] = [-s, -s, -s]
    vertices[3] = [ s, -s, -s]
    vertices[4] = [ 0,  0,  s]
    vertices[5] = [ 0,  0, -s*2]
    
    edges = [0,1, 1,2, 2,3, 3,0, 0,4, 1,4, 2,4, 3,4, 0,5, 1,5, 2,5, 3,5, 4,5]
    for i in range(len(edges)):
        edge_indices[i] = edges[i]
    
    gui = ti.GUI("3D MVP Auto Rotate", res=(800, 800))
    angle = 0.0
    
    while gui.running:
        angle += 0.5
        compute_transform(angle)
        pts = screen_coords.to_numpy()
        
        col_list = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF]
        for i in range(0, len(edges), 2):
            gui.line(pts[edge_indices[i]], pts[edge_indices[i+1]], 
                    radius=3, color=col_list[i//2 % 6])
        
        gui.show()

if __name__ == '__main__':
    main()