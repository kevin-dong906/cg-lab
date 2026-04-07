import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CTRL = 100
SAMPLING = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
ctrl_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CTRL)
line_indices = ti.field(dtype=ti.i32, shape=MAX_CTRL * 2)
bezier_points = ti.Vector.field(2, dtype=ti.f32, shape=SAMPLING + 1)

def compute_bezier(points, t):
    pts = points.copy()
    while len(pts) > 1:
        next_pts = []
        for i in range(len(pts)-1):
            x = (1-t)*pts[i][0] + t*pts[i+1][0]
            y = (1-t)*pts[i][1] + t*pts[i+1][1]
            next_pts.append([x, y])
        pts = next_pts
    return pts[0]

@ti.kernel
def clear_screen():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.05, 0.05, 0.1])

@ti.kernel
def render_curve(count: ti.i32):
    for i in range(count):
        p = bezier_points[i]
        x = int(p[0] * WIDTH)
        y = int(p[1] * HEIGHT)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pixels[x, y] = ti.Vector([0.1, 0.95, 0.3])

def main():
    window = ti.ui.Window("Custom Bezier Curve", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    points = []
    
    while window.running:
        for evt in window.get_events(ti.ui.PRESS):
            if evt.key == ti.ui.LMB:
                if len(points) < MAX_CTRL:
                    pos = window.get_cursor_pos()
                    points.append(pos)
            elif evt.key == 'c':
                points.clear()
        
        clear_screen()
        cnt = len(points)
        
        if cnt >= 2:
            arr = np.zeros((SAMPLING+1, 2), dtype=np.float32)
            for k in range(SAMPLING+1):
                t = k / SAMPLING
                arr[k] = compute_bezier(points, t)
            bezier_points.from_numpy(arr)
            render_curve(SAMPLING+1)
        
        canvas.set_image(pixels)
        
        if cnt > 0:
            np_arr = np.full((MAX_CTRL, 2), -10.0, dtype=np.float32)
            np_arr[:cnt] = np.array(points, dtype=np.float32)
            ctrl_field.from_numpy(np_arr)
            canvas.circles(ctrl_field, radius=0.007, color=(0.98, 0.3, 0.25))
            
            if cnt >= 2:
                idx_list = []
                for i in range(cnt-1):
                    idx_list += [i, i+1]
                idx_np = np.zeros(MAX_CTRL*2, dtype=np.int32)
                idx_np[:len(idx_list)] = np.array(idx_list, dtype=np.int32)
                line_indices.from_numpy(idx_np)
                canvas.lines(ctrl_field, width=0.0025, indices=line_indices, color=(0.4, 0.7, 0.9))
        
        window.show()

if __name__ == '__main__':
    main()