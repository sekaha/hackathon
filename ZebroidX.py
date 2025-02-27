import numpy as np
import time
from math import sin, cos, acos
from obj_handler import open_obj
from random import randint, gauss
from LED import *
from numba import njit
from lore import display_lore


set_orientation(1)

class Camera:
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.pos = np.array([x, y, z], dtype=np.float64)
        self.angle = np.array([roll, pitch, yaw], dtype=np.float64)
        self.shake = 0

class SpaceObject:
    def __init__(self, vertices, faces, cull = True):
        self.vertices = vertices
        self.faces = faces
        self.scale = 1
        self.direction = [0,0,0]
        self.pos = [0,0,0]
        self.angle = [0,0,0]
        self.speed = 1
        self.accel = 1
        self.decel = 1
        self.color = (0,255,0)
        self.cull = cull

    def update(self):
        pass    

    def draw_self(self, blend=True, fill=False):
        world_verts = get_world_vertices(self.vertices, self.scale, self.pos, get_rotation_matrix(self.angle[0], self.angle[1], self.angle[2]))
        proj_verts = np.zeros((len(world_verts), 6), dtype=np.float64)
        proj_verts[:, :3] = world_verts
        proj_verts = np.ascontiguousarray(proj_verts, dtype=np.float64)

        proj_verts = project_vertices(proj_verts, camera.pos, camera.angle) # , W, H, FOV_H, FOV_V

        vet1 = world_verts[self.faces[:, 1]] - world_verts[self.faces[:, 0]]
        vet2 = world_verts[self.faces[:, 2]] - world_verts[self.faces[:, 0]]

        normals = np.cross(vet1, vet2)

        # Handle zero-length normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        normals /= norms

        camera_rays = world_verts[self.faces[:, 0]] - camera.pos

        xxs = proj_verts[self.faces, 3]  # Screen x
        yys = proj_verts[self.faces, 4]  # Screen y
        z_min = np.min(proj_verts[self.faces, 5], axis=1)  # min z in camera space

        # back cull and get visibility
        valid_faces = np.array([
            filter_faces(
                z_min[i],
                normals[i],
                camera_rays[i],
                xxs[i],
                yys[i],
                self.cull
            ) for i in range(len(self.faces))
        ])
        
        valid_indices = np.where(valid_faces)[0]

        set_blend_mode(BM_ADD)
        drawn_edges = set()

        for index in valid_indices:
            triangle = self.faces[index]
            proj_vertices = proj_verts[triangle][:, 3:5]

            edges = [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0])
            ]

            for edge in edges:
                sorted_edge = frozenset(edge)  # Ensure consistent ordering
                
                if sorted_edge not in drawn_edges:
                    drawn_edges.add(sorted_edge)
                    x1, y1 = proj_vertices[edge[0] == triangle][0]
                    x2, y2 = proj_vertices[edge[1] == triangle][0]
                    draw_line(x1, y1, x2, y2, self.color)
 
class Point3D:
    def __init__(self):
        pass

class Particle(Point3D):
    def __init__(self, color, pos, speed, dir = None, decel=0.95):
        global particles
        self.color = color
        self.pos = pos
        self.speed = speed
        self.decel = decel

        if dir is None:
            theta = np.random.uniform(0, 2*np.pi) # azimuthal
            phi = acos(np.random.uniform(-1, 1))

            self.dir = np.zeros(3)
            self.dir[0] = cos(theta) * sin(phi)
            self.dir[1] = sin(theta) * sin(phi)
            self.dir[2] = cos(phi)
        else:
            self.dir = dir

        particles.add(self)


    def draw_self(self):
        x, y, z = project_point(self.pos, camera.pos, camera.angle) # , W, H, FOV_H, FOV_V
        self.pos += self.dir * self.speed
        self.speed *= self.decel

        #set_blend_mode(BM_ADD)

        if 0 < z and 0 <= x < W and 0 <= y < H:
            #if 8 < z:
            draw_pixel(x, y, self.color)
            #else:
            #draw_circle_outline(x-max(0, 4-z//2), y-max(0, 4-z//2), z, self.color)

        if self.speed <= 0.005:
            to_remove.add(self)

    
class Star(Point3D):
    def __init__(self):
        self.is_planet = False
        self.rad = 0
        self.color = WHITE

        # if randint(1, 10) == 1:
        #     self.is_planet = True
        #     self.color = color_hsv(randint(0, 255), 255, 75)
        #     self.rad = randint(2, randint(2, 5))

        theta = np.random.uniform(0, 2*np.pi) # azimuthal
        phi = acos(np.random.uniform(-1, 1))

        self.pos = np.zeros(3)
        self.pos[0] = cos(theta) * sin(phi)
        self.pos[1] = sin(theta) * sin(phi)
        self.pos[2] = cos(phi)

        self.pos *= 1000000

    def draw_self(self):
        x, y, z = project_point(self.pos, camera.pos, camera.angle) # , W, H, FOV_H, FOV_V

        set_blend_mode(BM_NORMAL)

        if 0 < z and 0 <= x < W and 0 <= y < H:
            if self.is_planet == False:
                draw_pixel(x, y, WHITE)
            else:
                draw_circle(x, y, self.rad, BLACK)
                draw_circle_outline(x, y, self.rad, self.color)

class Ring(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *open_obj('planet_ring.obj'), False)


class Planet(SpaceObject):
    def __init__(self):
        # type = randint(0, randint(2,4))
        SpaceObject.__init__(self, *open_obj('planet.obj'))


        theta = np.random.uniform(0, 2*np.pi) # azimuthal
        phi = acos(np.random.uniform(-1, 1))
        #self.vertices = np.concatenate([self.vertices, self.ring.vertices])
        #self.faces = np.concatenate([self.faces, self.ring.faces])

        self.pos = np.zeros(3)
        self.pos[0] = cos(theta) * sin(phi)
        self.pos[1] = sin(theta) * sin(phi)
        self.pos[2] = cos(phi)
        self.scale = randint(75, 255)
        self.pos *= randint(500, 1000)
        self.angle = np.random.uniform(0, 2*np.pi, 3)

        self.rad = randint(2, randint(9,14))

        # try harding the color ig
        hue = randint(0, 255)
        ring_hue = hue

        if np.random.random() < 0.7:
            ring_hue += gauss(0, 30)  # Analogous shift
        else:
            ring_hue += gauss(128, 30)  # Complementary shift

        self.color = color_hsv(hue, 255, 30)

        # Ring
        self.ring = None

        #if randint(0, 1) == 1:
        self.ring = Ring()
        self.ring.color = color_hsv(ring_hue, 255, 30)
        self.ring.angle = np.random.uniform(0, 2*np.pi, 3)
        self.ring.pos = self.pos
        self.ring.scale = self.scale * np.random.uniform(1, np.random.uniform(1, 2))

        # Gravity danger
        center = np.mean(self.vertices, axis=0)
        distances = np.linalg.norm(self.vertices - center, axis=1)
        self.danger_dist = np.mean(distances) * self.scale


    def draw_self(self):
        SpaceObject.draw_self(self, False, True)

        if self.ring:
            # self.angle += (self.ring.angle * 0.01) 
            self.ring.draw_self() # False, True

        if np.linalg.norm(self.pos - player.pos) < self.danger_dist:
            player.hp = 0

        # x, y, z = self._get_projected()

        # set_blend_mode(BM_NORMAL)

        # if 0 < z and -self.rad <= x < W+self.rad and -self.rad <= y < H+self.rad:
        #    draw_circle(x, y, self.rad, BLACK)
        #    draw_circle_outline(x, y, self.rad, self.col)

class Asteroid(SpaceObject):
    def __init__(self):
        self.type = randint(0,4)
        
        if self.type == 0:
            obj = open_obj('pyramid.obj')
            self.hp = 1
        elif self.type == 1:
            obj = open_obj('cube.obj')
            self.hp = 2
        else:
            obj = open_obj('icosphere.obj')
            self.hp = 3

        SpaceObject.__init__(self, *obj)

        self.pos = player.pos + (np.random.rand(3)*10)
        self.scale = np.random.uniform(0.2, 1)
        self.brightness = 190

        self.hue = randint(0,255)
        self.color = color_hsv(self.hue, 255, self.brightness)

        self.rot = np.random.uniform(-0.075, 0.075, 3)

    def draw_self(self):

        self.angle += self.rot
        SpaceObject.draw_self(self)

class Bullet(SpaceObject):
    def __init__(self, pos, angle, speed, theta_z):
        SpaceObject.__init__(self, *open_obj('bullet.obj'), False)

        #theta_z = np.random.uniform(0, 2) * np.pi / 2

        # Define a 90-degree rotation matrix around the X-axis
        if theta_z != 0:
            if theta_z == np.pi / 2:
                Rz = np.array([
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]
                ])
            else:
                Rz = np.array([
                    [np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]
                ])

            # Apply the rotation to the model's vertices
            self.vertices[:, :3] = self.vertices[:, :3] @ Rz.T

        self.pos = pos
        self.angle = angle
        self.speed = speed
        self.color = color_hsv(randint(0,255), 180, 233)
        self.hp = 100

    def update(self):
        global bullets, camera, score, to_remove
        self.pos += get_rotation_matrix(*self.angle) @ np.array([0, 0, self.speed])
        self.hp -= 0.1

        for a in asteroids:
            dist = np.sqrt(np.sum((a.pos - self.pos) ** 2))
            
            if dist < a.scale * 2:
                for _ in range(10):
                    Particle(color_hsv(a.hue, 255, a.brightness * 0.75), np.copy(a.pos), a.scale)

                a.hp -= 1
                camera.shake = min(0.4, camera.shake+0.05)
                self.hp = 0

                if a.hp == 0:
                    camera.shake = min(0.4, camera.shake+0.05)
                    score += 1
                    to_remove.add(a)
                break
                        
        if self.hp <= 0:
            to_remove.add(self)

class Player(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *open_obj('ufo.obj'))
        self.hp = 100
        self.hue = 0
        self.value = 0
        self.saturation = 255

    def draw_self(self):
        global camera
        #self.hp -= 0.05
        SpaceObject.draw_self(self)

        if get_key_pressed('h'):
            self.hp -= 19
            self.vertices += np.random.uniform(-1,1,size=self.vertices.shape) * 0.05
            camera.shake += 0.1

        self.hue = sin(game_time / 200) * 20 + self.hp * 1.3 - 10
        self.value = 50 + cos(game_time / 100) * 50 + self.hp*1.2
        self.color = color_hsv(self.hue, self.saturation, self.value)

score = 0

# Constants
camera = Camera(0.1, 0.1, 5, 0, 0, 0) # np.array([0.1, 0.1, 5, 0.1, 0.1], dtype=np.float64)
player = Player()
W, H = get_width_adjusted(), get_height_adjusted()
A = W/H
FOV_V = np.pi / 4 # 45DEG VERT
FOV_H = FOV_V*A
SENS = 0.02
start_game_time = time.time()
game_time = 0

@njit
def get_rotation_matrix(theta_x: float, theta_y: float, theta_z: float):
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta_x), -np.sin(theta_x)],
        [0.0, np.sin(theta_x), np.cos(theta_x)]
    ], dtype=np.float64)

    Ry = np.array([
        [np.cos(theta_y), 0.0, np.sin(theta_y)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta_y), 0.0, np.cos(theta_y)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0.0],
        [np.sin(theta_z), np.cos(theta_z), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return np.ascontiguousarray(Rz @ Ry @ Rx)


@njit
def get_world_vertices(vertices, scale, pos, rot_matrix):
    scaled_verts = vertices[:, :3] * scale
    rotated_verts = scaled_verts @ np.ascontiguousarray(rot_matrix.T)
    world_verts = rotated_verts + np.asarray(pos, dtype=np.float64)

    return world_verts

@njit
def project_point(pos, camera_pos, camera_angle): # , W, H, FOV_H, FOV_V
    cos_hor, sin_hor = np.cos(-camera_angle[0] + np.pi / 2), np.sin(-camera_angle[0] + np.pi / 2)
    cos_ver, sin_ver = np.cos(-camera_angle[1]), np.sin(-camera_angle[1])

    hor_fov_adjust = 0.5 * W / np.tan(FOV_H * 0.5)
    ver_fov_adjust = 0.5 * H / np.tan(FOV_V * 0.5)

    rot_hor = np.array([[cos_hor, 0.0, -sin_hor],
                        [0.0, 1.0, 0.0],
                        [sin_hor, 0.0, cos_hor]])

    rot_ver = np.array([[1.0, 0.0, 0.0],
                        [0.0, cos_ver, -sin_ver],
                        [0.0, sin_ver, cos_ver]])

    translated = pos[:3] - camera_pos
    rotated = translated @ rot_hor.T @ rot_ver.T

    if np.abs(rotated[2]) < 0.001:  
        rotated[2] = -0.001

    x_proj = int(-hor_fov_adjust * rotated[0] / rotated[2] + 0.5 * W)
    y_proj = int(-ver_fov_adjust * rotated[1] / rotated[2] + 0.5 * H)

    return np.array([x_proj, y_proj, rotated[2]])

@njit
def project_vertices(vertices, camera_pos, camera_angle):
    cos_hor, sin_hor = np.cos(-camera_angle[0] + np.pi / 2), np.sin(-camera_angle[0] + np.pi / 2)
    cos_ver, sin_ver = np.cos(-camera_angle[1]), np.sin(-camera_angle[1])

    hor_fov_adjust = 0.5 * W / np.tan(FOV_H * 0.5)
    ver_fov_adjust = 0.5 * H / np.tan(FOV_V * 0.5)

    rot_hor = np.asarray([[cos_hor, 0.0, -sin_hor],
                          [0.0, 1.0, 0.0],
                          [sin_hor, 0.0, cos_hor]], dtype=np.float64)

    rot_ver = np.asarray([[1.0, 0.0, 0.0],
                          [0.0, cos_ver, -sin_ver],
                          [0.0, sin_ver, cos_ver]], dtype=np.float64)

    translated = vertices[:, :3] - camera_pos

    # Ensure rotation works inside numba
    rotated = translated @ rot_hor.T
    rotated = rotated @ rot_ver.T

    # Avoid division by zero
    rotated[:, 2] = np.where(np.abs(rotated[:, 2]) < 0.001, -0.001, rotated[:, 2])

    # Create an output array to store the projected coordinates
    projected_vertices = np.empty_like(vertices)

    projected_vertices[:, :3] = vertices[:, :3]  # Keep original positions
    projected_vertices[:, 3] = (-hor_fov_adjust * rotated[:, 0] / rotated[:, 2] + 0.5 * W) #.astype(np.int64)
    projected_vertices[:, 4] = (-ver_fov_adjust * rotated[:, 1] / rotated[:, 2] + 0.5 * H) #.astype(np.int64)
    projected_vertices[:, 5] = rotated[:, 2]

    return projected_vertices

@njit
def filter_faces(z_min, normal, CameraRay, xxs, yys, cull):
    # only show vertices on +z (i.e. not behind us), facing the camera, check tringle bounding box
    valid = z_min > 0 and max(xxs) >= 0 and min(xxs) < W and max(yys) >= 0 and min(yys) < H
    valid &= np.dot(normal, CameraRay) < 0 or not cull

    return valid

def axis_angle_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    C = 1 - cos_a
    x, y, z = axis
    return np.array([
        [cos_a + x*x*C,    x*y*C - z*sin_a, x*z*C + y*sin_a],
        [y*x*C + z*sin_a,  cos_a + y*y*C,   y*z*C - x*sin_a],
        [z*x*C - y*sin_a,  z*y*C + x*sin_a, cos_a + z*z*C]
    ])

def move():
    global bullets, camera, smooth_delta
    smooth_delta = 0.99 * smooth_delta + get_delta(120) * 0.01

    MOVE_SPEED = 0.1 * (5 if get_button(JS_FACE1) else 1) * smooth_delta

    if get_key_pressed("g"):
        if get_fps() == 60:
            set_fps(120)
        else:
            set_fps(60)


    delta_yaw   = smooth_delta * SENS * (get_haxis(JS_RSTICK) + get_key('RIGHT') - get_key('LEFT'))
    delta_pitch = smooth_delta * SENS * (get_vaxis(JS_RSTICK) - get_key('UP') + get_key('DOWN'))

    # Get current local axes from the orientation matrix.
    # Assuming: column 0 = right, column 1 = up, column 2 = forward.
    right_vector = camera.angle[:, 0]
    up_vector    = camera.angle[:, 1]

    # Create rotation matrices about the local up (yaw) and right (pitch) axes.
    R_yaw   = axis_angle_rotation(up_vector, delta_yaw)
    R_pitch = axis_angle_rotation(right_vector, delta_pitch)

    # Update angle (note: order matters)
    camera.angle = R_pitch @ R_yaw @ camera.angle

    # Then extract the new forward, right, and up vectors:
    forward = camera.angle[:, 2]
    right   = camera.angle[:, 0]
    up      = camera.angle[:, 1]

    # forward = np.array([
    #     np.cos(camera.angle[0]) * np.cos(camera.angle[1]),  # X
    #     -np.sin(camera.angle[1]),                     # Y (positive when looking up)
    #     np.sin(camera.angle[0]) * np.cos(camera.angle[1])   # Z
    # ])

    # right = np.array([
    #     -np.sin(camera.angle[0]),  # X component
    #     0,                   # Y component (no vertical movement)
    #     np.cos(camera.angle[0])    # Z component
    # ])

    # up = np.cross(right, forward)

    def closest_angel(current, target):
        diff = (target - current) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        return current + diff

    smoothing_time = 0.9
    merge_am = min(1, 1 - smoothing_time ** smooth_delta)
    merge_am = 0.1
    ideal_angle = [camera.angle[1], -camera.angle[0] + np.pi / 2, 0]
    player.angle = [
        a + (closest_angel(a, b) - a) * merge_am 
        for a, b in zip(player.angle, ideal_angle)
    ]

    offset = camera.pos + forward * 2
    vert_offset = -0.065
    player.pos += (offset - player.pos) * merge_am
    R = get_rotation_matrix(player.angle[0], player.angle[1], player.angle[2])
    player.pos += R @ np.array([0, vert_offset + 0.02*sin(game_time/150), -vert_offset*2])
    
    # for _ in range(3):
    #     Particle(color_hsv(135 + randint(-20, 20), 255, 150), player.pos + R @ np.array([0, vert_offset, -1]) + np.random.uniform(-0.03, 0.03, 3), 0.1, -forward + np.random.uniform(-0.17, 0.17, 3)) # + np.random.random(3)

    xspeed = get_haxis(JS_LSTICK) + get_key('d') - get_key('a')
    zspeed = -get_vaxis(JS_LSTICK) + get_key('w') - get_key('s') 
    yspeed = get_key('e') - get_key('q') + get_button(JS_FACE3) - get_button(JS_FACE0)

    #if get_button(JS_R1):
    camera.pos += MOVE_SPEED * zspeed * forward
    camera.pos += MOVE_SPEED * xspeed * right
    camera.pos += MOVE_SPEED * yspeed * up

    camera.pos += (np.random.uniform(-1, 1, 3) * camera.shake)
    camera.shake *= 0.95
    
    if get_button_pressed(JS_R1) or get_key_pressed(' '):
        # Create the bullet with the adjusted angle
        bullets.add(Bullet(np.array(player.pos), np.array(player.angle), 0.3, np.pi / 2))
        #bullets.add(Bullet(np.array(player.pos) +  R @ np.array([1.5, 0, 0]),np.array(player.angle), 0.3, 0))
        #bullets.add(Bullet(np.array(player.pos) -  R @ np.array([1.5, 0, 0]),np.array(player.angle), 0.3, 0)) # right

def menu():
    options = ["Play", "Quit"]

    selection_index = 0
    
    while True:
        set_font(FNT_NORMAL)
        time_passed = time.time() - start_game_time
        title = "Space" if time_passed // 2 % 2 == 0 else "Zebra"
        center_text_horizontal()
        draw_text(get_width_adjusted() / 2, 0, title, WHITE)

        if get_key_pressed("down") or get_button_pressed(JS_PADD):
            selection_index = (selection_index - 1) % len(options)
        if get_key_pressed("up")  or get_button_pressed(JS_PADU):
            selection_index = (selection_index + 1) % len(options)
        if get_key_pressed("enter") or get_button_pressed(JS_FACE0):
            return options[selection_index]

        set_font(FNT_SMALL)

        for idx, i in enumerate(options):
            button_coords = (get_width_adjusted() / 4, 30 + idx * 15)
            button_outline_color = CYAN if selection_index == idx else (0, 0, 110)
            draw_rectangle_outline(button_coords[0], button_coords[1], get_width_adjusted() / 2, 10, button_outline_color)
            center_text_horizontal()
            draw_text(get_width_adjusted() / 2, button_coords[1] + 1 - 4, i, WHITE)

        draw()
        refresh()


def end_screen(score):
    zebra_walking_sprites = [
        Sprite(f"ZebraWalkPics/zebra-pos-{filename}.png") for filename in range(1, 5)
    ]
    selection_index = 0

    refresh()
    draw()
    refresh()

    options = ["Play", "Quit"]

    while True:
        refresh()

        # draw the score
        set_font(FNT_SMALL)
        align_text_right()
        draw_text(get_width_adjusted(), 0, f"score: {score}", WHITE)
        align_text_left()

        if get_key_pressed("right") or get_button_pressed(JS_PADR):
            selection_index = (selection_index - 1) % len(options)
        if get_key_pressed("left")  or get_button_pressed(JS_PADL):
            selection_index = (selection_index + 1) % len(options)
        if get_key_pressed("enter") or get_button_pressed(JS_FACE0):
            return options[selection_index]

        for idx, i in enumerate(options):
            button_coords = (get_width_adjusted() / 2 * idx + 3, 45)
            button_outline_color = CYAN if selection_index == idx else (0, 0, 110)
            draw_rectangle_outline(button_coords[0], button_coords[1], get_width_adjusted() / 2 - 6, 10, button_outline_color)
            center_text_horizontal()
            draw_text((get_width_adjusted() / 2 * idx + (get_width_adjusted() / 2) * (idx + 1)) / 2, button_coords[1] + 1 - 4, i, WHITE)
        draw_sprite(10, 0, zebra_walking_sprites[int((time.time() - start_game_time) / 0.5) % 4])
        draw()

# Main loop
def init_game():
    global asteroids, bullets, score, stars, planets, particles, to_remove, smooth_delta
    smooth_delta = 1
    asteroids = set(Asteroid() for _ in range(1))#range(int(level * 1.2 + 10)))
    stars = set(Star() for _ in range(100))
    planets = set(Planet() for _ in range(2))
    particles = set()
    bullets = set()
    to_remove = set()
    score = 0

level = 0
game_time = 0
# offset = np.array([1, -1, 1])

def game():
    global game_time, to_remove, asteroids, particles, bullets
    game_time += 1

    to_remove = set()

    if get_key('r'):
        init_game()

    if not asteroids or player.hp <= 0:
        return score
    
    for s in stars:
        s.draw_self()

    for p in planets:
        p.draw_self()

    for a in asteroids:
        a.draw_self()

    for p in iter(particles):
        p.draw_self()

    for b in list(bullets):
        b.update()
        b.draw_self()

    bullets -= to_remove
    particles -= to_remove
    asteroids -= to_remove

    # player.update()
    player.draw_self()
    move()

def hud():
    set_font(FNT_SMALL)
    align_text_right()
    draw_text(W-3, -3, str(score), WHITE)
    align_text_left()

def main():
    option = "Play" #menu()
    
    # if level == 0:
    #     display_lore()


    while True:
        if option == "Play":
            init_game()
            # Main loop
            while True:
                refresh()
                if (score := game()) is not None:
                    break
                hud()
                draw()
            option = end_screen(score)
            continue
        elif option == "Quit":
            return
        else:
            raise "Invalid option given in game"

main()