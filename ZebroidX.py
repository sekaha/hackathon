import numpy as np
import time
from obj_handler import open_obj
from random import randint
from LED import *

set_orientation(1)

# Zebroid X
class SpaceObject:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.scale = 1
        self.direction = [0,0,0]
        self.pos = [0,0,0]
        self.speed = [0,0,0]
        self.angle = [0,0,0]
        self.accel = 1
        self.decel = 1
        self.color = (0, 200, 0)

    def update(self):
        pass
    
    def _get_rotation_matrix(self):
        theta_x, theta_y, theta_z = self.angle

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def get_world_vertices(self):
        scaled_verts = self.vertices[:, :3] * self.scale
        R = self._get_rotation_matrix()
        rotated_verts = scaled_verts @ R.T
        world_verts = rotated_verts + self.pos
        return world_verts

    def draw_self(self):
        # self.angle[0] += 0.01
        # self.angle[1] += 0.05
        # self.angle[2] += 0.03
        world_verts = self.get_world_vertices()
        proj_verts = np.zeros((len(world_verts), 6))
        proj_verts[:, :3] = world_verts

        project_vertices(proj_verts)

        vet1 = world_verts[self.faces[:, 1]] - world_verts[self.faces[:, 0]]
        vet2 = world_verts[self.faces[:, 2]] - world_verts[self.faces[:, 0]]
        normals = np.cross(vet1, vet2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        camera_rays = world_verts[self.faces[:, 0]] - camera[:3]

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
                yys[i]
            ) for i in range(len(self.faces))
        ])
        valid_indices = np.where(valid_faces)[0]

        for index in valid_indices:
            triangle = self.faces[index]
            proj_vertices = proj_verts[triangle][:, 3:5]

            sorted_y = np.argsort(proj_vertices[:, 1])
            start, middle, end = proj_vertices[sorted_y]

            set_blend_mode(BM_ADD)
            draw_polygon_outline([start[:2], middle[:2], end[:2]], self.color)        

class Asteroid(SpaceObject):
    global player

    def __init__(self):
        self.type = randint(0,4)
        
        if self.type == 0:
            obj = open_obj('pyramid.obj')
        elif self.type == 1:
            obj = open_obj('cube.obj')
        else:
            obj = open_obj('icosphere.obj')

        SpaceObject.__init__(self, *obj)

        self.pos = player.pos + (np.random.rand(3)*10)
        self.scale = 0.25

    def draw_self(self):
        self.angle[0] += 0.1
        SpaceObject.draw_self(self)


class Player(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *open_obj('ship.obj'))

# Constants
player = Player()
W, H = get_width_adjusted(), get_height_adjusted()
A = W/H
FOV_V = np.pi / 4 # 45DEG VERT
FOV_H = FOV_V*A
SENS = 0.01
camera = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1])
start_game_time = time.time()

def project_vertices(vertices):
    global camera

    cos_hor, sin_hor = np.cos(-camera[3] + np.pi / 2), np.sin(-camera[3] + np.pi / 2)
    cos_ver, sin_ver = np.cos(-camera[4]), np.sin(-camera[4])

    hor_fov_adjust = 0.5 * W / np.tan(FOV_H * 0.5)
    ver_fov_adjust = 0.5 * H / np.tan(FOV_V * 0.5)

    rot_hor = np.array([[cos_hor, 0, -sin_hor],
                         [0, 1, 0],
                         [sin_hor, 0, cos_hor]])

    rot_ver = np.array([[1, 0, 0],
                         [0, cos_ver, -sin_ver],
                         [0, sin_ver, cos_ver]])

    translated = vertices[:, :3] - camera[:3]

    rotated = translated @ rot_hor.T @ rot_ver.T
    # rotated[:, 2] = np.where(np.abs(rotated[:, 2]) < 0.001, -0.001, rotated[:, 2])

    # project
    vertices[:, 3] = (-hor_fov_adjust * rotated[:, 0] / rotated[:, 2] + 0.5 * W).astype(int)
    vertices[:, 4] = (-ver_fov_adjust * rotated[:, 1] / rotated[:, 2] + 0.5 * H).astype(int)
    vertices[:, 5] = rotated[:, 2] 

def filter_faces(z_min, normal, CameraRay, xxs, yys):
    # only show vertices on +z (i.e. not behind us), facing the camera, check tringle bounding box
    if z_min > 0 and np.dot(normal, CameraRay) < 0 and max(xxs) >= 0 and min(xxs) < W and max(yys) >= 0 and min(yys) < H:
        return True
    else:
        return False        

def move():
    MOVE_SPEED = 0.1

    camera[3] = (camera[3] - SENS * get_haxis(JS_LSTICK)) % (2 * np.pi)
    camera[4] = (camera[4] + SENS * get_vaxis(JS_LSTICK) )% (2 * np.pi) 

    forward = np.array([
        np.cos(camera[3]) * np.cos(camera[4]),  # X
        -np.sin(camera[4]),                     # Y (positive when looking up)
        np.sin(camera[3]) * np.cos(camera[4])   # Z
    ])
    
    #player.angle = [camera[3], camera[4], 0]
    player.angle = [-camera[4], camera[3] - np.pi / 2, 0]
    offset = camera[:3] + forward * 2
    player.pos += (offset - player.pos) * 0.1


    if get_button(JS_FACE0):
        camera[:3] += MOVE_SPEED * forward

    if get_button(JS_FACE1):
        camera[:3] -= MOVE_SPEED * forward

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
            button_outline_color = CYAN if selection_index == idx else GREY
            draw_rectangle_outline(button_coords[0], button_coords[1], get_width_adjusted() / 2, 10, button_outline_color)
            center_text_horizontal()
            draw_text(get_width_adjusted() / 2, button_coords[1] + 1 - 4, i, WHITE)

        draw()
        refresh()

# Main loop
asteroids = [Asteroid() for _ in range(10)]
offset = np.array([1, -1, 1])

def game():

    for a in asteroids:
    #     a.update()
        a.draw_self()
    # player.update()
    player.draw_self()
    move()


def main():
    option = menu()
    if option == "Play":
        # Main loop
        while True:
            refresh()
            game()
            draw()
    elif option == "Quit":
        return
main()