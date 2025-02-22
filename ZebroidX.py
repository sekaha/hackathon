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
        self.direction = [0,0,0]
        self.pos = [0,0,0]
        self.speed = [0,0,0]
        self.angle = [0,0,0]
        self.accel = 1
        self.decel = 1
        self.color = (0, 200, 0)

    def update(self):
        project_vertices(self.vertices)

    def rotate(self):
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

        R = Rz @ Ry @ Rx

        return self.vertices[:, :3] @ R.T
            
    def draw_self(self):
        vertices = self.rotate()
        self.angle[0] += 0.01
            
        vet1 = obj.vertices[faces[:, 1], :3] - vertices[faces[:, 0], :3]
        vet2 = vertices[faces[:, 2], :3] - vertices[faces[:, 0], :3]
        normals = np.cross(vet1, vet2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        
        camera_rays = (vertices[faces[:, 0], :3] - camera[:3]) / vertices[faces[:, 0], 5:6]
        
        # Compute projected 2D vertices
        xxs = vertices[faces, 3]
        yys = vertices[faces, 4]
        z_min = np.min(vertices[faces, 5], axis=1)
        
        # Filter valid faces
        valid_faces = np.array([filter_faces(z_min[i], normals[i], camera_rays[i], xxs[i], yys[i]) for i in range(len(faces))])
        valid_indices = np.where(valid_faces)[0]

        for index in valid_indices:
            triangle = faces[index]
            proj_vertices = vertices[triangle][:, 3:]
            
            sorted_y = np.argsort(proj_vertices[:, 1])
            start, middle, end = proj_vertices[sorted_y]

            set_blend_mode(BM_ADD)
            draw_polygon_outline([start[:2], middle[:2], end[:2]], self.color)
        

class Asteroid(SpaceObject):
    def __init__(self):
        self.type = randint(4)
        
        if self.type == 0:
            self.vertices, self.faces = open_obj('pyramid.obj')
        elif self.type == 1:
            self.vertices, self.faces = open_obj('cube.obj')
        else:
            self.vertices, self.faces = open_obj('icosphere.obj')

# Constants
obj_ship = SpaceObject(*open_obj('ship.obj'))
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
    # Camera rotation update
    camera[3] = (camera[3] + SENS * get_haxis(JS_LSTICK)) % (2 * np.pi)  # Yaw
    camera[4] = np.clip(camera[4] + SENS * get_vaxis(JS_LSTICK), -1.57, 1.57)  # Pitch

    # Forward vector in full 3D space
    forward = np.array([
        np.cos(camera[3]),                 # X component
        np.sin(camera[4]) * np.sin(camera[3]),  # Y component (vertical based on pitch)
        np.cos(camera[4]) * np.cos(camera[3])   # Z component
    ])

    MOVE_SPEED = 0.1
    if get_button(JS_FACE0):  # Move forward
        camera[:3] += MOVE_SPEED * forward
    if get_button(JS_FACE1):  # Move backward
        camera[:3] -= MOVE_SPEED * forward        
    # rotate(obj_ship, 0.03, 0.02, 0.05)

def menu():
    options = ["Play", "Quit"]


    selection_index = 0
    while True:

        set_font(FNT_NORMAL)
        time_passed = time.time() - start_game_time
        title = "Space" if time_passed // 2 % 2 == 0 else "Zebra"
        center_text_horizontal()
        draw_text(get_width_adjusted() / 2, 0, title, WHITE)


        if get_key_pressed("down"):
            selection_index = (selection_index - 1) % len(options)
        if get_key_pressed("up"):
            selection_index = (selection_index + 1) % len(options)
        if get_key_pressed("enter"):
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

def main():
    option = menu()
    if option == "Play":
        # Main loop
        while True:
            refresh()
            obj_ship.update()
            obj_ship.draw_self()
            move()
            draw()
    elif option == "Quit":
        return
main()