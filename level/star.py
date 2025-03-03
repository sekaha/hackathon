import numpy as np
from LED import *
from entities import Point3D
from random import randint
from utils.graphics import project_point
from src.camera import camera

set_orientation(1)
W, H = get_size_adjusted()

class Star(Point3D):
    def __init__(self):
        self.is_planet = False
        self.rad = 0
        self.color = color_hsv(randint(0, 255), 33, 255)

        # if randint(1, 10) == 1:
        #     self.is_planet = True
        #     self.color = color_hsv(randint(0, 255), 255, 75)
        #     self.rad = randint(2, randint(2, 5))

        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        theta = np.random.uniform(0, 2*np.pi) # azimuthal
        phi = np.acos(np.random.uniform(-1, 1))

        self.pos = np.zeros(3)

        self.pos[0] = np.cos(theta) * np.sin(phi)
        self.pos[1] = np.sin(theta) * np.sin(phi)
        self.pos[2] = np.cos(phi)

        self.pos *= 1000000

    def draw_self(self):
        x, y, z = project_point(self.pos, camera.pos, camera.rotation_matrix, camera.shake) # , W, H, FOV_H, FOV_V

        set_blend_mode(BM_NORMAL)

        if 0 < z and 0 <= x < W and 0 <= y < H:
            if self.is_planet == False:
                draw_pixel(x, y, self.color)
            else:
                draw_circle(x, y, self.rad, BLACK)
                draw_circle_outline(x, y, self.rad, self.color)
