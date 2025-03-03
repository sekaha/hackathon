import numpy as np
from LED import *
from utils.quaternion import Quaternion
from entities import SpaceObject
# from random import randint, gauss
from src.game import game
from utils.obj_handler import open_obj

obj_shard = open_obj("assets/shard.obj")
obj_square_shard = open_obj("assets/square_shard.obj")

class Shard(SpaceObject):
    def __init__(self, pos=np.array([0.0, 0.0, 0.0]), hue=0, spd=1, square=False):
        if square:
            SpaceObject.__init__(self, *obj_shard, False)
        else:
            SpaceObject.__init__(self, *obj_shard, False)

        self.pos = pos

        self.scale = np.random.uniform(9, 24)
        brightness = 75
        self.hue = hue

        # Small random rotational velocity quaternion
        axis = np.random.uniform(-1, 1, 3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(-0.1, 0.1)

        self.rot = Quaternion.from_axis_angle(axis, angle)
        self.dir = Quaternion(*np.random.uniform(-1, 1, 4)).normalize()
        self.speed = np.random.uniform(1, 3) * spd
        self.color = color_hsv(self.hue, 200, brightness)

        game.particles.add(self)

    def draw_self(self):
        self.pos += self.dir.get_vector() * self.speed
        self.angle *= self.rot

        SpaceObject.draw_self(self) # , check_remove=True