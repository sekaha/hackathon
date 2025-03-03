import numpy as np
from LED import *
from utils.quaternion import Quaternion
from entities import SpaceObject
from random import randint, gauss
from player import player
from src.game import game
from utils.obj_handler import open_obj
from shard import Shard

obj_tetrahedron = open_obj("assets/tetrahedron.obj")
tetrahedron_center = np.mean(obj_tetrahedron[0], axis=0)
tetrahedron_radius = np.mean(
    np.linalg.norm(obj_tetrahedron[0] - tetrahedron_center, axis=1)
)

obj_cube = open_obj("assets/cube.obj")
cube_center = np.mean(obj_cube[0], axis=0)
cube_radius = np.mean(np.linalg.norm(obj_cube[0] - cube_center, axis=1))

obj_icosahedron = open_obj("assets/icosahedron.obj")
icosahedron_center = np.mean(obj_icosahedron[0], axis=0)
icosahedron_radius = np.mean(
    np.linalg.norm(obj_icosahedron[0] - icosahedron_center, axis=1)
)

obj_octahedron = open_obj("assets/octahedron.obj")
octahedron_center = np.mean(obj_octahedron[0], axis=0)
octahedron_radius = np.mean(
    np.linalg.norm(obj_octahedron[0] - octahedron_center, axis=1)
)


class Asteroid(SpaceObject):
    def __init__(self, type=None):
        if type == None:
            self.type = randint(1, 3)
            # self.type = 3
        else:
            self.type = type

        self.leeway = 0.9
        self.bullet_pad = 1.414

        self.face_count = {
            0: 4,
            1: 6,
            2: 8,
            #3: 12,
            3: 20,
        }

        self.shard_size = {
            0: 1,
            1: 0.65,
            2: 0.5,
            3: 0.35,
        }

        if self.type == 0:
            obj = obj_tetrahedron
            self.collision_radius = tetrahedron_radius
            self.hp = 1  # 3 // 3
        elif self.type == 1:
            obj = obj_cube
            self.collision_radius = cube_radius
            self.hp = 2  # 6 // 3
        elif self.type == 2:
            obj = obj_octahedron
            self.collision_radius = octahedron_radius
            self.hp = 3
        else:
            obj = obj_icosahedron
            self.collision_radius = icosahedron_radius
            self.hp = 4

        SpaceObject.__init__(self, *obj)

        self.pos = player.pos + (np.random.rand(3) * 200)
        self.scale = np.random.uniform(18, 48)  # np.random.uniform(3, 12)
        self.max_brightness = 130
        self.brightness = self.max_brightness
        self.collision_radius *= self.scale

        self.hue = randint(0, 255)

        # Small random rotational velocity quaternion
        axis = np.random.uniform(-1, 1, 3)
        axis /= np.linalg.norm(axis)  # Normalize axis
        angle = np.random.uniform(-0.075, 0.075)  # Small rotation per update
        self.rot = Quaternion.from_axis_angle(axis, angle)
        self.dir = Quaternion(*np.random.uniform(-1, 1, 4)).normalize()
        self.speed = np.random.uniform(
            0, np.random.uniform(player.slow_spd / 2, 
            np.random.uniform(player.slow_spd, player.slow_spd * 2))
        )
        self.color = color_hsv(self.hue, 200, self.brightness)

    def create_children(self):
        child1, child2 = None, None
        
        # based on volume of sphere to radius of two spheres with the same total volume
        # half_radius = (1/2) ** (1/3)

        if self.type != 0:
            child1 = Asteroid(randint(0, self.type - 1))
            child1.scale = self.scale * 0.5
            child1.pos = self.pos + np.random.uniform(0, self.scale, 3)
            child1.hue = (self.hue + gauss(0, 30)) % 255
            child1.brightness = 255
            game.asteroids.add(child1)

            if randint(0,2) != 0: 
                child2 = Asteroid(randint(0, self.type - 1))
                child2.pos = self.pos + np.random.uniform(0, self.scale, 3)
                child2.hue = (self.hue + gauss(0, 30)) % 255
                child2.scale = self.scale * 0.5
                child2.brightness = 255
                game.asteroids.add(child2)


        shard_count = self.face_count[self.type]
        
        if child1:
            shard_count -= self.face_count[child1.type]
        
        if child2:
            shard_count -= self.face_count[child2.type]
        
        while shard_count:
            if 2 <= shard_count and self.type == 1 and randint(0,1) == 1:
                shard = Shard(pos=np.array(self.pos), hue=self.hue, square=True)
                shard_count -= 2
                shard.scale = self.scale
            else:
                shard = Shard(pos=np.array(self.pos), hue=self.hue)
                shard_count -= 1
                shard.scale = self.scale * self.shard_size[self.type]
            

    def draw_self(self):
        if self.brightness > self.max_brightness + 1:
            self.brightness += (self.max_brightness - self.brightness) * 0.1
            self.color = color_hsv(self.hue, 200, self.brightness)

        # if game.game_time % 10 == 1:
        #     dist = np.linalg.norm(player.pos - self.pos)

        #     if self.scale / dist < 1:
        #         game.to_remove.add(self)

        self.pos += self.dir.get_vector() * self.speed
        self.angle *= self.rot

        super().draw_self(self, check_remove=True)