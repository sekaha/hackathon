import numpy as np
from LED import *
from utils.quaternion import Quaternion
from entities import SpaceObject
from random import randint, gauss, choice
from src.camera import camera
from player import player
from utils.obj_handler import open_obj
from entities import Particle
from shard import Shard

obj_planet_very_low_lod = open_obj("assets/dodecahedron.obj")
obj_planet_low_lod = open_obj("assets/planet.obj")
obj_planet_high_lod = open_obj("assets/planet_high_lod.obj")
obj_planet_medium_lod = open_obj("assets/planet_medium_lod.obj")
obj_ring = open_obj("assets/planet_ring.obj")


class Ring(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_ring, False)


class Planet(SpaceObject):
    def __init__(self, hue=None):
        # type = randint(0, randint(2,4))
        SpaceObject.__init__(self, *obj_planet_low_lod)

        # Spherical coordinate system random placement
        theta = np.random.uniform(0, 2 * np.pi)  # azimuthal
        phi = np.acos(np.random.uniform(-1, 1))

        self.pos = np.zeros(3)
        self.pos[0] = np.cos(theta) * np.sin(phi)
        self.pos[1] = np.sin(theta) * np.sin(phi)
        self.pos[2] = np.cos(phi)
        self.scale = randint(125, randint(600, 900))
        self.pos *= randint(
            2000, randint(3000, randint(4000, randint(5000, randint(6000, 70000))))
        )
        self.angle = Quaternion.from_euler(*np.random.uniform(0, 2 * np.pi, 3))
        brightness = 50
        saturation = 230

        self.rad = randint(2, randint(9, 14))

        # try harding the color ig
        if hue == None:
            hue = randint(0, 255)

        ring_hue = hue
        hue_off = 30

        if np.random.random() < 0.7:
            ring_hue += gauss(0, hue_off)  # Analogous shift
        else:
            ring_hue += gauss(128, hue_off)  # Complementary shift

        self.color = color_hsv(hue, saturation, brightness)

        # Ring
        self.ring = Ring()
        self.ring.color = color_hsv(ring_hue, saturation, brightness)
        self.ring.angle = Quaternion.from_euler(*np.random.uniform(0, 2 * np.pi, 3))
        self.ring.pos = self.pos
        self.ring.scale = self.scale * np.random.uniform(1, np.random.uniform(1, 2))

        spin_speed = 0.001
        ring_normal = self.ring.angle.rotate(np.array([0, 1, 0]))
        self.spin = Quaternion.from_axis_angle(ring_normal, spin_speed)
        self.show_ring = randint(0, 1)

        # Gravity danger
        center = np.mean(self.vertices, axis=0)
        distances = np.linalg.norm(self.vertices - center, axis=1)
        self.danger_dist = np.mean(distances) * self.scale

        # Ring dist
        ring_center = np.mean(self.ring.vertices, axis=0)
        ring_distances = np.linalg.norm(self.ring.vertices - ring_center, axis=1)
        self.ring_dist = np.mean(ring_distances) * self.ring.scale
        self.grav_factor = 100

    def draw_self(self):
        SpaceObject.draw_self(self, True, False, False)
        self.angle = self.spin * self.angle

        if self.show_ring:
            self.ring.draw_self(True, False)  # False, True

        dist = np.linalg.norm(self.pos - player.pos)

        draw_scale = dist / self.danger_dist

        if draw_scale < 3:
            self.vertices, self.faces = obj_planet_high_lod
        elif draw_scale < 6:
            self.vertices, self.faces = obj_planet_medium_lod
        else:
            self.vertices, self.faces = obj_planet_low_lod
        # else:
        #     self.vertices, self.faces = obj_planet_very_low_lod
        if draw_scale < 6:
            direction = (self.pos - player.pos) / dist
            gravity = self.grav_factor * (self.danger_dist / dist**2)
            player.gravity_velocity += gravity * direction

            camera.add_shake(round(gravity / 75, 3))

        if dist < self.danger_dist:
            player.hp -= 3 # * 0
            camera.add_shake(0.03)
            
            for _ in range(10):
                Particle(choice([player.color, self.color]), player.pos + np.random.uniform(-1, 1, 3), randint(2, 4))
            
            shard = Shard(pos=player.pos + np.random.uniform(-1, 1, 3), hue=player.hue, spd=0.5)
            shard.scale = player.scale * np.random.uniform(0.25, 1.5)

            player.gravity_velocity *= -1