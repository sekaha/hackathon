import numpy as np
from entities import SpaceObject, Particle
from utils.obj_handler import open_obj
from random import randint, gauss
from src.game import game
from LED import *
from src.camera import camera

obj_bullet = open_obj('assets/bullet.obj')

class Bullet(SpaceObject):
    def __init__(self, pos, angle, speed, initial_velocity): #theta_z
        SpaceObject.__init__(self, *obj_bullet, False)

        #theta_z = np.random.uniform(0, 2) * np.pi / 2

        # Define a 90-degree rotation matrix around the X-axis
        # Change this to quaternion
        # if theta_z != 0:
        #     if theta_z == np.pi / 2:
        #         Rz = np.array([
        #             [0, -1, 0],
        #             [1, 0, 0],
        #             [0, 0, 1]
        #         ])
        #     else:
        #         Rz = np.array([
        #             [np.cos(theta_z), -np.sin(theta_z), 0],
        #             [np.sin(theta_z), np.cos(theta_z), 0],
        #             [0, 0, 1]
        #         ])
        #     # Apply the rotation to the model's vertices
        #     self.vertices[:, :3] = self.vertices[:, :3] @ Rz.T

        self.pos = pos
        self.angle = angle
        self.direction = self.angle.rotate([0, 0, 1])
        self.speed = speed
        brightness = 100
        self.color = color_hsv(randint(0,255), 180, brightness)
        self.hp = 100
        self.scale = 2

        # adjust for the player's velocity
        self.velocity = self.direction * self.speed
        # k = 1
        # correction_force = initial_velocity * k
        # self.velocity += correction_force


    def update(self):
        self.pos += self.velocity
        self.hp -= 0.33

        for a in game.asteroids:
            dist = np.sqrt(np.sum((a.pos - self.pos) ** 2))
            
            if dist < a.collision_radius * a.bullet_pad:
                a.brightness = 255

                for _ in range(int(6.5 + a.scale / 6)):
                    Particle(color_hsv(a.hue, 255, a.brightness * 0.75), np.copy(a.pos), a.scale)

                a.hp -= 1
                camera.add_shake(0.2)
                self.hp = 0
 
                if a.hp == 0:
                    a.create_children()
                    camera.add_shake(0.2)
                    game.score += 1
                    game.to_remove.add(a)

                    for _ in range(int(a.scale//3)):
                        Particle(color_hsv(a.hue, 255, a.brightness * 0.75), np.copy(a.pos), a.scale)

                break
                        
        if self.hp <= 0:
            game.to_remove.add(self)

    # def draw_self(self):
    #     super().draw_self(check_remove=True,despawn_threshold=0)