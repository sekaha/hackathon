import numpy as np
from LED import *
from utils.quaternion import Quaternion
from entities import SpaceObject
from src.settings import settings
from random import randint, gauss
from src.game import game
from src.camera import camera
from utils.obj_handler import open_obj

obj_exterior = open_obj("assets/fighter.obj")
obj_interior_hull = open_obj("assets/fighter_interior_hull.obj")
obj_interior_main = open_obj("assets/fighter_interior_main.obj")
obj_chairs = open_obj("assets/fighter_interior_chairs.obj")
obj_display = open_obj("assets/fighter_interior_display.obj")
obj_dash = open_obj("assets/fighter_interior_dash.obj")
obj_floor = open_obj("assets/fighter_interior_floor.obj")

class ShipInteriorMain(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_interior_main)
        self.start_hue = 160 # 100
        self.hue = 0
        self.value = 0
        self.saturation = 235

    def draw_self(self):
        SpaceObject.draw_self(self, True)
        self.hue = np.sin(game.game_time / 200) * 10 + self.start_hue
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)

class ShipDash(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_dash)
        self.start_hue = 230
        self.hue = 0
        self.value = 0
        self.saturation = 235

    def draw_self(self):
        SpaceObject.draw_self(self, True)
        self.hue = np.sin(game.game_time / 200) * 10 + self.start_hue
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)


class ShipDisplay(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_display)
        self.start_hue = 90
        self.hue = 0
        self.value = 0
        self.saturation = 235

    def draw_self(self):
        SpaceObject.draw_self(self, True)
        self.hue = np.sin(game.game_time / 200) * 10 + self.start_hue
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)


class ShipChairs(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_chairs)
        self.start_hue = 128
        self.hue = 0
        self.value = 0
        self.saturation = 235

    def draw_self(self):
        SpaceObject.draw_self(self, True)
        self.hue = np.sin(game.game_time / 200) * 10 + self.start_hue
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)

class ShipInteriorhull(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_interior_hull)
        self.start_hue = 160
        self.hue = 0
        self.value = 0
        self.saturation = 235

    def draw_self(self):
        SpaceObject.draw_self(self, True)
        self.hue = np.sin(game.game_time / 200) * 10 + self.start_hue
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)

class ShipFloor(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_floor)
        self.start_hue = 190
        self.hue = 0
        self.value = 0
        self.saturation = 235

    def draw_self(self):
        SpaceObject.draw_self(self, True)
        self.hue = np.sin(game.game_time / 200) * 10 + self.start_hue
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)


class ShipExterior(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_exterior)
        self.scale = 4
        # self.interior = ShipInteriorMain()
        self.hull = ShipInteriorhull()
        self.chairs = ShipChairs()
        self.display = ShipDisplay()
        self.dash = ShipDash()
        self.floor = ShipFloor()
        self.pos += camera.angle.rotate([0, -0.3, 0])
        self.pos = np.array([0, -0.5, -2.5])
        self.hue = 0
        self.value = 0
        self.t = 0
        self.saturation = 235
        # camera.angle *= Quaternion.from_axis_angle(camera.angle.rotate([0, 1, 0]), 3.14)
        # camera.angle *= Quaternion.from_axis_angle(camera.angle.rotate([1, 0, 0]), -0.75)

    def draw_self(self):


        # Get input for yaw and pitch
        yaw_input = (
            game.smooth_delta
            * settings.sensitivity
            * (get_haxis(JS_RSTICK) + get_key("RIGHT") - get_key("LEFT"))
        )
        pitch_input = (
            game.smooth_delta
            * settings.sensitivity
            * (get_vaxis(JS_RSTICK) - get_key("UP") + get_key("DOWN"))
        )
        roll_input = (
            game.smooth_delta
            * settings.sensitivity
            * (get_trigger(JS_R2) - get_trigger(JS_L2))
        )

        # Create quaternions for local rotations using the current camera orientation.
        yaw_quat = Quaternion.from_axis_angle(
            camera.angle.rotate([0, 1, 0]), -yaw_input
        )
        pitch_quat = Quaternion.from_axis_angle(
            camera.angle.rotate([1, 0, 0]), pitch_input
        )
        roll_quat = Quaternion.from_axis_angle(
            camera.angle.rotate([0, 0, 1]), roll_input
        )

        # Update the camera's orientation by combining the rotations.
        # Note: the multiplication order matters.
        camera.angle = yaw_quat * pitch_quat * roll_quat * camera.angle
        camera.angle.normalize()


        forward = camera.angle.rotate([0, 0, 1])  # forward direction (z-axis)
        right = camera.angle.rotate([1, 0, 0])  # right direction (x-axis)
        up = camera.angle.rotate([0, 1, 0])  # up direction (y-axis)

        xspeed = get_haxis(JS_LSTICK) + get_key("d") - get_key("a")
        zspeed = get_vaxis(JS_LSTICK) + get_key("w") - get_key("s")
        yspeed = (
            get_key("e") - get_key("q") + get_button(JS_FACE3) - get_button(JS_FACE0)
        )


        self.pos += (forward * -zspeed + right * xspeed + up * -yspeed) * 0.015


        self.t += 1
        # SpaceObject.draw_self(self, True)
        #camera.angle *= Quaternion.from_axis_angle(camera.angle.rotate([1, 0, 0]), 0.001)

        # self.pos = camera.angle.rotate([ np.cos(self.t / 3333) * 0.02, np.sin(self.t / 3999) * 0.045 + 0.333, 2.4-self.t / 333]) # *
        # print(1.5 + 1.7 * np.sin(self.t/50))

        self.hue = np.sin(game.game_time / 200) * 10 + 140
        self.value = 120 + np.cos(game.game_time / 100) * 50
        self.color = color_hsv(self.hue, self.saturation, self.value)

        # self.interior.pos = self.pos
        # self.interior.angle = self.angle
        # self.interior.scale = self.scale
        # self.interior.draw_self()

        self.chairs.pos = self.pos
        self.chairs.angle = self.angle
        self.chairs.scale = self.scale
        self.chairs.draw_self()

        self.display.pos = self.pos
        self.display.angle = self.angle
        self.display.scale = self.scale
        self.display.draw_self()

        self.dash.pos = self.pos
        self.dash.angle = self.angle
        self.dash.scale = self.scale
        self.dash.draw_self()

        self.hull.pos = self.pos
        self.hull.angle = self.angle
        self.hull.scale = self.scale
        self.hull.draw_self()

        self.floor.pos = self.pos
        self.floor.angle = self.angle
        self.floor.scale = self.scale
        self.floor.draw_self()



exterior = ShipExterior()

def draw_menu():
    exterior.draw_self()