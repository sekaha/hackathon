from entities import SpaceObject
from utils.quaternion import Quaternion
from src.settings import settings
from utils.obj_handler import open_obj
from src.game import game
from src.camera import camera
from math import exp
from bullet import Bullet
from LED import *

# set_fps(60)
obj_ufo = open_obj("assets/ufo.obj")
obj_fighter = open_obj("assets/fighter.obj")


class Player(SpaceObject):
    def __init__(self):
        SpaceObject.__init__(self, *obj_fighter)
        self.hp = 100
        self.hue = 0
        self.value = 0
        self.saturation = 235
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.gravity_velocity = np.array([0.0, 0.0, 0.0])
        self.mod = 3
        self.max_spd = 0.675 * self.mod
        self.slow_spd = 0.675 * self.mod
        self.fast_spd = 1 * self.mod
        self.max_grav_spd = (
            self.max_spd / (2**0.5) * 0.8
        )  # so if you're flying by diagonally then at the very least you're in equillibrium
        self.friction = 0.97
        self.accel = 0.04 * self.mod
        self.reload_t = 0
        self.reload_spd = 30
        self.bullet_spd = 12
        self.nearest_asteroid = (None, float("inf"))

    def update(self):
        # MOVE_SPEED = 0.1 * (5 if get_button(JS_FACE1) else 1) * game.smooth_delta

        if get_key_pressed("g"):
            if get_fps() == 60:
                set_fps(120)
            else:
                set_fps(60)

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

        if get_button(JS_L1):
            self.max_spd += (self.fast_spd - self.max_spd) * 0.1
        else:
            self.max_spd += (self.slow_spd - self.max_spd) * 0.1

        max_spd = self.max_spd + min(
            abs(roll_input * 2) * self.max_spd, self.max_spd * 0.25
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

        grav_speed = np.linalg.norm(self.gravity_velocity)

        if grav_speed > self.max_grav_spd:
            self.gravity_velocity = (
                self.gravity_velocity / grav_speed
            ) * self.max_grav_spd

        small_offset = 1
        offset = np.copy(camera.pos)
        offset += forward * max(0, np.dot(self.velocity / self.mod, forward) * 10)
        offset += camera.angle.rotate(
            [0, -small_offset + 0.125 * np.sin(game.game_time / 150), small_offset * 3]
        )

        player.pos += (offset - player.pos) * (
            1 - exp(-self.mod * 2 * game.smooth_delta)
        )
        player.angle = player.angle.slerp(camera.angle, 0.1)

        # Joystick input for movement
        xspeed = get_haxis(JS_LSTICK) + get_key("d") - get_key("a")
        zspeed = -get_vaxis(JS_LSTICK) + get_key("w") - get_key("s")
        yspeed = (
            get_key("e") - get_key("q") + get_button(JS_FACE3) - get_button(JS_FACE0)
        )

        self.velocity *= self.friction

        self.velocity += self.accel * zspeed * forward
        self.velocity += self.accel * xspeed * -right
        self.velocity += self.accel * yspeed * up

        speed = np.linalg.norm(self.velocity)

        if speed > max_spd:
            self.velocity = (self.velocity / speed) * max_spd

        camera.pos += self.velocity + self.gravity_velocity
        self.gravity_velocity *= 0.9

        if self.reload_t == 0:
            if get_button(JS_R1) or get_key(" "):
                game.bullets.add(
                    Bullet(
                        np.array(self.pos), self.angle, self.bullet_spd, self.velocity
                    )
                )
                self.velocity -= forward * 0.1
                self.reload_t = self.reload_spd
        else:
            self.reload_t -= 1

    def draw_self(self):
        SpaceObject.draw_self(self, True, False, True)
        self.nearest_asteroid = (None, float("inf"))

        for a in game.asteroids:
            dist = np.linalg.norm((a.pos - self.pos))

            if dist < a.collision_radius * a.leeway:
                self.hp -= 1
                self.vertices += (
                    np.random.uniform(-1, 1, size=self.vertices.shape) * 0.005
                )
                camera.shake += 0.1

            if (
                (get_width_adjusted() < a.min_x
                or a.max_x < 0
                or get_height_adjusted() < a.min_y
                or a.max_y < 0)
                and dist < self.nearest_asteroid[1]
            ):
                self.nearest_asteroid = (a, dist)

        # if self.nearest_asteroid[0]:
        #     draw_rectangle_outline(
        #         self.nearest_asteroid[0].min_x,
        #         self.nearest_asteroid[0].min_y,
        #         self.nearest_asteroid[0].max_x - self.nearest_asteroid[0].min_x,
        #         self.nearest_asteroid[0].max_y - self.nearest_asteroid[0].min_y,
        #         GREEN,
        #     )

        self.hue = np.sin(game.game_time / 200) * 10 + self.hp * 1.5 - 10
        self.value = 50 + np.cos(game.game_time / 100) * 50 + self.hp * 1.2
        self.color = color_hsv(self.hue, self.saturation, self.value)


player = Player()
