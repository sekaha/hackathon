from LED import *
from src.lore import display_lore
from entities import Point3D
from src.game import game
from player import player
from src.camera import camera
from utils.graphics import project_point
import time
from math import sin

set_orientation(1)
W, H = get_width_adjusted(), get_height_adjusted()
start_game_time = time.time()
cnv_indicator = create_canvas(W, H)
spr_indicator = create_sprite("assets/indicator.png")
spr_indicator.center_origin()


class Crosshair(Point3D):
    def __init__(self):
        self.pos = np.array([0, 0, 0])
        self.draw_x, self.draw_y = 0, 0

    def draw_self(self):
        cross_pos = (
            player.pos + (player.angle.rotate([0, 0, player.bullet_spd])) * 24
        )  # - player.velocity
        cross_x, cross_y, cross_z = project_point(
            cross_pos, camera.pos, camera.rotation_matrix, camera.shake
        )  # , W, H, FOV_H, FOV_V
        self.draw_x, self.draw_y = cross_x, cross_y

        if 0 < cross_z and 0 <= cross_x < W and 0 <= cross_y < H:
            # if 8 < z:
            draw_circle_outline(cross_x, cross_y, 3, color_hsv(140, 255, 150))

        # lead indicator, shows where the bullet will end up if you continue your velocity
        lead_pos = (
            player.pos
            + (
                player.angle.rotate([0, 0, player.bullet_spd])
                - player.velocity
                - player.gravity_velocity
            )
            * 24
        )  #
        lead_x, lead_y, lead_z = project_point(
            lead_pos, camera.pos, camera.rotation_matrix, camera.shake
        )  # , W, H, FOV_H, FOV_V

        if 0 < lead_z and 0 <= lead_x < W and 0 <= lead_y < H:
            # if 8 < z:
            dist = ((lead_x - cross_x) ** 2 + (lead_y - cross_y) ** 2) ** 0.5
            brightness = min(150, 150 * (dist / 4))
            draw_circle_outline(lead_x, lead_y, 1, color_hsv(140, 255, brightness))


def asteroid_indicator():
    focus, dist = player.nearest_asteroid

    set_canvas(cnv_indicator)
    set_blend_mode(BM_NORMAL)
    set_alpha(244)
    draw_rectangle(0, 0, W, H, BLACK)
    set_alpha(255)
    set_blend_mode(BM_ADD)

    if focus:
        cx, cy = W // 2, H // 2

        focus_x, focus_y = (focus.min_x + focus.max_x) / 2, (focus.min_y + focus.max_y)

        dir_x = focus_x - cx
        dir_y = focus_y - cy

        if focus.max_z < 0 or focus.min_z < 0:
            dir_x = -dir_x
            dir_y = -dir_y
        
        screen_dist = (dir_x**2 + dir_y**2) ** 0.5

        if screen_dist != 0:
            dir_x /= screen_dist
            dir_y /= screen_dist

        scale_x = cx / abs(dir_x) if dir_x != 0 else float("inf")
        scale_y = cy / abs(dir_y) if dir_y != 0 else float("inf")

        scale = min(scale_x, scale_y) * 1.5
        draw_x = cx + dir_x * scale
        draw_y = cy + dir_y * scale

        intensity = min(1, abs(sin(game.game_time / dist)) / (dist / 88))
        draw_sprite(draw_x, draw_y, colorize(spr_indicator, intensity * np.array([255.0, 0.0, 64.0])))

    reset_canvas()
    draw_canvas(0, 0, cnv_indicator)


def hud():
    # set_blend_mode(BM_NORMAL)
    set_font(FNT_SMALL)
    align_text_right()
    draw_text(W - 3, -3, str(game.score), WHITE)
    align_text_left()
    crosshair.draw_self()
    asteroid_indicator()
    center_text()

    if np.linalg.norm(player.gravity_velocity) > player.max_grav_spd * 0.4:
        draw_text(W//2, H//2, "Gravity Warning", color_hsv(0, 255, sin(game.game_time / 33)*126 + 126))


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
        if get_key_pressed("up") or get_button_pressed(JS_PADU):
            selection_index = (selection_index + 1) % len(options)
        if get_key_pressed("enter") or get_button_pressed(JS_FACE0):
            return options[selection_index]

        set_font(FNT_SMALL)

        for idx, i in enumerate(options):
            button_coords = (get_width_adjusted() / 4, 30 + idx * 15)
            button_outline_color = CYAN if selection_index == idx else (0, 0, 110)
            draw_rectangle_outline(
                button_coords[0],
                button_coords[1],
                get_width_adjusted() / 2,
                10,
                button_outline_color,
            )
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

        # draw the game.score
        set_font(FNT_SMALL)
        align_text_right()
        draw_text(get_width_adjusted(), 0, f"game.score: {game.score}", WHITE)
        align_text_left()

        if get_key_pressed("right") or get_button_pressed(JS_PADR):
            selection_index = (selection_index - 1) % len(options)
        if get_key_pressed("left") or get_button_pressed(JS_PADL):
            selection_index = (selection_index + 1) % len(options)
        if get_key_pressed("enter") or get_button_pressed(JS_FACE0):
            return options[selection_index]

        for idx, i in enumerate(options):
            button_coords = (get_width_adjusted() / 2 * idx + 3, 45)
            button_outline_color = CYAN if selection_index == idx else (0, 0, 110)
            draw_rectangle_outline(
                button_coords[0],
                button_coords[1],
                get_width_adjusted() / 2 - 6,
                10,
                button_outline_color,
            )
            center_text_horizontal()
            draw_text(
                (
                    get_width_adjusted() / 2 * idx
                    + (get_width_adjusted() / 2) * (idx + 1)
                )
                / 2,
                button_coords[1] + 1 - 4,
                i,
                WHITE,
            )
        draw_sprite(
            10, 0, zebra_walking_sprites[int((time.time() - start_game_time) / 0.5) % 4]
        )
        draw()


crosshair = Crosshair()
