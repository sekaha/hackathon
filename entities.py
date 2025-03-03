from utils.quaternion import Quaternion
from LED import *
from utils.graphics import (
    compute_normals_and_visibility,
    compute_projected_vertices,
    filter_visible_faces,
    project_point,
)
from src.camera import camera
from src.game import game
import numpy as np

set_orientation(1)
W, H = get_width_adjusted(), get_height_adjusted()


class SpaceObject:
    def __init__(self, vertices, faces, cull=True):
        self.vertices = vertices
        self.faces = faces
        self.scale = 1
        self.pos = np.array([0, 0, 0], dtype=np.float64)
        self.angle = Quaternion.identity()
        self.color = (0, 255, 0)
        self.cull = cull
        self.min_x, self.max_x = 0, 0
        self.min_y, self.max_y = 0, 0
        self.min_z, self.max_z = 0, 0

    def update(self):
        pass

    def draw_self(
        self,
        blend=True,
        fill=False,
        prevent_duplicates=True,
        check_remove=False,
        despawn_threshold=1,
    ):
        # Compute projected vertices (Numba-optimized)
        proj_verts = compute_projected_vertices(
            self.vertices,
            self.scale,
            self.pos,
            self.angle.rotation_matrix,
            camera.pos,
            camera.rotation_matrix,
            camera.shake,
        )

        # Compute normals and camera rays
        normals, camera_rays = compute_normals_and_visibility(
            proj_verts[:, :3], self.faces, camera.pos
        )

        # Extract screen coordinates and depths
        xxs = proj_verts[self.faces, 3]  # Screen x
        yys = proj_verts[self.faces, 4]  # Screen y
        zzs = proj_verts[self.faces, 5]

        self.min_x, self.max_x = np.min(xxs), np.max(xxs)
        self.min_y, self.max_y = np.min(yys), np.max(yys)
        self.min_z, self.max_z = np.min(zzs), np.max(zzs)

        if check_remove and (
            self.max_x - self.min_x < despawn_threshold
            and self.max_x - self.min_x < despawn_threshold
        ):
            game.to_remove.add(self)

        valid_faces = filter_visible_faces(
            np.min(proj_verts[self.faces, 5], axis=1), normals, camera_rays, xxs, yys, self.cull, len(self.faces)
        )

        valid_indices = np.where(valid_faces)[0]

        drawn_edges = set()

        if fill:
            for index in valid_indices:
                triangle = self.faces[index]
                proj_vertices = proj_verts[triangle][:, 3:5]  # [x_screen, y_screen]
                set_blend_mode(BM_NORMAL)
                sorted_y = np.argsort(proj_vertices[:, 1])
                start, middle, end = proj_vertices[sorted_y]
                draw_polygon([start[:2], middle[:2], end[:2]], BLACK)

        if blend:
            set_blend_mode(BM_ADD)
        else:
            set_blend_mode(BM_NORMAL)

        for index in valid_indices:
            triangle = self.faces[index]
            proj_vertices = proj_verts[triangle][:, 3:5]  # [x_screen, y_screen]

            if prevent_duplicates:
                edges = [
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ]
                for edge in edges:
                    sorted_edge = frozenset(edge)
                    if sorted_edge not in drawn_edges:
                        drawn_edges.add(sorted_edge)
                        x1, y1 = proj_vertices[edge[0] == triangle][0]
                        x2, y2 = proj_vertices[edge[1] == triangle][0]
                        draw_line(x1, y1, x2, y2, self.color)
            else:
                sorted_y = np.argsort(proj_vertices[:, 1])
                start, middle, end = proj_vertices[sorted_y]
                draw_polygon_outline([start[:2], middle[:2], end[:2]], self.color)


class Point3D:
    def __init__(self):
        pass


class Particle(Point3D):
    def __init__(self, color, pos, speed, dir=None, decel=0.95):
        self.color = color
        self.pos = pos
        self.speed = speed
        self.decel = decel

        if dir is None:
            theta = np.random.uniform(0, 2 * np.pi)  # azimuthal
            phi = np.acos(np.random.uniform(-1, 1))

            self.dir = np.zeros(3)
            self.dir[0] = np.cos(theta) * np.sin(phi)
            self.dir[1] = np.sin(theta) * np.sin(phi)
            self.dir[2] = np.cos(phi)
        else:
            self.dir = dir

        game.particles.add(self)

    def draw_self(self):
        x, y, z = project_point(
            self.pos, camera.pos, camera.rotation_matrix, camera.shake
        )  # , W, H, FOV_H, FOV_V
        self.pos += self.dir * self.speed
        self.speed *= self.decel

        # set_blend_mode(BM_ADD)

        if 0 < z and 0 <= x < W and 0 <= y < H:
            # if 8 < z:
            draw_pixel(x, y, self.color)
            # else:
            # draw_circle_outline(x-max(0, 4-z//2), y-max(0, 4-z//2), z, self.color)

        if self.speed <= 0.005:
            game.to_remove.add(self)
