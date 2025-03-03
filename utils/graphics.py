import numpy as np
from numba import njit
from LED import *

# Constants
set_orientation(1)
W, H = get_width_adjusted(), get_height_adjusted()
A = W/H
FOV_V = np.pi / 4 # 45DEG VERT
FOV_H = FOV_V*A

@njit()
def get_world_vertices(vertices, scale, pos, rot_matrix):
    scaled_verts = vertices[:, :3] * scale
    rotated_verts = scaled_verts @ np.ascontiguousarray(rot_matrix.T)
    world_verts = rotated_verts + np.asarray(pos, dtype=np.float64)

    return world_verts

@njit()
def project_point(pos, camera_pos, camera_rot_matrix, camera_shake):
    # Translate point relative to camera
    translated = pos[:3] - camera_pos + (np.random.uniform(-1, 1, 3) * camera_shake)
    
    # Rotate into camera space using the precomputed rotation matrix
    rotated = translated @ camera_rot_matrix  # rot_matrix is already transposed
    
    # Avoid division by zero
    if np.abs(rotated[2]) < 0.001:
        rotated[2] = -0.001
    
    # Perspective projection
    hor_fov_adjust = 0.5 * W / np.tan(FOV_H * 0.5)
    ver_fov_adjust = 0.5 * H / np.tan(FOV_V * 0.5)
    x_proj = int(-hor_fov_adjust * rotated[0] / rotated[2] + 0.5 * W)
    y_proj = int(-ver_fov_adjust * rotated[1] / rotated[2] + 0.5 * H)
    
    return np.array([x_proj, y_proj, rotated[2]])

@njit()
def project_vertices(vertices, camera_pos, camera_rot_matrix, camera_shake):
    # Translate vertices relative to camera
    translated = vertices[:, :3] - camera_pos  + (np.random.uniform(-1, 1, 3) * camera_shake)
    
    # Rotate into camera space
    rotated = translated @ camera_rot_matrix  # rot_matrix is already transposed
    
    # Avoid division by zero
    rotated[:, 2] = np.where(np.abs(rotated[:, 2]) < 0.001, -0.001, rotated[:, 2])
    
    # Perspective projection
    hor_fov_adjust = 0.5 * W / np.tan(FOV_H * 0.5)
    ver_fov_adjust = 0.5 * H / np.tan(FOV_V * 0.5)
    
    # Create output array
    projected_vertices = np.empty_like(vertices)
    projected_vertices[:, :3] = vertices[:, :3]  # Preserve original positions
    projected_vertices[:, 3] = (-hor_fov_adjust * rotated[:, 0] / rotated[:, 2] + 0.5 * W)
    projected_vertices[:, 4] = (-ver_fov_adjust * rotated[:, 1] / rotated[:, 2] + 0.5 * H)
    projected_vertices[:, 5] = rotated[:, 2]
    
    return projected_vertices

@njit()
def filter_visible_faces(z_min, normals, camera_rays, xxs, yys, cull, num_faces):
    valid_faces = np.zeros(num_faces, dtype=np.bool_)

    for i in range(num_faces):
        # Check that the face is in front of the camera and its bounding box overlaps the screen.
        face_valid = (z_min[i] > 1e-6 and 
                      max(xxs[i]) >= 0 and min(xxs[i]) < W and 
                      max(yys[i]) >= 0 and min(yys[i]) < H)

        # Apply backface culling: if culling is enabled, only accept faces facing the camera.
        face_valid = face_valid and (np.dot(normals[i], camera_rays[i]) < -1e-6 or not cull)
        valid_faces[i] = face_valid
    
    return valid_faces


@njit()
def compute_projected_vertices(vertices, scale, pos, rotation_matrix, camera_pos, camera_rotation, shake):
    world_verts = get_world_vertices(vertices, scale, pos, rotation_matrix)
    proj_verts = np.zeros((len(world_verts), 6), dtype=np.float64)
    proj_verts[:, :3] = world_verts
    return project_vertices(proj_verts, camera_pos, camera_rotation, shake)

@njit()
def compute_normals_and_visibility(world_verts, faces, camera_pos):
    vet1 = world_verts[faces[:, 1]] - world_verts[faces[:, 0]]
    vet2 = world_verts[faces[:, 2]] - world_verts[faces[:, 0]]
    normals = np.cross(vet1, vet2)
    norms = np.sqrt(np.sum(normals * normals, axis=1))
    #norms[norms == 0] = 1  # Prevent division by zero
    #normals /= norms

    camera_rays = world_verts[faces[:, 0]] - camera_pos
    return normals, camera_rays