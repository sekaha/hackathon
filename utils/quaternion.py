import numpy as np
from numba import njit

@njit()
def compute_rotation_matrix(w, x, y, z):
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)

@njit()
def normalize_quaternion(w, x, y, z):
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    if norm == 0:
        return w, x, y, z
    return w / norm, x / norm, y / norm, z / norm

# https://en.wikipedia.org/wiki/Quaternion
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        
    # https://www.songho.ca/opengl/gl_quaternion.html
    # atp idk if i should make this a 4x4 with identity
    @property
    def rotation_matrix(self):
        return compute_rotation_matrix(self.w, self.x, self.y, self.z)

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}i, {self.y}j, {self.z}k)"

    def __mul__(self, other):
        if isinstance(other, Quaternion):  # Quaternion multiplication
            new_w = other.w * self.w - self.x * other.x - self.y * other.y - self.z * other.z
            new_x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            new_y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            new_z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(new_w, new_x, new_y, new_z)
        elif isinstance(other, (int, float)):  # Scalar multiplication
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError(f"Multiplication with type {type(other)} not supported.")

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        return self * other.inverse()

    def normalize(self):
        self.w, self.x, self.y, self.z = normalize_quaternion(self.w, self.x, self.y, self.z)

        return self

    @classmethod
    def identity(cls):
        return cls(1, 0, 0, 0)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    # v' = qvq^-1 (that's why we get double rotation)
    def rotate(self, vector):
        """Rotate a vecotr using a this quaternion"""
        return (self * Quaternion(0, *vector) * self.inverse()).get_vector()

    def slerp(self, other, t):
        # Compute the dot product (cosine of the angle)
        dot = self.dot(other)

        # Clamp to avoid numerical errors
        dot = np.clip(dot, -1.0, 1.0)

        # If the dot product is very close to 1, use linear interpolation to avoid numerical issues
        if dot > 0.9995:
            result = self * (1 - t) + other * t
            return result.normalize()

        # Compute the interpolation angles
        theta_0 = np.arccos(dot)  # Angle between quaternions
        theta = theta_0 * t  # Angle at time t

        # Compute the orthogonal quaternion
        other_orthogonal = (other - self * dot).normalize()

        return self * np.cos(theta) + other_orthogonal * np.sin(theta)

    def axis(self):
        sin_theta = np.sqrt(1 - self.w **2)
        
        if sin_theta < 1e-6:
            return np.array([1, 0, 0])

        return self.get_vector() / sin_theta

    # can undo rotation
    def inverse(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z).normalize()

    def get_vector(self):
        return np.array([self.x, self.y, self.z])
    
    # literally from 3b1b video lol
    @staticmethod
    def from_axis_angle(axis, angle):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        sin_half = np.sin(angle / 2)
        
        return Quaternion(np.cos(angle / 2), *(sin_half * axis))
    
    @staticmethod
    def from_euler(yaw, pitch, roll):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        return Quaternion(
            cy * cp * cr + sy * sp * sr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * sr - sy * sp * cr
        )