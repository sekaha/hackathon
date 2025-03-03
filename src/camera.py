from utils.quaternion import Quaternion
import numpy as np

class Camera:
    def __init__(self):
        self.pos = np.zeros(3, dtype=np.float64)
        self.shake = 0
        self.angle = Quaternion.identity()

    def update(self):
        self.shake *= 0.96

    @property
    def rotation_matrix(self):
        return self.angle.rotation_matrix
    
    def add_shake(self, amount):
        self.shake = min(self.shake + amount, 1)

camera = Camera()