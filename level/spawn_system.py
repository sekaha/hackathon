import numpy as np
from src.game import game
from asteroid import Asteroid
from utils.math_utils import get_random_spherical
from utils.quaternion import Quaternion
from random import randint
from player import player

class SpawnSystem:
    def __init__(self):
        self.asteroid_count = 33

    def update(self):
        if len(game.asteroids) < self.asteroid_count:
            a = Asteroid()
            game.asteroids.add(a)