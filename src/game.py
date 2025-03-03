from LED import *
from src.camera import camera

class GameState:
    def __init__(self):
        self.score = 0
        self.game_time = 0
        self.smooth_delta = 1
        self.asteroids = set()
        self.bullets = set()
        self.particles = set()
        self.to_remove = set()

    def update(self):
        camera.update()
        self.smooth_delta = 0.99 * self.smooth_delta + get_delta(120) * 0.01
        self.game_time += get_delta()
            
        self.bullets -= game.to_remove
        self.particles -= game.to_remove
        self.asteroids -= game.to_remove
        self.to_remove.clear()

    def reset(self):
        self.__init__()

game = GameState()