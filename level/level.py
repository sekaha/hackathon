from asteroid import Asteroid
from level.planet import Planet
from level.star import Star
from random import randint, gauss
from src.game import game 
import numpy as np

class Level:
    def __init__(self):
        self.level_number = 0
        self.planets = set()
        self.stars = set()

    def generate(self):
        game.asteroids = set(Asteroid() for _ in range(15)) #range(int(level * 1.2 + 10)))
        self.planets = set()
        
        self.stars = set(Star() for _ in range(100))
        self.create_planets(1, 5)

    def create_planets(self, low, high):
        N = randint(low, high)

        hues = [randint(0, 255)]
        hue_off = 20 # 40
        analogous_range = randint(30, 60)

        # Try harding colors
        if N == 2:
            if randint(0,1) == 0:
                # Analogous shift
                print("Analogous 2")
                hues.append((hues[0] + gauss(analogous_range, hue_off)) % 255)
            else:
                # Complementary shift
                print("Complementary")
                hues.append((hues[0] + gauss(127, hue_off)) % 255)
        elif N == 3:
            type = randint(0, 2)

            if type == 0:
                # Analogous shift
                print("Analogous 3")
                hues.append((hues[0] + gauss(analogous_range//2, hue_off)) % 255)
                hues.append((hues[0] + gauss(analogous_range, hue_off)) % 255)
            elif type == 1:
                # Triadic
                print("Triadic")
                hues.append((hues[0] + gauss(85, hue_off)) % 255)
                hues.append((hues[0] + gauss(170, hue_off)) % 255)
            else:
                # Split Complementary
                print("Split Complementary")
                hues.append((hues[0] + gauss(127 - 30, hue_off)) % 255)
                hues.append((hues[0] + gauss(127 + 30, hue_off)) % 255)

        elif N == 4:
            type = randint(0, 2)

            if type == 0:
                print(f"Analogous 4")
                hues.append((hues[0] + gauss(analogous_range//3, hue_off)) % 255)
                hues.append((hues[0] + gauss(2*analogous_range//3, hue_off)) % 255)
                hues.append((hues[0] + gauss(analogous_range, hue_off)) % 255)
            elif type == 1:
                # Tetradic
                print("Tetradic")
                hues.append((hues[0] + gauss(64, hue_off)) % 255)
                hues.append((hues[0] + gauss(128, hue_off)) % 255)
                hues.append((hues[0] + gauss(192, hue_off)) % 255)
            else:
                # Compound
                print("Compound")
                hues.append((hues[0] + gauss(30, hue_off)) % 255)
                hues.append((hues[0] + gauss(127, hue_off)) % 255)
                hues.append((hues[0] + gauss(180, hue_off)) % 255)
        else:
            if randint(0,1) == 0:
                print(f"Analogous {N}")
                for i in range(N-1):
                    hues.append((hues[-1]  + gauss(analogous_range*((i+1)//N), hue_off)) % 255)
            else:
                print(f"Random")
                for _ in range(N-1):
                    hues.append(randint(0, 255))


        for i in range(N):
            intersecting = True

            while intersecting:
                intersecting = False
                candidate = Planet(hues[i])
                
                for p in self.planets:
                    if np.linalg.norm(candidate.pos - p.pos) < max(candidate.ring_dist, p.ring_dist) * 2:
                        intersecting = True
                        break
                
                if not intersecting:
                    # hue_off = 30

                    # if np.random.random() < 0.5:
                    #     hue = gauss(hue, hue_off) % 256 # Analogous shift
                    # else:
                    #     hue = gauss(hue+128, hue_off) % 256  # Complementary shift
                    
                    self.planets.add(candidate)

level = Level()