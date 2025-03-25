import pygame
from pygame.locals import *
from OpenGL.GL import *

# Initialize Pygame
pygame.init()

# Set up display with OpenGL
screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT)  # Clear screen
    pygame.display.flip()  # Swap buffers

pygame.quit()
