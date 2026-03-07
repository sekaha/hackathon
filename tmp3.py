import pygame
from pygame.locals import *
from OpenGL.GL import *

WIDTH, HEIGHT = 800, 600
pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
clock = pygame.time.Clock()

# OpenGL init
glViewport(0, 0, WIDTH, HEIGHT)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)  # top-left is (0, 0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glDisable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending

# Main loop
while True:
    for e in pygame.event.get():
        if e.type == QUIT:
            pygame.quit()
            exit()

    # Clear OpenGL buffer
    glClearColor(0, 0, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT)

    # --- OpenGL DRAWING ---
    glBegin(GL_POINTS)
    glColor4f(1, 1, 1, 0.5)
    glVertex2f(400, 300)
    glEnd()

    # Read OpenGL result into a string buffer
    raw_data = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE)
    surf = pygame.image.fromstring(raw_data, (WIDTH, HEIGHT), "RGBA")
    surf = pygame.transform.flip(surf, False, True)  # Flip vertically

    # Switch to 2D surface blitting
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.blit(surf, (0, 0))

    # --- Pygame 2D drawing ---
    font = pygame.font.SysFont("Arial", 32)
    text = font.render("Hello Pygame", True, (255, 255, 255))
    screen.blit(text, (50, 50))

    pygame.display.flip()
    clock.tick(60)
