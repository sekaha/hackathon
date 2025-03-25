import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

# Initialize Pygame
pygame.init()

# Set up the display with OpenGL
width, height = 800, 600
pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Pygame and OpenGL Shapes")

# Configure OpenGL settings
glClearColor(0, 0, 0, 1)  # Black background
glEnable(GL_BLEND)  # Enable alpha blending for transparency
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0, width, 0, height)  # Set orthographic projection matching window coordinates
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

# Create Pygame surface for the circle (with transparency)
circle_surface = pygame.Surface((width, height), pygame.SRCALPHA)
pygame.draw.circle(circle_surface, (255, 0, 0, 255), (200, 300), 100)  # Red circle
circle_surface = pygame.transform.flip(circle_surface, False, True)  # Flip vertically for OpenGL
circle_data = pygame.image.tostring(circle_surface, 'RGBA', 0)

# Create Pygame surface for the square (with transparency)
square_surface = pygame.Surface((width, height), pygame.SRCALPHA)
pygame.draw.rect(square_surface, (0, 0, 255, 255), (600, 300, 100, 100))  # Blue square
square_surface = pygame.transform.flip(square_surface, False, True)  # Flip vertically
square_data = pygame.image.tostring(square_surface, 'RGBA', 0)

# Create OpenGL texture for the circle
circle_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, circle_tex)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, circle_data)

# Create OpenGL texture for the square
square_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, square_tex)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, square_data)

# Main rendering loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw the circle texture (bottom layer)
    glColor4f(1, 1, 1, 1)  # White to preserve texture colors
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, circle_tex)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)          # Bottom-left
    glTexCoord2f(1, 0); glVertex2f(width, 0)      # Bottom-right
    glTexCoord2f(1, 1); glVertex2f(width, height) # Top-right
    glTexCoord2f(0, 1); glVertex2f(0, height)     # Top-left
    glEnd()

    # Draw the triangle with OpenGL (middle layer)
    glDisable(GL_TEXTURE_2D)
    glColor4f(0, 1, 0, 1)  # Green, opaque
    glBegin(GL_TRIANGLES)
    glVertex2f(400, 200)  # Bottom vertex
    glVertex2f(500, 400)  # Top-right vertex
    glVertex2f(300, 400)  # Top-left vertex
    glEnd()

    # Draw the square texture (top layer)
    glColor4f(1, 1, 1, 1)  # White for texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, square_tex)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(width, 0)
    glTexCoord2f(1, 1); glVertex2f(width, height)
    glTexCoord2f(0, 1); glVertex2f(0, height)
    glEnd()

    # Update the display
    pygame.display.flip()

# Cleanup and exit
pygame.quit()
sys.exit()