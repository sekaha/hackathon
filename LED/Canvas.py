import numpy as np
import math
import sdl2
import sdl2.ext
from .Drawable import _Drawable
from PIL import Image
from .RenderContext import _RenderContext
from .constants import *
from OpenGL import GL


# Stuff to do (in order):
# Make it so canvases work as a render target
# Lock them when used as an array by the user, seamlessly ofc
# Then add ndarray conversion
# Export as image
class _Canvas(_Drawable):
    def __init__(
        self,
        context: _RenderContext,
        width: int,
        height: int,
        origin_x: float = 0,
        origin_y: float = 0,
    ):
        super().__init__(context, origin_x, origin_y)
        self._width = width
        self._height = height

        self._angle = 0
        self._cnv_scale = 1
        self._scaled8x = None

        self._fbo, self._texture = self._create_fbo()
        self._pbo = self._create_pbo()

        # This is triggered when the user tries to use the canvas as an array
        # from this point forth, it will do extra operations to give performance, but takes more memory
        self._unlocked = False

    def _create_fbo(self):
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            self._width,
            self._height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0
        )

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return fbo, texture

    def _create_pbo(self):
        pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
        GL.glBufferData(
            GL.GL_PIXEL_PACK_BUFFER,
            self._width * self._height * 4,
            None,
            GL.GL_STREAM_READ,
        )
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

        return pbo

    def _bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, self._width, self._height)

    def _unbind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def __del__(self):
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteTextures(1, [self.texture])
        GL.glDeleteBuffers(1, [self.pbo])

    # User accessible functions
    def center_origin(self) -> None:
        self.origin_x = self.width / 2
        self.origin_y = self.height / 2

    def set_origin_x(self, x):
        self.origin_x = x

    def set_origin_y(self, y):
        self.origin_y = y
