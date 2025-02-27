import sdl2
import sdl2.ext
from .OPCClient import _OPCClient

# from Canvas import _Canvas
from .RenderContext import _RenderContext
import numpy as np


class _LEDEngine:
    def __init__(self):
        self._CLIENT = _OPCClient()
        self._networked = False
        self._render_context = _RenderContext()
        # self._canvas_manager = _CanvasManager(self)

        # GRID HARDWARE SETTINGS
        self._brightness = 1

        # If self._networked then send self._pixels to the grid with our desired self._orientation
