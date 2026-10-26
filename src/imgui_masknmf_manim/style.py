"""
Colors and type, shared by the window and the manim canvas: imgui_lab4_interface's palette, one color per piece of
the movie, and colormaps that fade into the window's background.
"""

from pathlib import Path

import imgui_bundle
import imgui_data_loader as idl
import manim as mn
import manimpango
import numpy as np
from imgui_bundle import imgui
from matplotlib.colors import LinearSegmentedColormap

# imgui_lab4_interface's palette
ACCENT = (0.40, 0.68, 1.00)
KEY = (1.00, 0.80, 0.20)
DIM = (0.62, 0.62, 0.65)
TEXT = (0.92, 0.92, 0.94)
CARD = (0.17, 0.18, 0.21)
CARD_HOVER = (0.21, 0.22, 0.26)
EDGE = (0.35, 0.35, 0.37)
OK = (0.40, 0.90, 0.40)
WARN = (1.00, 0.75, 0.30)
ERROR = (1.00, 0.40, 0.40)
BG = idl.Theme.dark().bg[:3]

# one color per piece, the same in the window and on the canvas
RAW = TEXT
BACKGROUND = (0.45, 0.82, 0.74)
SIGNAL = KEY
RESIDUAL = (0.78, 0.58, 1.00)
PICK = ACCENT

THEME = idl.Theme.dark().replace(
    accent=(*ACCENT, 1.0),
    accent_hover=(0.50, 0.75, 1.00, 1.0),
    accent_active=(0.32, 0.60, 0.95, 1.0),
    text=(*TEXT, 1.0),
    text_dim=(*DIM, 1.0),
    border=(*EDGE, 0.8),
    button_face=(*CARD, 1.0),
    button_face_hover=(*CARD_HOVER, 1.0),
)

# a movie's darkest pixels are the canvas background; a signed movie's 0 is a card, so its edges show where it is 0
COLORMAPS = {
    "movie": LinearSegmentedColormap.from_list("movie", [BG, (0.55, 0.56, 0.60), (0.97, 0.97, 0.98)]),
    "diverging": LinearSegmentedColormap.from_list("diverging", [(0.30, 0.58, 1.00), CARD, (1.00, 0.66, 0.22)]),
    "sequential": LinearSegmentedColormap.from_list("sequential", [BG, (0.20, 0.38, 0.66), ACCENT, (0.86, 0.94, 1.00)]),
}
LUTS = {name: (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8) for name, cmap in COLORMAPS.items()}

FILEPATH_FONT = Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "DroidSans.ttf"
FONT = "Droid Sans"  # hello_imgui's default font: the canvas reads like the window around it
DIR_MANIM_CACHE = Path.home() / ".cache" / "imgui_masknmf_manim"  # manim writes each Text's SVG here


def vec4(color: tuple, alpha: float = 1.0) -> imgui.ImVec4:
    return imgui.ImVec4(color[0], color[1], color[2], alpha)


def manim_color(color: tuple) -> mn.ManimColor:
    return mn.ManimColor.from_rgb(color)


def setup_manim() -> None:
    """Register the window's font with Pango, make it manim's default, and keep manim's files out of the working directory."""
    mn.config.media_dir = str(DIR_MANIM_CACHE)
    manimpango.register_font(str(FILEPATH_FONT))
    mn.Text.set_default(font=FONT, color=manim_color(TEXT))
