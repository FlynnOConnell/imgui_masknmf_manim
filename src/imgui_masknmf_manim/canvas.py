"""
manim as the window's drawing engine. A Story is a page's mobjects and one scrubbable animation; set_story puts it
at a time and a movie frame, and Renderer draws it through manim's Cairo camera into an RGBA array.

A story's animation is nested AnimationGroups (Succession only plays forward), but set_story drives its leaf
animations itself. Scrubbing outside a Scene has four traps, all handled here:

- AnimationGroup.mobject leaves out introducers (Create, FadeIn), which a Scene adds itself at begin(), so what is
  drawn comes from the leaf animations.
- A Scene begins each animation when it starts, so it starts from what the ones before it left. Here every
  animation is begun up front, so begin_for_scrubbing begins them in time order, finishing each, then rewinds them.
  It never calls AnimationGroup.begin(): that begins every leaf at once, so a FadeIn would take its already faded
  mobject as the target and stay invisible.
- Going back in time, AnimationGroup.interpolate rewinds its animations in their own order, so when two chapters
  move one mobject, the later one's start wins. set_story rewinds the latest first.
- FadeTransform.begin() interpolates to 0 before it ghosts its target; the rewind sets it right.
"""

from dataclasses import dataclass, field
from typing import Optional

import manim as mn
import numpy as np

from imgui_masknmf_manim import style

FRAME_WIDTH, FRAME_HEIGHT = 128 / 9, 8.0  # manim's 16:9 frame in scene units
WIDTHS_RENDER = (960, 1280, 1600)
POINTS_TRACE = 1500  # a longer trace is drawn as its min/max per bin


@dataclass
class Panel:
    """An image mobject showing one frame of a movie: the frame follows the window's."""

    mobject: mn.ImageMobject
    frames: Optional[np.ndarray]  # (frames, height, width); None for a still image
    vmin: float
    vmax: float
    cmap: str
    pick: str = ""  # what a click on it picks: "pixel", "block" or nothing


@dataclass
class Cursor:
    """A vertical line on a trace at the window's frame. Like every mobject but the statics, an animation brings it in."""

    line: mn.Line
    bottom: np.ndarray  # (3,) scene point of frame 0 at the lowest value
    top: np.ndarray
    per_frame: np.ndarray  # (3,) how far it moves per frame


@dataclass
class Story:
    statics: list  # drawn under the animation, always
    animation: mn.AnimationGroup
    duration: float  # seconds
    leaves: list  # the leaf animations, in the order they start
    starts: np.ndarray  # (leaves,) seconds
    ends: np.ndarray
    alphas: np.ndarray  # (leaves,) where each leaf was last set
    panels: list = field(default_factory=list)
    cursors: list = field(default_factory=list)


@dataclass
class Hit:
    """Where a clickable panel sits on the canvas, as fractions of its width and height."""

    pick: str
    left: float
    top: float
    right: float
    bottom: float
    shape: tuple


def to_rgba(image: np.ndarray, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    """(height, width, 4) uint8 of an image through one of style.LUTS."""
    index = np.clip((image - vmin) * (255.0 / max(vmax - vmin, 1e-12)), 0, 255).astype(np.uint8)
    return style.LUTS[cmap][index]


def image_mobject(image: np.ndarray, box_width: float, box_height: float, vmin: float, vmax: float,
                  cmap: str) -> mn.ImageMobject:
    """An image with square, unsmoothed pixels, as large as fits in the box."""
    mobject = mn.ImageMobject(to_rgba(image, vmin, vmax, cmap))
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["nearest"])
    mobject.scale_to_fit_height(box_height)
    if mobject.width > box_width:
        mobject.scale_to_fit_width(box_width)
    return mobject


def matrix_mobject(matrix: np.ndarray, width: float, height: float, vmin: float, vmax: float,
                   cmap: str) -> mn.ImageMobject:
    """A matrix as a heatmap stretched to (width, height)."""
    mobject = mn.ImageMobject(to_rgba(matrix, vmin, vmax, cmap))
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["nearest"])
    return mobject.stretch_to_fit_height(height).stretch_to_fit_width(width)


def panel(frames: np.ndarray, box_width: float, box_height: float, vmin: float, vmax: float, cmap: str,
          pick: str = "pixel") -> Panel:
    """A Panel for a movie (frames, height, width) or a still (height, width)."""
    first = frames if frames.ndim == 2 else frames[0]
    mobject = image_mobject(first, box_width, box_height, vmin, vmax, cmap)
    return Panel(mobject=mobject, frames=frames if frames.ndim == 3 else None, vmin=vmin, vmax=vmax, cmap=cmap, pick=pick)


def pixel_point(mobject: mn.Mobject, row: float, col: float, shape: tuple) -> np.ndarray:
    """Scene point at the center of pixel (row, col) of an image mobject showing an array of this (height, width)."""
    size_pixel = mobject.height / shape[0]
    return mobject.get_corner(mn.UL) + (col + 0.5) * size_pixel * mn.RIGHT + (row + 0.5) * size_pixel * mn.DOWN


def pixel_marker(mobject: mn.Mobject, pixel: tuple, shape: tuple) -> mn.Square:
    size_pixel = mobject.height / shape[0]
    marker = mn.Square(side_length=max(size_pixel * 1.8, 0.12), color=style.manim_color(style.PICK), stroke_width=2.5)
    return marker.move_to(pixel_point(mobject, pixel[0], pixel[1], shape))


def block_rectangle(mobject: mn.Mobject, block, shape: tuple, **kwargs_style) -> mn.Rectangle:
    """The outline of an imgui_masknmf Block on an image mobject."""
    size_pixel = mobject.height / shape[0]
    rectangle = mn.Rectangle(width=block.width * size_pixel, height=block.height * size_pixel, **kwargs_style)
    corner = mobject.get_corner(mn.UL) + block.col * size_pixel * mn.RIGHT + block.row * size_pixel * mn.DOWN
    return rectangle.move_to(corner, aligned_edge=mn.UL)


def plain_axes(x_max: float, y_min: float, y_max: float, width: float, height: float) -> mn.Axes:
    """Axes without ticks or tips, x from 0."""
    return mn.Axes(
        x_range=[0, x_max, x_max], y_range=[y_min, y_max, y_max - y_min], x_length=width, y_length=height, tips=False,
        axis_config={"include_ticks": False, "stroke_width": 1.5, "color": style.manim_color(style.EDGE)},
    )


def axes_points(axes: mn.Axes, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, 3) scene points of the coordinates (x, y) on linear axes."""
    origin = axes.c2p(0, 0)
    return origin + np.outer(x, axes.c2p(1, 0) - origin) + np.outer(y, axes.c2p(0, 1) - origin)


def envelope(y: np.ndarray, n_max: int = POINTS_TRACE) -> tuple:
    """(index, y) of a trace with at most n_max points: each bin's min and max, in time order."""
    if y.size <= n_max:
        return np.arange(y.size), y
    edges = np.linspace(0, y.size, n_max // 2 + 1).astype(int)
    index, values = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        chunk = y[start:stop]
        low, high = start + int(np.argmin(chunk)), start + int(np.argmax(chunk))
        for i in sorted((low, high)):
            index.append(i)
            values.append(y[i])
    return np.array(index), np.array(values)


def trace_mobject(axes: mn.Axes, y: np.ndarray, fs: float, color: tuple, stroke_width: float = 1.6,
                  y_min: float = None, y_max: float = None) -> mn.VMobject:
    """A trace over the whole movie on axes, clipped to (y_min, y_max)."""
    index, values = envelope(y)
    if y_min is not None:
        values = np.clip(values, y_min, y_max)
    points = axes_points(axes, index / fs, values)
    return mn.VMobject(stroke_color=style.manim_color(color), stroke_width=stroke_width).set_points_as_corners(points)


def cursor(axes: mn.Axes, fs: float, y_min: float, y_max: float) -> Cursor:
    """
    A cursor on axes, placed from where the axes are now: build it before an animation (Create) collapses them.
    """
    bottom, top = axes_points(axes, np.zeros(2), np.array([y_min, y_max]))
    per_frame = axes_points(axes, np.array([1 / fs]), np.zeros(1))[0] - axes_points(axes, np.zeros(1), np.zeros(1))[0]
    line = mn.Line(bottom, top, color=style.manim_color(style.TEXT), stroke_width=1.5, stroke_opacity=0.8)
    return Cursor(line=line, bottom=bottom, top=top, per_frame=per_frame)


def label(text: str, color: tuple = style.DIM, font_size: float = 20) -> mn.Text:
    """
    Text without the SVG paths its glyphs were parsed from. Every animation deep-copies its mobject, and copying those
    paths made a label 30 times slower to copy.
    """
    mobject = mn.Text(text, font_size=font_size, color=style.manim_color(color))
    for glyph in mobject.get_family():
        if hasattr(glyph, "path_obj"):
            glyph.path_obj = None
    mobject.id_to_vgroup_dict = {}
    return mobject


def chapter(duration: float, *animations: mn.Animation, lag_ratio: float = 0.0) -> mn.AnimationGroup:
    """One chapter of a story: its animations, lasting duration seconds whatever they add up to."""
    return mn.AnimationGroup(mn.AnimationGroup(*animations, lag_ratio=lag_ratio, run_time=duration * 0.8),
                             mn.Wait(run_time=duration))


def leaf_animations(animation: mn.Animation) -> list:
    """The animations of a (nested) group that are not groups, in order."""
    if isinstance(animation, mn.AnimationGroup):
        return [leaf for a in animation.animations for leaf in leaf_animations(a)]
    return [animation]


def leaf_timings(animation: mn.Animation, start: float = 0.0, scale: float = 1.0) -> list:
    """(start, end, leaf) in seconds of every leaf animation of a (nested) group, in the order they start."""
    if not isinstance(animation, mn.AnimationGroup):
        return [(start, start + scale * animation.run_time, animation)]
    inner = scale * animation.run_time / max(animation.max_end_time, 1e-12)  ## a group's run_time rescales its rows
    out = []
    for row in animation.anims_with_timings:
        out += leaf_timings(row["anim"], start + inner * row["start"], inner)
    return sorted(out, key=lambda timing: timing[0])


def begin_for_scrubbing(leaves: list) -> None:
    """Begin leaf animations, in the order they start, as a Scene would, then rewind them all to 0."""
    for leaf in leaves:
        leaf.begin()
        leaf.interpolate(1)
    for leaf in reversed(leaves):
        leaf.interpolate(0)


def story(statics: list, chapters: list, panels: list = (), cursors: list = ()) -> Story:
    """A Story whose chapters play one after the other."""
    animation = mn.AnimationGroup(*chapters, lag_ratio=1.0)
    timings = leaf_timings(animation)
    leaves = [leaf for _, _, leaf in timings]
    begin_for_scrubbing(leaves)
    return Story(statics=list(statics), animation=animation, duration=float(animation.run_time), leaves=leaves,
                 starts=np.array([start for start, _, _ in timings]), ends=np.array([end for _, end, _ in timings]),
                 alphas=np.zeros(len(leaves)), panels=list(panels), cursors=list(cursors))


def leaf_alphas(s: Story, t: float) -> np.ndarray:
    """(leaves,) how far each leaf animation is at time t (seconds)."""
    return np.clip((t - s.starts) / np.maximum(s.ends - s.starts, 1e-9), 0.0, 1.0)


def set_story(s: Story, t: float, frame: int) -> list:
    """
    Put a story at time t (seconds) and the movie at frame; returns the mobjects to draw, bottom first.

    The animation sets each image's opacity (in its alpha channel); the frame then replaces its colors.
    """
    alphas = leaf_alphas(s, t)
    changed = np.flatnonzero(alphas != s.alphas)
    for i in changed[::-1]:
        if alphas[i] == 0:
            s.leaves[i].interpolate(0.0)
    for i in changed:
        if alphas[i] > 0:
            s.leaves[i].interpolate(float(alphas[i]))
    s.alphas = alphas
    for p in s.panels:
        if p.frames is None:
            continue
        rgba = to_rgba(p.frames[min(frame, p.frames.shape[0] - 1)], p.vmin, p.vmax, p.cmap)
        rgba[..., 3] = p.mobject.pixel_array[..., 3]
        p.mobject.pixel_array = rgba
    for c in s.cursors:
        c.line.put_start_and_end_on(c.bottom + frame * c.per_frame, c.top + frame * c.per_frame)
    return [*s.statics, *[leaf.mobject for leaf in leaf_animations(s.animation)]]


def hits(s: Story) -> list:
    """The clickable panels that are showing, where they are on the canvas."""
    out = []
    for p in s.panels:
        if not p.pick or p.mobject.pixel_array[..., 3].mean() < 128 or p.mobject.width < 0.1:
            continue
        left, top = p.mobject.get_corner(mn.UL)[:2]
        right, bottom = p.mobject.get_corner(mn.DR)[:2]
        out.append(Hit(pick=p.pick, left=left / FRAME_WIDTH + 0.5, top=0.5 - top / FRAME_HEIGHT,
                       right=right / FRAME_WIDTH + 0.5, bottom=0.5 - bottom / FRAME_HEIGHT,
                       shape=p.mobject.pixel_array.shape[:2]))
    return out


def render_width(width_on_screen: float) -> int:
    """The smallest of WIDTHS_RENDER at least as wide as the canvas is on screen, in framebuffer pixels."""
    return next((w for w in WIDTHS_RENDER if w >= 0.9 * width_on_screen), WIDTHS_RENDER[-1])


def new_camera(width: int) -> mn.Camera:
    return mn.Camera(pixel_width=width, pixel_height=int(round(width * 9 / 16)), frame_width=FRAME_WIDTH,
                     frame_height=FRAME_HEIGHT, background_color=style.manim_color(style.BG))


def capture(camera: mn.Camera, mobjects: list) -> np.ndarray:
    """(height, width, 4) uint8 of the mobjects through manim's Cairo camera."""
    camera.reset()
    camera.capture_mobjects(mobjects)
    return camera.pixel_array.copy()
