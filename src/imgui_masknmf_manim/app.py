"""
The `imgui-masknmf-manim` window. Each page animates one part of how masknmf's compression splits a movie,

    registered = background + signal + residual,

with manim, drawn from the movie itself. A timeline scrubs the animation while the movie plays inside its panels.
masknmf runs on one worker thread and manim on another, so the window never waits on either.
"""

import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import imgui_data_loader as idl
import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, immapp, immvision
from imgui_bundle import icons_fontawesome_6 as fa
from PIL import Image

from imgui_masknmf import config
from imgui_masknmf.pipeline import Settings

from imgui_masknmf_manim import __version__, stories, style
from imgui_masknmf_manim import canvas as cv
from imgui_masknmf_manim.decomposition import Decomposition, busiest_pixel, decompose, variance_shares

ACCENT = style.vec4(style.ACCENT)
DIM = style.vec4(style.DIM)
TEXT = style.vec4(style.TEXT)
CARD = style.vec4(style.CARD)
CARD_HOVER = style.vec4(style.CARD_HOVER)
EDGE = style.vec4(style.EDGE)
OK = style.vec4(style.OK)
WARN = style.vec4(style.WARN)
ERROR = style.vec4(style.ERROR)
BG = style.vec4(style.BG)
LIT = style.vec4(style.ACCENT, 0.22)
CLEAR = imgui.ImVec4(0, 0, 0, 0)
COLORS_PIECE = {"background": style.vec4(style.BACKGROUND), "signal": style.vec4(style.SIGNAL),
                "residual": style.vec4(style.RESIDUAL)}

TITLE_WINDOW = "masknmf · manim"
SIZE_WINDOW = (1400, 960)
WIDTH_RAIL_EM = 13
HEIGHT_BELOW_CANVAS_EM = 11.0  # timeline, chapters, note and movie row, until they have been measured once
PAGES = (
    ("overview", fa.ICON_FA_LAYER_GROUP, "Overview", "registered = background + signal + residual"),
    ("raw", fa.ICON_FA_FILM, "Raw movie", "the frames as recorded, then registered"),
    ("background", fa.ICON_FA_IMAGE, "Background", "the mean image plus a slow trend"),
    ("compression", fa.ICON_FA_FILE_ZIPPER, "Compression", "noise SD x U V, block by block: the signal"),
    ("residual", fa.ICON_FA_WAVE_SQUARE, "Residual", "what the compression leaves: close to white noise"),
)
STEPS = (("movie", "load the movie"), ("register", "register"), ("compress", "compress"), ("split", "split it in three"))


@dataclass(frozen=True)
class Request:
    """One picture of the canvas."""

    page: str
    pixel: tuple
    block: int
    t: float  # seconds into the page's story
    frame: int
    width: int  # pixels


@dataclass
class Drawing:
    request: Request
    rgba: np.ndarray  # (height, width, 4) uint8
    hits: list  # canvas.Hit of the clickable panels showing
    notes: list  # one paragraph per chapter
    ms_build: float  # 0 when the story was already built
    ms_draw: float


@dataclass
class Stage:
    """What the manim thread keeps between drawings. Only that thread touches it."""

    d: Decomposition
    stories: dict = field(default_factory=dict)  # page -> (key, Story, notes)
    cameras: dict = field(default_factory=dict)  # width -> Camera


def story_key(page: str, pixel: tuple, block: int) -> tuple:
    """What a story depends on besides the movie: the compression page follows the block, the others the pixel."""
    return (page, block) if page == "compression" else (page, pixel)


def story_for(stage: Stage, page: str, pixel: tuple, block: int) -> tuple:
    """(story, notes, milliseconds spent building) for a page, built only when its pixel or block changed."""
    key = story_key(page, pixel, block)
    cached = stage.stories.get(page)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2], 0.0
    start = time.perf_counter()
    story = stories.build(page, stage.d, pixel, block)
    notes = stories.notes(page, stage.d, pixel, block)
    stage.stories[page] = (key, story, notes)
    return story, notes, 1000 * (time.perf_counter() - start)


def draw(stage: Stage, request: Request) -> Drawing:
    """Put the page's story at the request's time and frame, and draw it through manim's Cairo camera."""
    story, notes, ms_build = story_for(stage, request.page, request.pixel, request.block)
    if request.width not in stage.cameras:
        stage.cameras[request.width] = cv.new_camera(request.width)
    start = time.perf_counter()
    rgba = cv.capture(stage.cameras[request.width], cv.set_story(story, request.t, request.frame))
    return Drawing(request=request, rgba=rgba, hits=cv.hits(story), notes=notes, ms_build=ms_build,
                   ms_draw=1000 * (time.perf_counter() - start))


def prepare(stage: Stage, page: str, pixel: tuple, block: int) -> None:
    """Build a page's story ahead of time, so opening the page doesn't wait for it."""
    story_for(stage, page, pixel, block)


def block_at(blocks: list, row: float, col: float) -> int:
    """The block whose center is nearest (row, col)."""
    distances = [(b.row + b.height / 2 - row) ** 2 + (b.col + b.width / 2 - col) ** 2 for b in blocks]
    return int(np.argmin(distances))


def pick_at(hits: list, x: float, y: float) -> Optional[tuple]:
    """(pick, row, col) under the canvas point (x, y), as fractions of its width and height; None off every panel."""
    for hit in reversed(hits):
        if hit.left <= x < hit.right and hit.top <= y < hit.bottom:
            row = int((y - hit.top) / (hit.bottom - hit.top) * hit.shape[0])
            col = int((x - hit.left) / (hit.right - hit.left) * hit.shape[1])
            return hit.pick, min(row, hit.shape[0] - 1), min(col, hit.shape[1] - 1)
    return None


def chapter_starts(page: str) -> np.ndarray:
    """(chapters + 1,) seconds: where each chapter starts, then where the story ends."""
    return np.concatenate([[0.0], np.cumsum(stories.durations(page))])


def chapter_at(page: str, t: float) -> int:
    starts = chapter_starts(page)
    return int(np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(starts) - 2))


def u32(color: imgui.ImVec4) -> int:
    return imgui.color_convert_float4_to_u32(color)


class App:
    """The window's state, its two worker threads, and the idl.FileDialog shell it draws into."""

    def __init__(self, settings: Settings, scale: float = 1.0, page: str = "overview"):
        self.settings = settings
        self.scale = config.clamp(scale)
        self.scale_applied = 1.0
        self.executor_compute = ThreadPoolExecutor(max_workers=1)
        self.executor_manim = ThreadPoolExecutor(max_workers=1)
        self.future_compute: Optional[Future] = None
        self.future_drawing: Optional[Future] = None
        self.step = ""
        self.error = ""
        self.d: Optional[Decomposition] = None
        self.stage: Optional[Stage] = None
        self.shares: dict = {}
        self.page = page
        self.pixel = (0, 0)
        self.block = 0
        self.t = 0.0
        self.playing = True
        self.frame_clock = 0.0  # frames, fractional while the movie plays
        self.movie_playing = True
        self.drawing: Optional[Drawing] = None
        self.texture_stale = False
        self.height_below: Optional[float] = None  # what the controls under the canvas took last frame
        self.canvas_rect: Optional[tuple] = None  # (left, top, width, height) on screen, last frame
        self.prepared = False
        self.hold: Optional[tuple] = None  # (t, frame) to draw once, then exit: the screenshot mode
        self.frames_shown = 0
        self.config = idl.FileDialogConfig(
            title=TITLE_WINDOW,
            buttons=[],
            theme=style.THEME,
            header_draw=self.draw_header,
            body_draw=self.draw_body,
            footer_draw=self.draw_footer,
            show_options_button=False,
            close_on_select=False,
            quit_on_escape=False,
            on_cancel=self.quit,
        )
        self.dialog = idl.FileDialog(self.config)
        self.future_compute = self.executor_compute.submit(decompose, self.settings, self.set_step)


    def set_step(self, name: str) -> None:
        """Called from the compute thread before each step."""
        self.step = name

    def quit(self) -> None:
        hello_imgui.get_runner_params().app_shall_exit = True

    def poll(self) -> None:
        """Take what the worker threads finished."""
        if self.future_compute is not None and self.future_compute.done():
            future, self.future_compute = self.future_compute, None
            try:
                self.load(future.result())
            except Exception as e:  # shown in the footer: the window stays up to say what failed
                self.error = f"{type(e).__name__}: {e}"
        if self.future_drawing is not None and self.future_drawing.done():
            future, self.future_drawing = self.future_drawing, None
            try:
                self.drawing = future.result()
                self.texture_stale = True
            except Exception as e:
                self.error = f"manim: {type(e).__name__}: {e}"
            if not self.prepared and self.drawing is not None:
                self.prepared = True
                for key, *_ in PAGES:
                    if key != self.page:
                        self.executor_manim.submit(prepare, self.stage, key, self.pixel, self.block)

    def load(self, d: Decomposition) -> None:
        self.d = d
        self.stage = Stage(d=d)
        self.shares = variance_shares(d)
        self.pixel = busiest_pixel(d)
        self.block = block_at(d.blocks, *self.pixel)
        self.t = 0.0
        self.step = ""

    def set_page(self, page: str) -> None:
        if page != self.page:
            self.page = page
            self.t = 0.0
            self.playing = True

    def duration(self) -> float:
        return float(sum(stories.durations(self.page)))

    def toggle_play(self) -> None:
        if not self.playing and self.t >= self.duration():
            self.t = 0.0
        self.playing = not self.playing

    def jump_chapter(self, step: int) -> None:
        """Next or previous chapter; back from inside a chapter first goes to its start, as a media player does."""
        starts = chapter_starts(self.page)
        i = chapter_at(self.page, self.t)
        if step < 0 and self.t - starts[i] > 0.5:
            step = 0
        self.t = float(starts[int(np.clip(i + step, 0, len(starts) - 2))])
        self.playing = True

    def delta_time(self) -> float:
        """Seconds since the last frame: the window's clock. scripts/demo.py steps it at a fixed rate instead."""
        return imgui.get_io().delta_time

    def advance_clocks(self) -> None:
        dt = self.delta_time()
        if self.hold is not None:
            self.t, self.frame_clock = self.hold
            return
        if self.playing:
            self.t = min(self.t + dt, self.duration())
            self.playing = self.t < self.duration()
        if self.movie_playing:
            self.frame_clock = (self.frame_clock + dt * self.d.fs) % self.d.n_frames


    def post_init(self) -> None:
        """Space, arrows and Esc belong to the timeline, not to imgui's keyboard navigation."""
        imgui.get_io().config_flags &= ~imgui.ConfigFlags_.nav_enable_keyboard

    def apply_scale(self) -> None:
        """Font and imgui sizes to self.scale. Called every frame; does work only when the scale moved."""
        if self.scale == self.scale_applied:
            return
        style_imgui = imgui.get_style()
        style_imgui.font_scale_main = self.scale
        style_imgui.scale_all_sizes(self.scale / self.scale_applied)
        self.scale_applied = self.scale

    def keys(self) -> None:
        """Ctrl +/-/0 scale; space plays; left/right step chapters; up/down step pages; Esc goes to the overview."""
        io = imgui.get_io()
        if io.key_ctrl:
            if imgui.is_key_pressed(imgui.Key.equal) or imgui.is_key_pressed(imgui.Key.keypad_add):
                self.scale = config.clamp(self.scale * 1.1)
            elif imgui.is_key_pressed(imgui.Key.minus) or imgui.is_key_pressed(imgui.Key.keypad_subtract):
                self.scale = config.clamp(self.scale / 1.1)
            elif imgui.is_key_pressed(imgui.Key._0) or imgui.is_key_pressed(imgui.Key.keypad0):
                self.scale = config.clamp(config.detect_scale())
            return
        if self.d is None or imgui.is_any_item_active():
            return
        keys_page = [key for key, *_ in PAGES]
        i_page = keys_page.index(self.page)
        if imgui.is_key_pressed(imgui.Key.space):
            self.toggle_play()
        elif imgui.is_key_pressed(imgui.Key.left_arrow):
            self.jump_chapter(-1)
        elif imgui.is_key_pressed(imgui.Key.right_arrow):
            self.jump_chapter(1)
        elif imgui.is_key_pressed(imgui.Key.up_arrow):
            self.set_page(keys_page[max(i_page - 1, 0)])
        elif imgui.is_key_pressed(imgui.Key.down_arrow):
            self.set_page(keys_page[min(i_page + 1, len(keys_page) - 1)])
        elif imgui.is_key_pressed(imgui.Key.escape):
            self.set_page("overview")


    def draw_header(self) -> None:
        """Name, version and the movie's status."""
        em = imgui.get_font_size()
        imgui.dummy(imgui.ImVec2(0, 0.2 * em))
        imgui.push_font(None, 1.5 * em)
        imgui.text_colored(ACCENT, f"{fa.ICON_FA_LAYER_GROUP}  masknmf · manim")
        imgui.pop_font()
        imgui.same_line(0, 0.6 * em)
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 0.45 * em)
        imgui.text_disabled(f"v{__version__}")
        if self.error:
            color, state = ERROR, "failed"
        elif self.d is None:
            color, state = WARN, dict(STEPS).get(self.step, "starting")
        else:
            color, state = OK, f"{self.d.n_frames} frames, split"
        name = self.d.name if self.d is not None else (Path(self.settings.movie.path).name or "simulated movie")
        text = f"{fa.ICON_FA_FILM}  {name}    {fa.ICON_FA_CIRCLE}  {state}"
        imgui.same_line()
        room = imgui.get_content_region_avail().x - imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, room))
        imgui.text_colored(color, text)
        idl.text_wrapped_colored(DIM, "How masknmf's compression splits a movie into a background, a signal and a "
                                      "residual, animated with manim from the movie itself")

    def draw_body(self) -> None:
        """The page rail, then the current page."""
        self.poll()
        self.apply_scale()
        self.keys()
        busy = self.playing or self.movie_playing or self.future_compute is not None or self.future_drawing is not None
        hello_imgui.get_runner_params().fps_idling.enable_idling = not busy
        em = imgui.get_font_size()
        avail = imgui.get_content_region_avail()
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        with imgui_ctx.begin_child("##rail", imgui.ImVec2(WIDTH_RAIL_EM * em, avail.y), 0, imgui.WindowFlags_.no_scrollbar):
            self.draw_rail()
        x = p.x + WIDTH_RAIL_EM * em + 0.4 * em
        dl.add_line(imgui.ImVec2(x, p.y), imgui.ImVec2(x, p.y + avail.y), u32(EDGE), 1.0)
        imgui.same_line(0, 0.8 * em)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(1.2 * em, 0.6 * em))
        with imgui_ctx.begin_child("##content", imgui.ImVec2(0, avail.y), imgui.ChildFlags_.always_use_window_padding,
                                   imgui.WindowFlags_.no_scrollbar):
            self.draw_content()
        imgui.pop_style_var()

    def draw_footer(self, dialog: idl.FileDialog) -> None:
        """Status on the left, Quit on the right."""
        em = imgui.get_font_size()
        imgui.separator()
        imgui.spacing()
        text_quit = f"{fa.ICON_FA_XMARK}  Quit"
        width_quit = imgui.calc_text_size(text_quit).x + 2 * imgui.get_style().frame_padding.x + 1.5 * em
        if not imgui.begin_table("##footer", 2, imgui.TableFlags_.sizing_stretch_prop):
            return
        imgui.table_setup_column("##status", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("##buttons", imgui.TableColumnFlags_.width_fixed, width_quit)
        imgui.table_next_row()
        imgui.table_next_column()
        if self.error:
            idl.text_wrapped_colored(ERROR, f"{fa.ICON_FA_CIRCLE_EXCLAMATION}  {self.error}")
        elif self.d is None:
            idl.text_wrapped_colored(WARN, f"{fa.ICON_FA_SPINNER}  masknmf: {dict(STEPS).get(self.step, 'starting')}")
        elif self.drawing is not None:
            r = self.drawing.request
            built = f"; built the story in {self.drawing.ms_build:.0f} ms" if self.drawing.ms_build else ""
            busy = f"{fa.ICON_FA_SPINNER}  " if self.future_drawing is not None and self.drawing.request.page != self.page else ""
            idl.text_wrapped_colored(DIM, f"{busy}{fa.ICON_FA_CIRCLE_CHECK}  manim drew this {r.width} px wide in "
                                          f"{self.drawing.ms_draw:.0f} ms{built}")
        imgui.table_next_column()
        idl.push_button_style(dialog.theme, primary=False)
        if imgui.button(text_quit, imgui.ImVec2(width_quit, 1.7 * em)):
            dialog.cancel()
        idl.pop_button_style()
        imgui.end_table()

    def draw_rail(self) -> None:
        """Page buttons, then the movie's numbers and what the pages follow."""
        em = imgui.get_font_size()
        width = imgui.get_content_region_avail().x
        imgui.push_style_var(imgui.StyleVar_.button_text_align, imgui.ImVec2(0.0, 0.5))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
        for key, icon, name, hint in PAGES:
            lit = key == self.page
            imgui.push_style_color(imgui.Col_.button, LIT if lit else CLEAR)
            imgui.push_style_color(imgui.Col_.button_hovered, LIT if lit else CARD_HOVER)
            imgui.push_style_color(imgui.Col_.button_active, LIT)
            imgui.push_style_color(imgui.Col_.text, ACCENT if lit else TEXT)
            if imgui.button(f"{icon}   {name}##rail_{key}", imgui.ImVec2(width, 2.2 * em)):
                self.set_page(key)
            imgui.pop_style_color(4)
            if imgui.is_item_hovered():
                idl.wrapped_tooltip(f"{name}: {hint}")
        imgui.pop_style_var(2)
        if self.d is None:
            return
        d = self.d
        imgui.dummy(imgui.ImVec2(0, 0.8 * em))
        imgui.text_disabled("THIS MOVIE")
        imgui.separator()
        imgui.text_colored(DIM, f"{d.n_frames} frames, {d.shape[0]} x {d.shape[1]} px")
        imgui.text_colored(DIM, f"{d.fs:g} Hz, {d.n_frames / d.fs:.0f} s")
        imgui.text_colored(DIM, f"rank {d.rank}, {100 * d.stored_fraction:.1f}% stored")
        imgui.dummy(imgui.ImVec2(0, 0.6 * em))
        imgui.text_disabled("VARIANCE OVER TIME")
        imgui.separator()
        self.draw_shares(width)
        imgui.dummy(imgui.ImVec2(0, 0.6 * em))
        imgui.text_disabled("FOLLOWING")
        imgui.separator()
        imgui.text_colored(ACCENT, f"{fa.ICON_FA_CROSSHAIRS}  pixel ({self.pixel[0]}, {self.pixel[1]})")
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Click a pixel in any panel to follow it on every page.")
        b = d.blocks[self.block]
        imgui.text_colored(ACCENT, f"{fa.ICON_FA_BORDER_ALL}  block {self.block}, rank {int(b.kept.sum())}")
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Click a block on the Compression page, or use the slider under it.")
        busiest = busiest_pixel(d)
        if self.pixel != busiest:
            idl.push_button_style(style.THEME, primary=False)
            if imgui.button(f"{fa.ICON_FA_ROTATE_LEFT}  busiest pixel", imgui.ImVec2(width, 0)):
                self.pixel = busiest
            idl.pop_button_style()
            if imgui.is_item_hovered():
                idl.wrapped_tooltip("Go back to the pixel where the signal varies most.")

    def draw_shares(self, width: float) -> None:
        """One bar per piece: its variance over time as a share of the registered movie's."""
        em = imgui.get_font_size()
        dl = imgui.get_window_draw_list()
        imgui.begin_group()
        for name, color in COLORS_PIECE.items():
            share = self.shares[name]
            imgui.text_colored(color, name)
            text = f"{100 * share:.0f}%"
            imgui.same_line(width - imgui.calc_text_size(text).x)
            imgui.text_colored(DIM, text)
            p = imgui.get_cursor_screen_pos()
            r = 0.15 * em
            dl.add_rect_filled(p, imgui.ImVec2(p.x + width, p.y + 2 * r), u32(CARD_HOVER), r)
            dl.add_rect_filled(p, imgui.ImVec2(p.x + width * min(share, 1.0), p.y + 2 * r), u32(color), r)
            imgui.dummy(imgui.ImVec2(width, 2 * r + 0.25 * em))
        imgui.end_group()
        if imgui.is_item_hovered():
            total = sum(self.shares.values())
            idl.wrapped_tooltip(f"Each piece's variance over time, summed over pixels, as a share of the registered "
                                f"movie's. They add up to {100 * total:.0f}%, not 100%, because the pieces covary a "
                                "little.")


    def draw_content(self) -> None:
        """Page title and subtitle, then the canvas and its controls."""
        em = imgui.get_font_size()
        key, icon, name, hint = next(page for page in PAGES if page[0] == self.page)
        imgui.push_font(None, 1.35 * em)
        imgui.text_colored(ACCENT, f"{icon}  {name}")
        imgui.pop_font()
        idl.text_wrapped_colored(DIM, hint)
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 0.4 * em))
        if self.d is None:
            self.draw_waiting()
            return
        self.advance_clocks()
        self.draw_canvas()
        top = imgui.get_cursor_pos_y()
        self.draw_timeline()
        self.draw_chapters()
        self.draw_note()
        self.draw_movie_row()
        self.height_below = imgui.get_cursor_pos_y() - top

    def draw_waiting(self) -> None:
        """masknmf's steps, the done ones checked, while the movie is split."""
        em = imgui.get_font_size()
        imgui.dummy(imgui.ImVec2(0, 2 * em))
        names = [key for key, _ in STEPS]
        current = names.index(self.step) if self.step in names else -1
        for i, (key, text) in enumerate(STEPS):
            if self.error and i == current:
                color, icon = ERROR, fa.ICON_FA_CIRCLE_EXCLAMATION
            elif i < current:
                color, icon = OK, fa.ICON_FA_CIRCLE_CHECK
            elif i == current:
                color, icon = WARN, fa.ICON_FA_SPINNER
            else:
                color, icon = DIM, fa.ICON_FA_CIRCLE
            idl.center_text(f"{icon}  {text}", color)

    def draw_canvas(self) -> None:
        """The latest drawing, as big as fits; asks the manim thread for the next one; clicks pick a pixel or block."""
        em = imgui.get_font_size()
        avail = imgui.get_content_region_avail()
        below = self.height_below if self.height_below is not None else HEIGHT_BELOW_CANVAS_EM * em
        width = min(avail.x, (avail.y - below) * 16 / 9)
        width = max(width, 16 * em)
        height = width * 9 / 16
        request = Request(page=self.page, pixel=self.pixel, block=self.block, t=round(self.t, 3),
                          frame=int(self.frame_clock),
                          width=cv.render_width(width * imgui.get_io().display_framebuffer_scale.x))
        self.request_drawing(request)
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail.x - width) / 2)
        corner = imgui.get_cursor_screen_pos()
        self.canvas_rect = (corner.x, corner.y, width, height)
        if self.drawing is None:
            imgui.dummy(imgui.ImVec2(width, height))
            return
        mouse = immvision.image_display("##canvas", self.drawing.rgba, (int(width), int(height)),
                                        refresh_image=self.texture_stale)
        self.texture_stale = False
        if self.hold is not None and self.drawing.request == request:
            self.frames_shown += 1
            if self.frames_shown > 3:
                self.quit()
        if mouse[0] < 0 or self.drawing.request.page != self.page:
            return
        shape = self.drawing.rgba.shape
        picked = pick_at(self.drawing.hits, mouse[0] / shape[1], mouse[1] / shape[0])
        if picked is None:
            return
        pick, row, col = picked
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
        if pick == "block":
            target = block_at(self.d.blocks, row, col)
            idl.wrapped_tooltip(f"block {target}: click to open it")
        else:
            idl.wrapped_tooltip(f"pixel ({row}, {col}): click to follow it")
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            if pick == "block":
                self.block = block_at(self.d.blocks, row, col)
            else:
                self.pixel = (row, col)

    def request_drawing(self, request: Request) -> None:
        """Ask the manim thread for this picture, unless it is drawing one already or the last one was this."""
        if self.future_drawing is None and (self.drawing is None or self.drawing.request != request):
            self.future_drawing = self.executor_manim.submit(draw, self.stage, request)

    def draw_timeline(self) -> None:
        """Restart and play, then the story's bar: gaps between chapters, drag anywhere to scrub."""
        em = imgui.get_font_size()
        imgui.dummy(imgui.ImVec2(0, 0.3 * em))
        idl.push_button_style(style.THEME, primary=False)
        if imgui.button(f"{fa.ICON_FA_BACKWARD_STEP}##restart", imgui.ImVec2(2.2 * em, 0)):
            self.t = 0.0
            self.playing = True
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("from the start")
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_PAUSE if self.playing else fa.ICON_FA_PLAY}##play", imgui.ImVec2(2.2 * em, 0)):
            self.toggle_play()
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("play or pause the animation (space); left and right arrows step chapters")
        idl.pop_button_style()
        duration = self.duration()
        text_time = f"{self.t:4.1f} / {duration:.1f} s"
        imgui.same_line()
        width = imgui.get_content_region_avail().x - imgui.calc_text_size(text_time).x - 1.0 * em
        height = imgui.get_frame_height()
        p = imgui.get_cursor_screen_pos()
        imgui.invisible_button("##timeline", imgui.ImVec2(width, height))
        if imgui.is_item_active():
            self.t = float(np.clip((imgui.get_mouse_pos().x - p.x) / width, 0, 1)) * duration
            self.playing = False
        starts = chapter_starts(self.page)
        if imgui.is_item_hovered():
            t_mouse = float(np.clip((imgui.get_mouse_pos().x - p.x) / width, 0, 1)) * duration
            i = chapter_at(self.page, t_mouse)
            idl.wrapped_tooltip(f"{i + 1}. {stories.CHAPTERS[self.page][i][0]}")
        dl = imgui.get_window_draw_list()
        y, r = p.y + height / 2, 0.2 * em
        x_t = p.x + width * self.t / duration
        dl.add_rect_filled(imgui.ImVec2(p.x, y - r), imgui.ImVec2(p.x + width, y + r), u32(CARD_HOVER), r)
        dl.add_rect_filled(imgui.ImVec2(p.x, y - r), imgui.ImVec2(x_t, y + r), u32(ACCENT), r)
        for start in starts[1:-1]:
            x = p.x + width * start / duration
            dl.add_line(imgui.ImVec2(x, y - 2 * r), imgui.ImVec2(x, y + 2 * r), u32(BG), 3.0)
        dl.add_circle_filled(imgui.ImVec2(x_t, y), 0.45 * em, u32(TEXT))
        imgui.same_line(0, 1.0 * em)
        imgui.text_colored(DIM, text_time)

    def draw_chapters(self) -> None:
        """One chip per chapter, the current one lit; a click plays from its start."""
        current = chapter_at(self.page, self.t)
        starts = chapter_starts(self.page)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
        for i, (title, _) in enumerate(stories.CHAPTERS[self.page]):
            text = f"{i + 1}  {title}"
            width = imgui.calc_text_size(text).x + 2 * imgui.get_style().frame_padding.x
            if i:
                imgui.same_line()
                if imgui.get_cursor_pos_x() + width > right:
                    imgui.new_line()
            lit = i == current
            imgui.push_style_color(imgui.Col_.button, LIT if lit else CARD)
            imgui.push_style_color(imgui.Col_.button_hovered, LIT if lit else CARD_HOVER)
            imgui.push_style_color(imgui.Col_.button_active, LIT)
            imgui.push_style_color(imgui.Col_.text, ACCENT if lit else DIM)
            if imgui.button(f"{text}##chapter_{i}"):
                self.t = float(starts[i])
                self.playing = True
            imgui.pop_style_color(4)
        imgui.pop_style_var()

    def draw_note(self) -> None:
        """The current chapter's paragraph, from the drawing so it matches the pixel and block on the canvas."""
        em = imgui.get_font_size()
        notes = self.drawing.notes if self.drawing is not None and self.drawing.request.page == self.page else []
        current = chapter_at(self.page, self.t)
        imgui.dummy(imgui.ImVec2(0, 0.2 * em))
        with imgui_ctx.begin_child("##note", imgui.ImVec2(0, 4.4 * em), 0, imgui.WindowFlags_.no_scrollbar):
            if current < len(notes):
                idl.text_wrapped_colored(TEXT, notes[current])

    def draw_movie_row(self) -> None:
        """Play or pause the movie in the panels, the frame slider, and on Compression the block slider."""
        em = imgui.get_font_size()
        d = self.d
        idl.push_button_style(style.THEME, primary=False)
        if imgui.button(f"{fa.ICON_FA_PAUSE if self.movie_playing else fa.ICON_FA_PLAY}##movie", imgui.ImVec2(2.2 * em, 0)):
            self.movie_playing = not self.movie_playing
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("play or pause the movie inside the panels")
        idl.pop_button_style()
        imgui.same_line()
        imgui.text_colored(DIM, fa.ICON_FA_FILM)
        imgui.same_line()
        width_block = 14 * em if self.page == "compression" else 0.0
        imgui.set_next_item_width(imgui.get_content_region_avail().x - width_block - 6 * em)
        changed, frame = imgui.slider_int("##frame", int(self.frame_clock), 0, d.n_frames - 1, f"frame %d of {d.n_frames}")
        if changed:
            self.frame_clock = float(frame)
            self.movie_playing = False
        imgui.same_line()
        imgui.text_colored(DIM, f"{self.frame_clock / d.fs:5.1f} s")
        if self.page == "compression":
            imgui.same_line()
            imgui.set_next_item_width(imgui.get_content_region_avail().x)
            changed, block = imgui.slider_int("##block", self.block, 0, len(d.blocks) - 1, f"block %d of {len(d.blocks)}")
            if changed:
                self.block = block


def runner_params(app: App, hidden: bool = False) -> hello_imgui.RunnerParams:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = TITLE_WINDOW
    params.app_window_params.window_geometry.size = tuple(int(v * min(app.scale, 1.5)) for v in SIZE_WINDOW)
    params.app_window_params.hidden = hidden
    params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
    params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6  # hello_imgui loads 4 otherwise
    params.ini_disable = True
    params.fps_idling.fps_idle = 20
    app.dialog.apply_host_theme(params)
    params.callbacks.show_gui = app.dialog.render
    params.callbacks.post_init = app.post_init
    return params


def run() -> None:
    """Open the window and block until it is closed."""
    parser = argparse.ArgumentParser(prog="imgui-masknmf-manim", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--scale", type=float, default=0.0, help="UI scale, e.g. 2 on a 2x HiDPI panel (default: imgui_masknmf's, then detected)")
    parser.add_argument("--movie", default="", help="a .npz from imgui_masknmf's scripts/export_mesc_movie.py, or a .npy / .tif movie")
    parser.add_argument("--fs", type=float, default=0.0, help="the movie's frame rate in Hz, for a .npy or .tif")
    parser.add_argument("--negative", action="store_true", help="the indicator dims with activity (ASAP, JEDI)")
    parser.add_argument("--page", default="overview", choices=[key for key, *_ in PAGES])
    parser.add_argument("--screenshot", default="", help="draw the page in a hidden window, save this PNG, exit")
    parser.add_argument("--time", type=float, default=-1.0, help="with --screenshot: seconds into the story (default: its end)")
    parser.add_argument("--frame", type=int, default=200, help="with --screenshot: the movie frame")
    args = parser.parse_args()
    settings = Settings()
    if args.movie:
        settings.movie.path = str(Path(args.movie).expanduser())
        settings.movie.source = "recorded" if args.movie.endswith(".npz") else "file"
        settings.movie.duration_s = 600.0
        if args.fs > 0:
            settings.movie.fs = args.fs
        settings.movie.negative = args.negative
    style.setup_manim()
    idl.ensure_assets()
    immvision.use_rgb_color_order()
    app = App(settings=settings, scale=config.ui_scale(args.scale), page=args.page)
    if args.screenshot:
        app.hold = (args.time if args.time >= 0 else app.duration(), float(args.frame))
        app.playing = app.movie_playing = False
    immapp.run(runner_params(app, hidden=bool(args.screenshot)))
    app.executor_compute.shutdown(wait=False, cancel_futures=True)
    app.executor_manim.shutdown(wait=False, cancel_futures=True)
    if args.screenshot:
        image = np.asarray(hello_imgui.final_app_window_screenshot())
        Image.fromarray(np.ascontiguousarray(image[..., :3])).save(args.screenshot)
        print(f"{args.screenshot}: {app.page} at {app.hold[0]:.1f} s, frame {int(app.hold[1])}")


if __name__ == "__main__":
    run()
