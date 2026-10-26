"""
The README's demo: the real window in a hidden GL context, on a fixed 30 fps clock, driven by a scripted cursor that
sends real mouse events. Each frame is read back from the framebuffer and piped to ffmpeg, losslessly, then encoded as
docs/demo.mp4 (H.264) and docs/demo.gif. Needs a desktop session for the GL context, and ffmpeg.

    python scripts/demo.py

The movie playing inside the panels changes much of every frame, which a GIF stores badly (40 MB at 1280 px). So the
GIF is a second recording with the movie paused on its busiest frame, scaled to the width GitHub shows a README at.

The tour: the Overview story plays; a click on a cell in the background panel moves every trace to that pixel; the
Compression page plays; then the Residual page. About 45 s.
"""

import argparse
import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from imgui_bundle import imgui, immapp, immvision

from imgui_masknmf.pipeline import Settings

from imgui_masknmf_manim import style
from imgui_masknmf_manim.app import App, Request, draw, runner_params

REPO = Path(__file__).resolve().parent.parent
FPS = 30
SIZE = (1280, 920)
WARMUP_FRAMES = 6  # the first frames settle the layout (the canvas sizes itself from the controls under it)
SECONDS_MOVE = 0.7
SECONDS_RIPPLE = 0.45
RAIL_X = 105  # rail button centers at scale 1: the rail is anchored top-left, whatever the window's size
RAIL_Y = {"overview": 120, "raw": 159, "background": 198, "compression": 237, "residual": 276}
PARK = (110, 790)  # empty rail, under its last line: no tooltip there
# (seconds into the recording, what the cursor does, its target)
TOUR = (
    (10.0, "move", ("panel", 1)),  # the background panel, at the second-busiest cell
    (11.2, "click", None),
    (11.6, "move", ("park",)),
    (13.5, "move", ("rail", "compression")),
    (14.5, "click", None),
    (14.9, "expect", "compression"),
    (14.9, "move", ("park",)),
    (31.0, "move", ("rail", "residual")),
    (32.0, "click", None),
    (32.4, "expect", "residual"),
    (32.4, "move", ("park",)),
    (43.5, "end", None),
)
# an arrow cursor, tip at (0, 0), in pixels
ARROW = ((0, 0), (0, 17), (4.2, 13.2), (7.2, 19.6), (9.8, 18.5), (6.9, 12.2), (12.2, 12.2))

GL = ctypes.CDLL("libGL.so.1")
GL.glClearColor.argtypes = [ctypes.c_float] * 4
GL_RGBA, GL_UNSIGNED_BYTE, GL_COLOR_BUFFER_BIT, GL_BACK, GL_PACK_ALIGNMENT = 0x1908, 0x1401, 0x4000, 0x0405, 0x0D05


def read_framebuffer(width: int, height: int) -> np.ndarray:
    """(height, width, 4) uint8 of the back buffer, top row first."""
    rgba = np.empty((height, width, 4), np.uint8)
    GL.glPixelStorei(GL_PACK_ALIGNMENT, 1)
    GL.glReadBuffer(GL_BACK)
    GL.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.ctypes.data_as(ctypes.c_void_p))
    return rgba[::-1]


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def second_cell(signal: np.ndarray, pixel: tuple, radius: float = 12.0) -> tuple:
    """(row, col) where the signal varies most, at least radius px from pixel: a different cell."""
    sd = signal.std(axis=0)
    rows, cols = np.indices(sd.shape)
    sd[(rows - pixel[0]) ** 2 + (cols - pixel[1]) ** 2 < radius**2] = 0
    return tuple(int(v) for v in np.unravel_index(np.argmax(sd), sd.shape))


class Recorder(App):
    """The window on a fixed clock: it draws with manim on the main thread, follows TOUR, and pipes every frame out."""

    def __init__(self, settings: Settings, ffmpeg: subprocess.Popen, movie_still: bool = False):
        super().__init__(settings=settings, scale=1.0, page="overview")
        self.load(self.future_compute.result())
        self.future_compute = None
        self.prepared = True  # nothing builds on the manim thread: this class draws on the main one
        self.playing = False
        if movie_still:
            self.movie_playing = False
            self.frame_clock = float(np.argmax(self.d.signal.reshape(self.d.n_frames, -1).std(axis=1)))
        self.ffmpeg = ffmpeg
        self.target_pixel = second_cell(self.d.signal, self.pixel)
        self.n_drawn = 0
        self.i_event = 0
        self.cursor = PARK
        self.move: Optional[tuple] = None  # (start time, from, to)
        self.release_at: Optional[float] = None
        self.ripple: Optional[tuple] = None  # (start time, where)
        self.done = False

    def delta_time(self) -> float:
        return 1.0 / FPS

    def request_drawing(self, request: Request) -> None:
        if self.drawing is None or self.drawing.request != request:
            self.drawing = draw(self.stage, request)
            self.texture_stale = True

    def point_of(self, target: tuple) -> tuple:
        """Screen point of a tour target: ("park",), ("rail", page) or ("panel", index) at self.target_pixel."""
        if target[0] == "park":
            return PARK
        if target[0] == "rail":
            return (RAIL_X, RAIL_Y[target[1]])
        hit = self.drawing.hits[target[1]]
        row, col = self.target_pixel
        x = hit.left + (col + 0.5) / hit.shape[1] * (hit.right - hit.left)
        y = hit.top + (row + 0.5) / hit.shape[0] * (hit.bottom - hit.top)
        left, top, width, height = self.canvas_rect
        return (left + x * width, top + y * height)

    def follow_tour(self, now: float) -> None:
        """Fire the tour's events that are due, move the cursor, and send imgui the mouse."""
        io = imgui.get_io()
        while self.i_event < len(TOUR) and TOUR[self.i_event][0] <= now:
            _, kind, target = TOUR[self.i_event]
            self.i_event += 1
            if kind == "move":
                self.move = (now, self.cursor, self.point_of(target))
            elif kind == "click":
                io.add_mouse_button_event(0, True)
                self.release_at = now + 2.0 / FPS
                self.ripple = (now, self.cursor)
            elif kind == "expect" and self.page != target:
                raise RuntimeError(f"the click at {now:.1f} s missed: on {self.page}, not {target}")
            elif kind == "end":
                self.done = True
                self.quit()
        if self.release_at is not None and now >= self.release_at:
            io.add_mouse_button_event(0, False)
            self.release_at = None
        if self.move is not None:
            start, a, b = self.move
            s = smoothstep((now - start) / SECONDS_MOVE)
            self.cursor = (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))
        io.add_mouse_pos_event(*self.cursor)

    def draw_cursor(self, now: float) -> None:
        """The cursor and a click's ripple, drawn into the frame: a hidden window has no hardware cursor to record."""
        dl = imgui.get_foreground_draw_list()
        if self.ripple is not None and now - self.ripple[0] < SECONDS_RIPPLE:
            progress = (now - self.ripple[0]) / SECONDS_RIPPLE
            center = imgui.ImVec2(*self.ripple[1])
            dl.add_circle(center, 5 + 20 * progress, imgui.color_convert_float4_to_u32(style.vec4(style.ACCENT, 1 - progress)),
                          0, 2.5)
        points = [imgui.ImVec2(self.cursor[0] + x, self.cursor[1] + y) for x, y in ARROW]
        dl.add_concave_poly_filled(points, imgui.color_convert_float4_to_u32(imgui.ImVec4(1, 1, 1, 1)))
        dl.add_polyline(points, imgui.color_convert_float4_to_u32(imgui.ImVec4(0, 0, 0, 1)), 1.2, imgui.ImDrawFlags_.closed)

    def draw_body(self) -> None:
        if self.n_drawn == WARMUP_FRAMES:
            self.playing = True
        now = (self.n_drawn - WARMUP_FRAMES) / FPS
        if now >= 0 and not self.done:
            self.follow_tour(now)
        super().draw_body()
        self.draw_cursor(now)
        self.n_drawn += 1

    def capture(self) -> None:
        """after_swap: draw imgui's frame again into the back buffer, read it, and send it to ffmpeg."""
        if self.n_drawn <= WARMUP_FRAMES or self.done:
            return
        io = imgui.get_io()
        width = int(io.display_size.x * io.display_framebuffer_scale.x)
        height = int(io.display_size.y * io.display_framebuffer_scale.y)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(*style.BG, 1.0)
        GL.glClear(GL_COLOR_BUFFER_BIT)
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        self.ffmpeg.stdin.write(read_framebuffer(width, height).tobytes())


def record(filepath_lossless: Path, movie_still: bool = False) -> int:
    """Run the tour into a lossless FFV1 file; returns the frames written."""
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{SIZE[0]}x{SIZE[1]}",
         "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(filepath_lossless)],
        stdin=subprocess.PIPE,
    )
    app = Recorder(settings=Settings(), ffmpeg=ffmpeg, movie_still=movie_still)
    params = runner_params(app, hidden=True)
    params.app_window_params.window_geometry.size = SIZE
    params.fps_idling.enable_idling = False
    params.callbacks.after_swap = app.capture
    immapp.run(params)
    app.executor_compute.shutdown(wait=False, cancel_futures=True)
    app.executor_manim.shutdown(wait=False, cancel_futures=True)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the recording")
    return app.n_drawn - WARMUP_FRAMES


def encode_mp4(filepath_lossless: Path, filepath_mp4: Path, crf: int) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(filepath_lossless), "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(crf), "-tune", "animation", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(filepath_mp4)], check=True)


def encode_gif(filepath_lossless: Path, filepath_gif: Path, fps: int, width: int) -> None:
    """
    One palette from every pixel of the movie, so the window's rarer colors (the green status) keep their hue, and
    ordered dithering, so still areas stay still from frame to frame.
    """
    graph = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
             "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(filepath_lossless), "-vf", graph,
                    str(filepath_gif)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--crf", type=int, default=20, help="the mp4's x264 quality: lower is better and bigger")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=1000)
    parser.add_argument("--gif-movie", action="store_true", help="keep the movie playing in the GIF too (much bigger)")
    args = parser.parse_args()
    style.setup_manim()
    immvision.use_rgb_color_order()
    with tempfile.TemporaryDirectory() as dir_tmp:
        filepath_lossless = Path(dir_tmp) / "tour.mkv"
        n_frames = record(filepath_lossless)
        encode_mp4(filepath_lossless, REPO / "docs" / "demo.mp4", args.crf)
        if not args.gif_movie:
            record(filepath_lossless, movie_still=True)
        encode_gif(filepath_lossless, REPO / "docs" / "demo.gif", args.gif_fps, args.gif_width)
    for name in ("demo.mp4", "demo.gif"):
        filepath = REPO / "docs" / name
        print(f"{filepath}: {filepath.stat().st_size / 1e6:.1f} MB")
    print(f"{n_frames} frames, {n_frames / FPS:.1f} s at {FPS} fps, {SIZE[0]} x {SIZE[1]}")


if __name__ == "__main__":
    main()
