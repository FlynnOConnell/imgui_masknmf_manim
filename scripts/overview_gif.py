"""
The README's GIF: the Overview page's manim animation alone, without the window around it. The story is drawn
straight through manim's Cairo camera on a fixed clock while the movie plays in its panels, then cropped to what the
animation ever covers, and encoded with ffmpeg. Writes docs/overview.gif.

    python scripts/overview_gif.py
    python scripts/overview_gif.py --still    # the movie held on its busiest frame: a smaller GIF
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from imgui_masknmf.pipeline import Settings

from imgui_masknmf_manim import canvas as cv
from imgui_masknmf_manim import stories, style
from imgui_masknmf_manim.decomposition import busiest_pixel, decompose

REPO = Path(__file__).resolve().parent.parent
FPS = 20  # a GIF's frame delays are in 1/100 s, so 20 fps is exact
MOVIE_HZ = 10  # the movie moves on every other frame, at its own speed: the noisy panels cost half as much
SECONDS_HOLD = 3.0  # after the story ends, with the movie still playing
SECONDS_FADE = 0.6  # to the background at the end, so the loop restarts from an empty canvas without a jump
PAD = 14  # px kept around everything the animation draws


def record(filepath_lossless: Path, width: int, still: bool) -> tuple:
    """Draw every frame into a lossless FFV1 file; returns (frames, (top, bottom, left, right) the animation covers)."""
    d = decompose(Settings())
    story = stories.build("overview", d, busiest_pixel(d), 0)
    camera = cv.new_camera(width)
    background = cv.capture(camera, [])[..., :3]
    height = background.shape[0]
    frame_still = int(np.argmax(d.signal.reshape(d.n_frames, -1).std(axis=1)))
    n_frames = int(round((story.duration + SECONDS_HOLD) * FPS))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-c:v", "ffv1", str(filepath_lossless)],
        stdin=subprocess.PIPE,
    )
    top, bottom, left, right = height, 0, width, 0
    for i in range(n_frames):
        t = i / FPS
        frame_movie = frame_still if still else int(np.floor(t * MOVIE_HZ) / MOVIE_HZ * d.fs) % d.n_frames
        rgb = cv.capture(camera, cv.set_story(story, min(t, story.duration), frame_movie))[..., :3]
        drawn = np.any(rgb != background, axis=-1)
        rows, cols = np.flatnonzero(drawn.any(axis=1)), np.flatnonzero(drawn.any(axis=0))
        if rows.size:
            top, bottom = min(top, rows[0]), max(bottom, rows[-1])
            left, right = min(left, cols[0]), max(right, cols[-1])
        fade = min(max((t - (n_frames / FPS - SECONDS_FADE)) / SECONDS_FADE, 0.0), 1.0)
        if fade > 0:
            rgb = np.round((1 - fade) * rgb + fade * background).astype(np.uint8)
        ffmpeg.stdin.write(rgb.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the frames")
    return n_frames, (max(top - PAD, 0), min(bottom + PAD, height - 1), max(left - PAD, 0), min(right + PAD, width - 1))


def encode_gif(filepath_lossless: Path, filepath_gif: Path, box: tuple) -> None:
    """
    Cropped to box, with one palette from every pixel and no dithering. The panels are blocks of flat color (each movie
    pixel is several screen pixels), and dithering would break them into patterns that cost a third more.
    """
    top, bottom, left, right = box
    width, height = (right - left + 1) // 2 * 2, (bottom - top + 1) // 2 * 2
    graph = (f"crop={width}:{height}:{left}:{top},split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
             "[b][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(filepath_lossless), "-vf", graph,
                    str(filepath_gif)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--width", type=int, default=1100, help="manim's frame in px, before the crop")
    parser.add_argument("--still", action="store_true", help="hold the movie on its busiest frame")
    args = parser.parse_args()
    style.setup_manim()
    filepath_gif = REPO / "docs" / "overview.gif"
    with tempfile.TemporaryDirectory() as dir_tmp:
        filepath_lossless = Path(dir_tmp) / "overview.mkv"
        n_frames, box = record(filepath_lossless, args.width, args.still)
        encode_gif(filepath_lossless, filepath_gif, box)
    top, bottom, left, right = box
    print(f"{filepath_gif}: {filepath_gif.stat().st_size / 1e6:.1f} MB, {right - left + 1} x {bottom - top + 1} px, "
          f"{n_frames} frames at {FPS} fps")


if __name__ == "__main__":
    main()
