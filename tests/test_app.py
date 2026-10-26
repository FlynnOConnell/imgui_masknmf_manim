import os
import subprocess
import sys

import numpy as np
import pytest

from imgui_masknmf_manim.app import block_at, chapter_at, chapter_starts, pick_at
from imgui_masknmf_manim.canvas import Hit
from imgui_masknmf_manim import stories


def test_pick_maps_canvas_fractions_to_pixels():
    hits = [Hit(pick="pixel", left=0.2, top=0.1, right=0.4, bottom=0.5, shape=(64, 64))]
    assert pick_at(hits, 0.2, 0.1) == ("pixel", 0, 0)
    assert pick_at(hits, 0.2 + 0.2 * 32.5 / 64, 0.1 + 0.4 * 32.5 / 64) == ("pixel", 32, 32)
    assert pick_at(hits, 0.3999, 0.4999) == ("pixel", 63, 63)
    assert pick_at(hits, 0.5, 0.3) is None


def test_pick_takes_the_topmost_panel():
    hits = [Hit(pick="pixel", left=0.0, top=0.0, right=1.0, bottom=1.0, shape=(10, 10)),
            Hit(pick="block", left=0.5, top=0.5, right=1.0, bottom=1.0, shape=(64, 64))]
    assert pick_at(hits, 0.75, 0.75)[0] == "block"
    assert pick_at(hits, 0.25, 0.25)[0] == "pixel"


def test_block_at_picks_the_nearest_center(d):
    for i in (0, len(d.blocks) // 2, len(d.blocks) - 1):
        b = d.blocks[i]
        assert block_at(d.blocks, b.row + b.height / 2, b.col + b.width / 2) == i


@pytest.mark.parametrize("page", list(stories.CHAPTERS))
def test_chapter_at_follows_the_chapter_starts(page):
    starts = chapter_starts(page)
    assert chapter_at(page, 0.0) == 0
    for i, start in enumerate(starts[:-1]):
        assert chapter_at(page, float(start) + 1e-6) == i
    assert chapter_at(page, float(starts[-1])) == len(starts) - 2


@pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")), reason="needs a display")
def test_window_draws_a_page(tmp_path):
    """The real window, hidden: it computes, draws through manim and saves a screenshot."""
    filepath_png = tmp_path / "residual.png"
    out = subprocess.run([sys.executable, "-m", "imgui_masknmf_manim.app", "--scale", "1", "--page", "residual",
                          "--screenshot", str(filepath_png)], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    from PIL import Image

    image = np.asarray(Image.open(filepath_png))
    assert image.shape[0] > 500 and image.std() > 10
