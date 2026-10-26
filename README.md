# imgui_masknmf_manim

A window that explains how [masknmf](https://github.com/apasarkar/masknmf-toolbox)'s compression splits a movie into
three movies:

    registered = background + signal + residual

Each page is one [manim](https://www.manim.community) animation, built from the movie itself and drawn by manim's
Cairo camera into an [imgui-bundle](https://github.com/pthom/imgui_bundle) window. A timeline scrubs the animation
forward or back while the movie plays inside its panels. Click a pixel to follow it on every page, or a block on the
Compression page to open that block.

The window is styled like [imgui_lab4_interface](https://github.com/FlynnOConnell/imgui_lab4_interface):
imgui-data-loader's shell, the same palette, Font Awesome icons. The movie, registration and compression come from
imgui_masknmf's own steps, which call masknmf.

![The Overview animation: the registered movie splits into background, signal and residual, then one pixel's four traces](docs/overview.gif)

That is the Overview page's animation, drawn by manim from imgui_masknmf's simulated movie. [docs/demo.mp4](docs/demo.mp4)
is a 45 s tour of the whole window: the Overview, a click that follows another pixel, then the Compression and
Residual pages.

## The three pieces

masknmf's `CompressionArray` rebuilds a registered frame as

    noise SD x (U V) + mean image + spatial trend x temporal trend

and `decomposition.split` names the parts:

| piece | what it is | in masknmf |
|-------|------------|------------|
| background | the mean image plus a slow trend (the maximin spline detrender) | `mean_image + spatial_trend_basis @ temporal_trend_basis` |
| signal | noise SD x U V: a few smooth components per block, blended | `noise_variance_image * (spatial_compressed @ temporal_compressed)` |
| residual | registered - background - signal | `CompressionResidualArray` |

`tests/test_decomposition.py` checks background + signal against `CompressionArray(rescale=True,
include_trend=True)`, and the residual against `CompressionResidualArray`.

On imgui_masknmf's default simulated movie (64 x 64 px, 1800 frames at 30 Hz), the compression has rank 145, and U
and V hold 4.3% of the movie's values. Of the registered movie's variance over time, the background holds 3%, the
signal 44% and the residual 58%; the pieces covary a little, so the shares add up to 104%. A residual pixel
correlates 0.01 with itself one frame later (median over pixels), against 0.49 for the fluctuations it came from.

## Pages

| page | chapters |
|------|----------|
| [Overview](docs/overview.png) | the registered movie shrinks into the first of four panels: = background + signal + residual; one pixel's four traces |
| [Raw movie](docs/raw.png) | a stack of recorded frames; recorded beside registered; one pixel's trace, recorded and registered |
| [Background](docs/background.png) | the mean image; the trend at one pixel; the background movie; registered - background, the fluctuations |
| [Compression](docs/compression.png) | fluctuations / noise SD; the blocks; one block's movie as a matrix Y; Y ~ U x V with the components kept; the rank image and the signal movie |
| [Residual](docs/residual.png) | registered - background - signal; the residual at one pixel against 2 noise SDs; its histogram over every pixel against a normal curve |

Under the animation are, from top to bottom:

- the timeline: drag to scrub; the gaps mark chapters
- the chapters: click one to play from its start
- the current chapter's paragraph, with this movie's numbers in it
- the movie's frame slider, and on Compression the block slider

Keys: space plays and pauses, left and right step chapters, up and down step pages, Esc goes to Overview, Ctrl +,
Ctrl - and Ctrl 0 resize.

## Install

Needs masknmf-toolbox and imgui_masknmf checkouts next to this one (`../masknmf-toolbox`, `../imgui_masknmf`; see
`[tool.uv.sources]` in `pyproject.toml`). imgui_masknmf is not published yet.

```bash
cd imgui_masknmf_manim
uv venv -p 3.12 && uv pip install -e ".[dev]"
imgui-masknmf-manim
```

manim draws with cairo and pango. On Linux, pycairo and manimpango build from source against the system libraries.
No LaTeX is needed: every label is manim `Text`.

`pillow<13` is pinned. With Pillow 13, manim 0.22.0 raises `ValueError: Invalid mode for data type` on any image;
manim fixed this on main (#5074) after 0.22.0.

## Movies

The same as imgui_masknmf: its simulated movie by default, or `imgui-masknmf-manim --movie unit.npz --negative`, or
`--movie movie.tif --fs 30`.

## Tests, screenshots and the demo

```bash
pytest                          # 28 tests, about 25 s; one opens a hidden window
python scripts/screenshot.py    # docs/<page>.png
python scripts/overview_gif.py  # docs/overview.gif, about 20 s; needs ffmpeg
python scripts/demo.py          # docs/demo.mp4, about 70 s
imgui-masknmf-manim --page residual --time 5 --screenshot residual.png
```

`scripts/overview_gif.py` draws the Overview story through manim's camera alone and crops to what it covers.
`scripts/demo.py` records the real window. It runs hidden on a fixed 30 fps clock, a scripted cursor sends imgui real
mouse events, and each frame is read back from the framebuffer and piped to ffmpeg.

## Layout

| file | contents |
|------|----------|
| `src/imgui_masknmf_manim/decomposition.py` | runs imgui_masknmf's steps and splits the movie in three |
| `src/imgui_masknmf_manim/canvas.py` | manim as a drawing engine: panels, traces, `Story`, scrubbing, the camera |
| `src/imgui_masknmf_manim/stories.py` | one story per page, and its notes |
| `src/imgui_masknmf_manim/style.py` | the palette, colormaps, and the font manim shares with the window |
| `src/imgui_masknmf_manim/app.py` | the window |
| `scripts/screenshot.py` | the page screenshots in `docs/` |
| `scripts/overview_gif.py` | `docs/overview.gif` |
| `scripts/demo.py` | `docs/demo.mp4` |

See [ARCHITECTURE.md](ARCHITECTURE.md).
