# Architecture

## Modules

- `decomposition.py`: `run_steps` runs imgui_masknmf's movie, register and compress steps; `split` turns the
  `Run` into a `Decomposition`. That holds the recorded and registered movies, the three pieces, and what the
  stories draw: the mean image, the trend, the noise SD, imgui_masknmf's `Block`s with every candidate, the rank
  image, and display ranges.
- `canvas.py`: manim used as a drawing engine. `panel` is an `ImageMobject` that shows one frame of a movie;
  `trace_mobject` draws a trace as its min/max per bin past 1500 points. A `Story` is a page's mobjects and one
  animation. `set_story` puts it at a time and a movie frame, and `capture` draws it through a Cairo `Camera` into
  an RGBA array.
- `stories.py`: `CHAPTERS` (titles and seconds), one `build_<page>(d, pixel, block)` per page, and `notes`, one
  paragraph per chapter with the movie's numbers in it.
- `style.py`: imgui_lab4_interface's palette, one color per piece, colormaps, and `setup_manim`. That registers
  hello_imgui's DroidSans with Pango, so the canvas text matches the window's, and points manim's file output at
  `~/.cache/imgui_masknmf_manim`.
- `app.py`: `App`, the window, drawn into an `idl.FileDialog` shell (header, body, footer) as imgui_lab4_interface
  is.

## Threads

Two worker threads, each with one worker:

- **compute** runs `decompose` (imgui_masknmf's steps, torch) once per movie.
- **manim** owns a `Stage`: the decomposition, one built story per page, and one camera per width. Only this thread
  touches manim objects. `draw(stage, request)` builds the page's story if its pixel or block changed, sets it, and
  captures it.

Each frame, `App.draw_canvas` makes a `Request` (page, pixel, block, time, frame, width). If no drawing is in flight
and the last one was for a different request, it submits one. The window always shows the latest finished drawing,
so its own frame rate doesn't wait on manim, which draws at its own pace (15 to 30 ms at 960 px). After the first
drawing lands, the other pages' stories are built ahead of time.

`scripts/demo.py` subclasses `App` at two seams. Its `delta_time` steps a fixed 30 fps clock, and its
`request_drawing` draws on the main thread, so every recorded frame shows the picture for that frame's time.

## Scrubbing manim outside a Scene

A story's animation is nested `AnimationGroup`s: one per chapter, played one after the other. `Succession` only
moves forward, so it is not used. manim doesn't document driving animations by hand like this. Four things differ
from a `Scene`, all handled in `canvas.py`:

1. **What to draw.** `AnimationGroup.mobject` leaves out introducers (`FadeIn`, `Create`) because a Scene adds them
   itself. `set_story` draws every leaf animation's mobject instead.
2. **Begin in time order.** A Scene begins each animation when it starts, from the state the earlier ones left.
   `begin_for_scrubbing` begins the leaves in the order they start, finishing each before the next, then rewinds
   them all. It never calls `AnimationGroup.begin()`. That begins every leaf at once: a later chapter's `FadeIn`
   would take its already faded mobject as the target and stay invisible.
3. **Rewind latest first.** Going back in time, `AnimationGroup.interpolate` rewinds its animations in their own
   order. When two chapters move one mobject, the later chapter's starting state wins. `set_story` sets the leaves
   itself instead: first the leaves not yet started go to 0, latest first; then the started ones are set in the order
   they start. Only leaves whose progress changed are touched.
4. **`FadeTransform` shows both mobjects at 0.** It draws its source at full opacity until it starts. The block it
   lifts is therefore brought in by a `FadeIn` of its own, and the source and target group is faded out with
   `FadeOut(unroll.mobject)`, not by fading the target.

`tests/test_stories.py` checks each page at every chapter's midpoint, after random jumps back and forth: the picture
matches a fresh story set straight to that time, pixel for pixel.

## Costs

Every animation deep-copies its mobject when it begins (a `Transform` up to three times). A `Text`'s glyphs kept the
svgelements paths they were parsed from, so a 26-character label took 15 ms to copy. `canvas.label` drops them,
which brings that to 0.5 ms and the Compression story's build from 1.6 s to 0.5 s. A page's first build in a session
costs more, because Pango renders each label once; manim keeps each text's SVG after that.

`capture` costs 15 to 30 ms at 960 px wide. Cairo draws each `ImageMobject` through a PIL transform and composites
it at full frame, so the images dominate, more so at 1280 and 1600 px (`WIDTHS_RENDER`).

## Gotchas

- `ImageMobject` keeps its opacity in the alpha channel of `pixel_array`. `set_story` replaces a panel's RGB with the
  current frame and keeps the alpha the animation set.
- A trace cursor stores its scene coordinates when it is built. Reading them from the axes each frame fails while
  `Create(axes)` has collapsed them.
- hello_imgui loads Font Awesome 4 unless told otherwise. The icons here are Font Awesome 6, so `runner_params` sets
  `default_icon_font`; without it, icons newer than FA4 draw as "?".
- The UI scale comes from imgui_masknmf's `config.ui_scale` (its saved scale, `IMGUI_MASKNMF_SCALE`, or the
  detected one). Ctrl +/- changes it for the session without saving.
