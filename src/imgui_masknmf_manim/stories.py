"""
One story per page: its chapters (title, seconds), a builder that makes it from a Decomposition, the picked pixel and
the picked block, and the notes the window shows under each chapter.

Every mobject but a story's statics is brought in by an animation, so scrubbing back to 0 hides it again.
"""

import manim as mn
import numpy as np

from imgui_masknmf_manim import canvas as cv
from imgui_masknmf_manim import style
from imgui_masknmf_manim.decomposition import Decomposition, lag1_correlation, variance_shares

CHAPTERS = {
    "overview": (("The registered movie", 2.5), ("is three movies added up", 3.5), ("at every pixel", 3.5)),
    "raw": (("A stack of frames", 2.5), ("Registration lines them up", 3.5), ("One pixel over time", 3.5)),
    "background": (("The mean image", 3.0), ("A slow trend", 3.5), ("The background movie", 3.0),
                   ("What's left: fluctuations", 3.5)),
    "compression": (("Divide by the noise", 3.0), ("Cut into blocks", 3.0), ("One block is a matrix Y", 3.0),
                    ("Y is about U times V", 3.5), ("Every block: the signal", 3.5)),
    "residual": (("Subtract both", 3.0), ("Noise at one pixel", 3.5), ("Noise everywhere", 3.5)),
}
HISTOGRAM_BINS = 33
HISTOGRAM_LIMIT = 4.0  # noise SDs


def durations(page: str) -> list:
    return [seconds for _, seconds in CHAPTERS[page]]


def pixel_range(*traces: np.ndarray, symmetric: bool = False) -> tuple:
    """(low, high) that shows these traces, padded; centered on 0 when symmetric."""
    values = np.concatenate([t.ravel() for t in traces])
    low, high = np.percentile(values, [0.2, 99.8])
    if symmetric:
        limit = max(abs(low), abs(high)) * 1.1
        return -limit, limit
    pad = 0.08 * (high - low)
    return low - pad, high + pad


def put(mobject: mn.Mobject, x: float, y: float) -> mn.Mobject:
    return mobject.move_to(np.array([x, y, 0.0]))


def name_under(mobject: mn.Mobject, text: str, color: tuple, font_size: float = 20) -> mn.Text:
    return cv.label(text, color, font_size).next_to(mobject, mn.DOWN, buff=0.14)


def sign(text: str, x: float, y: float) -> mn.Text:
    return put(cv.label(text, style.DIM, 36), x, y)


def trace_axes(d: Decomposition, x: float, y: float, width: float, height: float, y_range: tuple) -> mn.Axes:
    return put(cv.plain_axes(d.n_frames / d.fs, y_range[0], y_range[1], width, height), x, y)


def build_overview(d: Decomposition, pixel: tuple, block: int) -> cv.Story:
    """The registered movie shrinks into the first of four panels: registered = background + signal + residual."""
    low, high = d.range_movie
    limit = d.limit_signed
    shape = d.shape
    big = cv.panel(d.registered, 6.6, 5.4, low, high, "movie")
    put(big.mobject, 0, 0.35)
    xs, y_top, box = (-5.25, -1.75, 1.75, 5.25), 1.45, 2.9
    small = big.mobject.copy().scale_to_fit_height(box)
    if small.width > box:
        small.scale_to_fit_width(box)
    put(small, xs[0], y_top)
    pieces = [
        cv.panel(d.background, box, box, low, high, "movie"),
        cv.panel(d.signal, box, box, -limit, limit, "diverging"),
        cv.panel(d.residual, box, box, -limit, limit, "diverging"),
    ]
    for p, x in zip(pieces, xs[1:]):
        put(p.mobject, x, y_top)
    signs = [sign(text, x, y_top) for text, x in (("=", -3.5), ("+", 0.0), ("+", 3.5))]
    colors = (style.RAW, style.BACKGROUND, style.SIGNAL, style.RESIDUAL)
    names = [name_under(m, text, color) for m, text, color in
             zip([small, *[p.mobject for p in pieces]], ("registered", "background", "signal", "residual"), colors)]

    row, col = pixel
    traces = [d.registered, d.background, d.signal, d.residual]
    traces = [t[:, row, col] for t in traces]
    range_raw = pixel_range(traces[0], traces[1])
    range_signed = pixel_range(traces[2], traces[3], symmetric=True)
    ranges = (range_raw, range_raw, range_signed, range_signed)
    axes = [trace_axes(d, x, -1.75, 3.0, 1.35, r) for x, r in zip(xs, ranges)]
    paths = [cv.trace_mobject(a, t, d.fs, c, 1.4, *r) for a, t, c, r in zip(axes, traces, colors, ranges)]
    cursors = [cv.cursor(a, d.fs, *r) for a, r in zip(axes, ranges)]
    markers = [cv.pixel_marker(m, pixel, shape) for m in [small, *[p.mobject for p in pieces]]]
    caption = put(cv.label(f"pixel ({row}, {col}) over {d.n_frames / d.fs:.0f} s", style.DIM, 18), 0, -2.75)

    seconds = durations("overview")
    chapters = [
        cv.chapter(seconds[0], mn.FadeIn(big.mobject, scale=0.92)),
        cv.chapter(seconds[1], mn.Transform(big.mobject, small),
                   *[mn.GrowFromPoint(p.mobject, big.mobject.get_center()) for p in pieces],
                   *[mn.FadeIn(s) for s in signs], *[mn.FadeIn(n, shift=0.1 * mn.UP) for n in names], lag_ratio=0.12),
        cv.chapter(seconds[2], *[mn.Create(a) for a in axes], *[mn.Create(p) for p in paths],
                   *[mn.FadeIn(m) for m in markers], *[mn.FadeIn(c.line) for c in cursors], mn.FadeIn(caption),
                   lag_ratio=0.04),
    ]
    return cv.story([], chapters, panels=[big, *pieces], cursors=cursors)


def build_raw(d: Decomposition, pixel: tuple, block: int) -> cv.Story:
    """A fanned stack of recorded frames, the recorded and registered movies side by side, then one pixel's traces."""
    low, high = d.range_movie
    shape = d.shape
    recorded = cv.panel(d.recorded, 5.6, 3.9, low, high, "movie")
    registered = cv.panel(d.registered, 5.6, 3.9, low, high, "movie")
    put(recorded.mobject, -3.4, 1.25)
    put(registered.mobject, 3.4, 1.25)
    depths = range(4, 0, -1)
    stack = [cv.image_mobject(d.recorded[int(k * d.n_frames / 6)], 5.6, 3.9, low, high, "movie") for k in depths]
    for depth, m in zip(depths, stack):
        m.move_to(recorded.mobject).shift(0.12 * depth * mn.UR)
    shift_max = float(np.abs(d.shifts).max())
    names = [name_under(recorded.mobject, "as recorded", style.DIM),
             name_under(registered.mobject, f"registered, shifts up to {shift_max:.1f} px", style.RAW)]
    markers = [cv.pixel_marker(p.mobject, pixel, shape) for p in (recorded, registered)]

    row, col = pixel
    pair = (d.recorded[:, row, col], d.registered[:, row, col])
    y_range = pixel_range(*pair)
    axes = trace_axes(d, 0, -2.55, 12.4, 1.5, y_range)
    paths = [cv.trace_mobject(axes, pair[0], d.fs, style.DIM, 1.2, *y_range),
             cv.trace_mobject(axes, pair[1], d.fs, style.RAW, 1.4, *y_range)]
    cursor = cv.cursor(axes, d.fs, *y_range)
    legend = mn.VGroup(cv.label("as recorded", style.DIM, 18), cv.label("registered", style.RAW, 18),
                       cv.label(f"pixel ({row}, {col})", style.PICK, 18)).arrange(mn.RIGHT, buff=0.4)
    legend.next_to(axes, mn.UP, buff=0.08).align_to(axes, mn.LEFT)

    seconds = durations("raw")
    chapters = [
        cv.chapter(seconds[0], *[mn.FadeIn(m, shift=0.15 * mn.DL) for m in stack], mn.FadeIn(recorded.mobject),
                   lag_ratio=0.2),
        cv.chapter(seconds[1], *[mn.FadeOut(m) for m in stack], mn.GrowFromPoint(registered.mobject, recorded.mobject.get_center()),
                   *[mn.FadeIn(n) for n in names], *[mn.FadeIn(m) for m in markers], lag_ratio=0.1),
        cv.chapter(seconds[2], mn.Create(axes), *[mn.Create(p) for p in paths], mn.FadeIn(cursor.line), mn.FadeIn(legend),
                   lag_ratio=0.1),
    ]
    return cv.story([], chapters, panels=[recorded, registered], cursors=[cursor])


def build_background(d: Decomposition, pixel: tuple, block: int) -> cv.Story:
    """The mean image, the slow trend at one pixel, the background movie, then the fluctuations it leaves."""
    low, high = d.range_movie
    limit = d.limit_signed
    shape = d.shape
    xs, y_top, box = (-4.7, 0.0, 4.7), 2.05, (4.2, 2.55)
    registered = cv.panel(d.registered, *box, low, high, "movie")
    mean = cv.panel(d.mean_image, *box, low, high, "movie")
    background = cv.panel(d.background, *box, low, high, "movie")
    fluctuations = cv.panel(d.fluctuations, *box, -limit, limit, "diverging")
    for p, x in zip((registered, mean, background), xs):
        put(p.mobject, x, y_top)
    put(fluctuations.mobject, xs[1], y_top)
    trend_rank = d.trend_spatial.shape[2]
    names = {
        "registered": name_under(registered.mobject, "registered", style.RAW),
        "mean": name_under(mean.mobject, f"mean of {d.n_frames} frames", style.BACKGROUND),
        "background": name_under(background.mobject, "background = mean + trend", style.BACKGROUND),
        "fluctuations": name_under(fluctuations.mobject, "registered - background", style.RAW),
    }

    row, col = pixel
    raw_trace, background_trace = d.registered[:, row, col], d.background[:, row, col]
    fluctuation_trace = raw_trace - background_trace
    range_top = pixel_range(raw_trace, background_trace)
    range_bottom = pixel_range(fluctuation_trace, symmetric=True)
    axes_top = trace_axes(d, 0, -0.95, 12.4, 1.2, range_top)
    axes_bottom = trace_axes(d, 0, -2.95, 12.4, 1.1, range_bottom)
    raw_path = cv.trace_mobject(axes_top, raw_trace, d.fs, style.DIM, 1.2, *range_top)
    background_path = cv.trace_mobject(axes_top, background_trace, d.fs, style.BACKGROUND, 2.6, *range_top)
    fluctuation_path = cv.trace_mobject(axes_bottom, fluctuation_trace, d.fs, style.RAW, 1.3, *range_bottom)
    trend_text = f"its trend, {trend_rank} spline components" if trend_rank else "no trend: the detrender is off"
    legend_top = mn.VGroup(cv.label(f"pixel ({row}, {col})", style.PICK, 18), cv.label("registered", style.DIM, 18),
                           cv.label(f"background: mean + {trend_text}", style.BACKGROUND, 18)).arrange(mn.RIGHT, buff=0.4)
    legend_top.next_to(axes_top, mn.UP, buff=0.06).align_to(axes_top, mn.LEFT)
    legend_bottom = cv.label("registered - background", style.RAW, 18).next_to(axes_bottom, mn.UP, buff=0.06)
    legend_bottom.align_to(axes_bottom, mn.LEFT)
    cursors = [cv.cursor(axes_top, d.fs, *range_top), cv.cursor(axes_bottom, d.fs, *range_bottom)]
    markers = [cv.pixel_marker(p.mobject, pixel, shape) for p in (registered, background)]

    seconds = durations("background")
    chapters = [
        cv.chapter(seconds[0], mn.FadeIn(registered.mobject), mn.FadeIn(names["registered"]),
                   mn.GrowFromPoint(mean.mobject, registered.mobject.get_center()), mn.FadeIn(names["mean"]), lag_ratio=0.2),
        cv.chapter(seconds[1], mn.FadeIn(markers[0]), mn.Create(axes_top), mn.Create(raw_path), mn.Create(background_path),
                   mn.FadeIn(legend_top), mn.FadeIn(cursors[0].line), lag_ratio=0.12),
        cv.chapter(seconds[2], mn.GrowFromPoint(background.mobject, mean.mobject.get_center()),
                   mn.FadeIn(names["background"]), mn.FadeIn(markers[1]), lag_ratio=0.2),
        cv.chapter(seconds[3], mn.FadeOut(mean.mobject), mn.FadeOut(names["mean"]), mn.FadeIn(fluctuations.mobject),
                   mn.FadeIn(names["fluctuations"]), mn.Create(axes_bottom), mn.Create(fluctuation_path),
                   mn.FadeIn(legend_bottom), mn.FadeIn(cursors[1].line), lag_ratio=0.08),
    ]
    return cv.story([], chapters, panels=[registered, mean, background, fluctuations], cursors=cursors)


def build_compression(d: Decomposition, pixel: tuple, block: int) -> cv.Story:
    """Fluctuations over the noise SD, the blocks, one block's Y ~ U V, then the rank image and the signal movie."""
    limit = d.limit_signed
    shape = d.shape
    b = d.blocks[block]
    normalized_frames = d.fluctuations / np.maximum(d.noise_sd, 1e-12)
    y_top, box = 0.9, 3.0
    fluctuations = cv.panel(d.fluctuations, box, box, -limit, limit, "diverging", pick="")
    noise = cv.panel(d.noise_sd, box, box, 0, float(d.noise_sd.max()), "sequential", pick="")
    normalized = cv.panel(normalized_frames, box, box, -3, 3, "diverging", pick="block")
    for p, x in zip((fluctuations, noise, normalized), (-5.0, 0.0, 5.0)):
        put(p.mobject, x, y_top)
    signs = [sign("/", -2.5, y_top), sign("=", 2.5, y_top)]
    names_top = [name_under(fluctuations.mobject, "fluctuations", style.RAW),
                 name_under(noise.mobject, f"noise SD, {d.noise_sd.min():.1f} to {d.noise_sd.max():.1f}", style.DIM),
                 name_under(normalized.mobject, "in noise SDs", style.RAW)]

    big = normalized.mobject.copy().scale_to_fit_height(5.0)
    if big.width > 5.2:
        big.scale_to_fit_width(5.2)
    put(big, -4.25, 0.15)
    outlines = mn.VGroup(*[cv.block_rectangle(big, other, shape, stroke_width=1.0, stroke_opacity=0.5,
                                              color=style.manim_color(style.EDGE)) for other in d.blocks])
    highlight = cv.block_rectangle(big, b, shape, stroke_width=3.5, color=style.manim_color(style.PICK))
    blocks_name = name_under(big, f"{len(d.blocks)} blocks of {d.block_sizes[0]} x {d.block_sizes[1]} px, overlapping by half",
                             style.DIM)

    block_movie = normalized_frames[:, b.row:b.row + b.height, b.col:b.col + b.width]
    y_matrix = block_movie.reshape(d.n_frames, -1).T  ## (block pixels, frames)
    crop = cv.panel(block_movie, highlight.width, highlight.height, -3, 3, "diverging", pick="")
    crop.mobject.move_to(highlight)
    kept = b.kept.astype(bool)
    n_kept = int(kept.sum())
    y_mob = cv.matrix_mobject(y_matrix, 2.4, 3.0, -3, 3, "diverging")
    pieces = [y_mob]
    if n_kept:
        spatial = b.spatial[:, :, kept].reshape(-1, n_kept)  ## (block pixels, kept)
        temporal = b.temporal[kept]  ## (kept, frames)
        temporal = (temporal - temporal.mean(axis=1, keepdims=True)) / np.maximum(temporal.std(axis=1, keepdims=True), 1e-12)
        limit_u = float(np.abs(spatial).max())
        u_mob = cv.matrix_mobject(spatial, min(0.2 * n_kept, 1.2), 3.0, -limit_u, limit_u, "diverging")
        v_mob = cv.matrix_mobject(temporal, 2.0, min(0.2 * n_kept, 1.2), -3, 3, "diverging")
        pieces += [cv.label("~", style.TEXT, 40), u_mob, cv.label("x", style.TEXT, 30), v_mob]
    else:
        pieces += [cv.label("rank 0: nothing smoother than noise", style.DIM, 20)]
    equation = mn.Group(*pieces).arrange(mn.RIGHT, buff=0.22)
    put(equation, 3.1, 0.6)
    names_equation = [name_under(y_mob, f"Y: {y_matrix.shape[0]} px x {d.n_frames} frames", style.RAW, 18)]
    if n_kept:
        names_equation += [name_under(pieces[2], f"U: {n_kept} images", style.SIGNAL, 18),
                           name_under(pieces[4], f"V: {n_kept} traces", style.SIGNAL, 18)]
    rough = int((b.kept & ~b.smooth).sum())
    verdict = cv.label(f"block {block}: kept {n_kept} of {len(kept)} candidates"
                       + (f", {rough} of them rough" if rough else ""), style.DIM, 18)
    put(verdict, 3.1, -1.75)

    rank = cv.panel(d.rank_image.astype(np.float32), box, box, 0, float(d.rank_image.max()), "sequential")
    signal = cv.panel(d.signal, box, box, -limit, limit, "diverging")
    put(rank.mobject, 0.9, 0.6)
    put(signal.mobject, 4.9, 0.6)
    names_bottom = [name_under(rank.mobject, f"components per pixel, up to {int(d.rank_image.max())}", style.DIM, 18),
                    name_under(signal.mobject, "signal = noise SD x U V", style.SIGNAL, 18)]
    totals = put(cv.label(f"rank {d.rank} over every block: U and V hold {100 * d.stored_fraction:.1f}% of the movie",
                          style.SIGNAL, 18), 2.9, -1.75)

    seconds = durations("compression")
    top = [fluctuations.mobject, *signs, noise.mobject, *names_top]
    unroll = mn.FadeTransform(crop.mobject, y_mob)  # draws a copy of y_mob: fade its group out, not y_mob
    chapters = [
        cv.chapter(seconds[0], mn.FadeIn(fluctuations.mobject), mn.FadeIn(names_top[0]), mn.FadeIn(signs[0]),
                   mn.FadeIn(noise.mobject), mn.FadeIn(names_top[1]), mn.FadeIn(signs[1]), mn.FadeIn(normalized.mobject),
                   mn.FadeIn(names_top[2]), lag_ratio=0.12),
        cv.chapter(seconds[1], *[mn.FadeOut(m) for m in top], mn.Transform(normalized.mobject, big), mn.FadeIn(blocks_name),
                   mn.LaggedStart(*[mn.Create(o) for o in outlines], lag_ratio=0.05), mn.Create(highlight), lag_ratio=0.1),
        cv.chapter(seconds[2], mn.FadeIn(crop.mobject, scale=1.12), unroll, mn.FadeIn(names_equation[0]), lag_ratio=0.3),
        cv.chapter(seconds[3], *[mn.FadeIn(p, shift=0.2 * mn.LEFT) for p in pieces[1:]],
                   *[mn.FadeIn(n) for n in names_equation[1:]], mn.FadeIn(verdict), lag_ratio=0.15),
        cv.chapter(seconds[4], mn.FadeOut(mn.Group(*pieces[1:], *names_equation, verdict)), mn.FadeOut(unroll.mobject),
                   mn.FadeOut(outlines), mn.FadeIn(rank.mobject), mn.FadeIn(names_bottom[0]),
                   mn.FadeIn(signal.mobject), mn.FadeIn(names_bottom[1]), mn.FadeIn(totals), lag_ratio=0.08),
    ]
    return cv.story([], chapters, panels=[fluctuations, noise, normalized, crop, rank, signal])


def build_residual(d: Decomposition, pixel: tuple, block: int) -> cv.Story:
    """registered - background - signal = residual, its trace against the noise SD, and its histogram."""
    low, high = d.range_movie
    limit = d.limit_signed
    shape = d.shape
    xs, y_top, box = (-5.4, -1.8, 1.8, 5.4), 1.6, 2.7
    pieces = [cv.panel(d.registered, box, box, low, high, "movie"), cv.panel(d.background, box, box, low, high, "movie"),
              cv.panel(d.signal, box, box, -limit, limit, "diverging"), cv.panel(d.residual, box, box, -limit, limit, "diverging")]
    for p, x in zip(pieces, xs):
        put(p.mobject, x, y_top)
    signs = [sign(text, x, y_top) for text, x in (("-", -3.6), ("-", 0.0), ("=", 3.6))]
    colors = (style.RAW, style.BACKGROUND, style.SIGNAL, style.RESIDUAL)
    names = [name_under(p.mobject, text, color) for p, text, color in
             zip(pieces, ("registered", "background", "signal", "residual"), colors)]

    row, col = pixel
    residual_trace = d.residual[:, row, col]
    sd = float(d.noise_sd[row, col])
    y_range = pixel_range(residual_trace, np.array([-3 * sd, 3 * sd]), symmetric=True)
    axes = trace_axes(d, -2.35, -2.05, 8.6, 1.7, y_range)
    band = mn.Polygon(*cv.axes_points(axes, np.array([0, d.n_frames / d.fs, d.n_frames / d.fs, 0]),
                                      np.array([-2 * sd, -2 * sd, 2 * sd, 2 * sd])),
                      stroke_width=0, fill_color=style.manim_color(style.RESIDUAL), fill_opacity=0.14)
    path = cv.trace_mobject(axes, residual_trace, d.fs, style.RESIDUAL, 1.2, *y_range)
    cursor = cv.cursor(axes, d.fs, *y_range)
    legend = mn.VGroup(cv.label(f"pixel ({row}, {col})", style.PICK, 18), cv.label("residual", style.RESIDUAL, 18),
                       cv.label(f"band: +/- 2 noise SD ({sd:.1f})", style.DIM, 18)).arrange(mn.RIGHT, buff=0.4)
    legend.next_to(axes, mn.UP, buff=0.06).align_to(axes, mn.LEFT)
    marker = cv.pixel_marker(pieces[3].mobject, pixel, shape)

    sample = d.residual[:: max(1, d.n_frames // 300)] / np.maximum(d.noise_sd, 1e-12)
    edges = np.linspace(-HISTOGRAM_LIMIT, HISTOGRAM_LIMIT, HISTOGRAM_BINS + 1)
    density, _ = np.histogram(sample, bins=edges, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    normal = np.exp(-0.5 * centers**2) / np.sqrt(2 * np.pi)
    top = float(max(density.max(), normal.max())) * 1.1
    hist_axes = mn.Axes(x_range=[-HISTOGRAM_LIMIT, HISTOGRAM_LIMIT, 1], y_range=[0, top, top], x_length=3.9, y_length=1.7,
                        tips=False, axis_config={"include_ticks": False, "stroke_width": 1.5,
                                                 "color": style.manim_color(style.EDGE)})
    put(hist_axes, 4.75, -2.05)
    width_bar = hist_axes.c2p(edges[1], 0)[0] - hist_axes.c2p(edges[0], 0)[0]
    bars = mn.VGroup(*[
        mn.Rectangle(width=width_bar * 0.86, height=max(hist_axes.c2p(0, h)[1] - hist_axes.c2p(0, 0)[1], 1e-3),
                     stroke_width=0, fill_color=style.manim_color(style.RESIDUAL), fill_opacity=0.75)
        .move_to(hist_axes.c2p(c, 0), aligned_edge=mn.DOWN) for c, h in zip(centers, density)
    ])
    fine = np.linspace(-HISTOGRAM_LIMIT, HISTOGRAM_LIMIT, 200)
    curve = mn.VMobject(stroke_color=style.manim_color(style.TEXT), stroke_width=2).set_points_smoothly(
        cv.axes_points(hist_axes, fine, np.exp(-0.5 * fine**2) / np.sqrt(2 * np.pi)))
    hist_legend = mn.VGroup(cv.label("residual / noise SD", style.RESIDUAL, 18), cv.label("normal", style.TEXT, 18))
    hist_legend.arrange(mn.RIGHT, buff=0.3).next_to(hist_axes, mn.UP, buff=0.06)

    seconds = durations("residual")
    chapters = [
        cv.chapter(seconds[0], *[m for pair in zip([mn.FadeIn(p.mobject) for p in pieces], [mn.FadeIn(n) for n in names])
                                 for m in pair], *[mn.FadeIn(s) for s in signs], lag_ratio=0.1),
        cv.chapter(seconds[1], mn.FadeIn(marker), mn.Create(axes), mn.FadeIn(band), mn.Create(path), mn.FadeIn(legend),
                   mn.FadeIn(cursor.line), lag_ratio=0.12),
        cv.chapter(seconds[2], mn.Create(hist_axes), mn.LaggedStart(*[mn.GrowFromEdge(bar, mn.DOWN) for bar in bars],
                                                                    lag_ratio=0.04),
                   mn.Create(curve), mn.FadeIn(hist_legend), lag_ratio=0.2),
    ]
    return cv.story([], chapters, panels=pieces, cursors=[cursor])


BUILDERS = {"overview": build_overview, "raw": build_raw, "background": build_background,
            "compression": build_compression, "residual": build_residual}


def build(page: str, d: Decomposition, pixel: tuple, block: int) -> cv.Story:
    return BUILDERS[page](d, pixel, block)


def notes(page: str, d: Decomposition, pixel: tuple, block: int) -> list:
    """One paragraph per chapter, with the movie's numbers in it."""
    shares = variance_shares(d)
    pct = {k: f"{100 * v:.0f}%" for k, v in shares.items()}
    seconds = d.n_frames / d.fs
    trend_rank = d.trend_spatial.shape[2]
    b = d.blocks[block]
    n_kept = int(b.kept.sum())
    rough = int((b.kept & ~b.smooth).sum())
    if page == "overview":
        return [
            f"{d.name}: {d.n_frames} frames of {d.shape[0]} x {d.shape[1]} px at {d.fs:g} Hz ({seconds:.0f} s), "
            "after registration. Click a pixel in any panel to follow it on every page.",
            "masknmf's compression rebuilds every frame as three movies added up. The background is the mean image "
            f"plus a slow trend ({pct['background']} of the movie's variance over time). The signal is a few smooth "
            f"components per block ({pct['signal']}). The residual is everything else ({pct['residual']}).",
            "At the picked pixel, the registered trace is the other three traces added up, frame by frame. Background "
            "and registered share a scale; signal and residual share another.",
        ]
    if page == "raw":
        return [
            f"A movie is a stack of frames: {d.n_frames} of them here, {d.shape[0]} x {d.shape[1]} px each.",
            "masknmf's RigidMotionCorrector shifts each frame onto a template, the median of the first frames refined "
            f"3 times. The largest shift here is {float(np.abs(d.shifts).max()):.1f} px.",
            "One pixel over time is a trace. Without registration, the tissue moves under the pixel and the trace "
            "picks up the motion. Everything after this works on the registered movie.",
        ]
    if page == "background":
        row, col = pixel
        trend_sd = float((d.background[:, row, col] - d.mean_image[row, col]).std())
        fluctuation_sd = float(d.fluctuations[:, row, col].std())
        trend = (f"from a maximin spline detrender: {trend_rank} components, slower than the activity. At the picked "
                 f"pixel it varies by {trend_sd:.1f} (SD), against {fluctuation_sd:.1f} for the fluctuations"
                 if trend_rank else "zero: the detrender is off, so the background is the mean image alone")
        return [
            "The mean image is each pixel's average over the whole movie: the tissue, without its activity.",
            f"The trend is {trend}.",
            f"The background movie is mean image + trend. It holds {pct['background']} of the movie's variance over "
            "time, and almost all of its brightness.",
            "Registered minus background is the fluctuations: activity, neuropil and noise, centered on 0. The "
            "compression works on these.",
        ]
    if page == "compression":
        return [
            "Each pixel's fluctuations are divided by its noise SD, estimated from the lag-1 autocovariance. Now one "
            "unit is one noise SD everywhere, and bright pixels don't dominate the SVD.",
            f"The field of view is cut into blocks of {d.block_sizes[0]} x {d.block_sizes[1]} px that overlap by "
            f"half: {len(d.blocks)} blocks. Pick one with the slider or by clicking it.",
            f"Block {block}'s movie, unrolled, is a matrix Y: one row per pixel ({b.height * b.width}), one column per "
            f"frame ({d.n_frames}).",
            f"A randomized SVD gives up to {len(b.kept)} candidates. Each must be smoother than pure noise in space and "
            f"time. This block kept {n_kept}" + (f", including {rough} rough {'one' if rough == 1 else 'ones'}: masknmf "
                                                 "keeps candidates up to and including the first rough one." if rough else "."),
            f"All blocks together, rank {d.rank}: U ({d.shape[0] * d.shape[1]} x {d.rank}, sparse) and V ({d.rank} x "
            f"{d.n_frames}) hold {100 * d.stored_fraction:.1f}% of the movie's values. Signal = noise SD x U V, with "
            "overlapping blocks blended by pyramid weights.",
        ]
    lag_residual = float(np.median(lag1_correlation(d.residual)))
    lag_fluct = float(np.median(lag1_correlation(d.fluctuations)))
    ratio = float((d.residual / np.maximum(d.noise_sd, 1e-12)).std())
    row, col = pixel
    inside = float(np.mean(np.abs(d.residual[:, row, col]) <= 2 * d.noise_sd[row, col]))
    return [
        "Take the background and the signal away from the registered movie, and the residual is left.",
        f"At the picked pixel, {100 * inside:.0f}% of the residual's frames fall within 2 noise SDs, where white "
        "noise would put 95%. A cell's events left behind would show as runs of frames outside the band.",
        f"Over every pixel, residual / noise SD has an SD of {ratio:.2f} and a bell shape. A pixel's residual "
        f"correlates {lag_residual:.2f} with itself one frame later (median over pixels), against {lag_fluct:.2f} "
        "for the fluctuations: it is close to white noise.",
    ]
