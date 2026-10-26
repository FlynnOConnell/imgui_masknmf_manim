import numpy as np
import pytest

from imgui_masknmf_manim import canvas as cv
from imgui_masknmf_manim import stories


@pytest.mark.parametrize("page", list(stories.CHAPTERS))
def test_scrubbing_lands_where_a_fresh_story_does(page, d, pixel, manim_ready):
    """Random jumps back and forth, then a time: the picture matches a story set straight to that time."""
    block = 7
    camera = cv.new_camera(cv.WIDTHS_RENDER[0])
    s = stories.build(page, d, pixel, block)
    assert s.duration == pytest.approx(sum(stories.durations(page)))
    rng = np.random.default_rng(0)
    starts = np.concatenate([[0.0], np.cumsum(stories.durations(page))])
    for t in [0.0, *(0.5 * (starts[1:] + starts[:-1])), s.duration]:
        for jump in rng.uniform(0, s.duration, 4):
            cv.set_story(s, jump, 100)
        scrubbed = cv.capture(camera, cv.set_story(s, t, 200))
        fresh = cv.capture(camera, cv.set_story(stories.build(page, d, pixel, block), t, 200))
        assert np.array_equal(scrubbed, fresh), f"{page} at {t:.2f} s"


@pytest.mark.parametrize("page", list(stories.CHAPTERS))
def test_one_note_per_chapter(page, d, pixel):
    assert len(stories.notes(page, d, pixel, 7)) == len(stories.CHAPTERS[page])


def test_nothing_shows_at_time_zero_but_the_first_animations(d, pixel, manim_ready):
    """Every mobject comes in through an animation, so the overview at 0 s is the background alone."""
    camera = cv.new_camera(cv.WIDTHS_RENDER[0])
    s = stories.build("overview", d, pixel, 7)
    picture = cv.capture(camera, cv.set_story(s, 0.0, 0))
    assert np.array_equal(picture, cv.capture(camera, []))


def test_movie_frame_changes_the_panels(d, pixel, manim_ready):
    camera = cv.new_camera(cv.WIDTHS_RENDER[0])
    s = stories.build("overview", d, pixel, 7)
    first = cv.capture(camera, cv.set_story(s, s.duration, 0))
    later = cv.capture(camera, cv.set_story(s, s.duration, 300))
    assert not np.array_equal(first, later)
