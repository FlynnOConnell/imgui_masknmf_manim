import pytest

from imgui_masknmf.pipeline import Settings

from imgui_masknmf_manim import style
from imgui_masknmf_manim.decomposition import busiest_pixel, run_steps, split


@pytest.fixture(scope="session")
def run():
    """imgui_masknmf's default simulated movie, registered and compressed (about 3 s)."""
    return run_steps(Settings())


@pytest.fixture(scope="session")
def d(run):
    return split(run)


@pytest.fixture(scope="session")
def manim_ready():
    style.setup_manim()


@pytest.fixture(scope="session")
def pixel(d):
    return busiest_pixel(d)
