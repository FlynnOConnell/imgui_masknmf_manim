"""
Screenshots for the README: the window's own framebuffer, from a hidden window (`--screenshot`), one process per
page. Each runs the default simulated movie and saves docs/<page>.png. Needs a desktop session for the GL context.

    python scripts/screenshot.py              # every page
    python scripts/screenshot.py compression  # one
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# page: seconds into its story (the end, except Compression at Y ~ U x V)
SHOTS = {"overview": 9.5, "raw": 9.5, "background": 13.0, "compression": 11.0, "residual": 10.0}


def main() -> None:
    pages = sys.argv[1:] or list(SHOTS)
    for page in pages:
        filepath_png = REPO / "docs" / f"{page}.png"
        command = [sys.executable, "-m", "imgui_masknmf_manim.app", "--scale", "1", "--page", page,
                   "--time", str(SHOTS[page]), "--screenshot", str(filepath_png)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(filepath_png)


if __name__ == "__main__":
    main()
