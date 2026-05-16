import os
import platform


def configure_sdl_driver() -> None:
    if os.environ.get("SDL_VIDEODRIVER"):
        return

    requested_driver = os.environ.get("BOIDS_SDL_DRIVER")
    if requested_driver:
        os.environ["SDL_VIDEODRIVER"] = requested_driver
        return

    is_wsl = "microsoft" in platform.uname().release.lower()
    if is_wsl and os.environ.get("DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "x11"


configure_sdl_driver()

from src.simulation import Simulation


if __name__ == "__main__":
    Simulation().run()
