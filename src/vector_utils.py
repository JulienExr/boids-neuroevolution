import pygame

from src.settings import HEIGHT, WIDTH


def limit_vector(vector: pygame.Vector2, max_value: float) -> pygame.Vector2:
    """Return a copy of vector capped to max_value length."""
    if vector.length_squared() > max_value * max_value:
        return vector.normalize() * max_value
    return vector


def toroidal_delta(source: pygame.Vector2, target: pygame.Vector2) -> pygame.Vector2:
    """Shortest direction from source to target on a wraparound screen."""
    dx = target.x - source.x
    dy = target.y - source.y

    if dx > WIDTH / 2:
        dx -= WIDTH
    elif dx < -WIDTH / 2:
        dx += WIDTH

    if dy > HEIGHT / 2:
        dy -= HEIGHT
    elif dy < -HEIGHT / 2:
        dy += HEIGHT

    return pygame.Vector2(dx, dy)
