from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import pygame

from src.settings import (
    HEIGHT,
    PREDATOR_SEPARATION_RANGE,
    PREDATOR_SEPARATION_WEIGHT,
    WIDTH,
)
from src.vector_utils import limit_vector, toroidal_delta

if TYPE_CHECKING:
    from src.bird import Bird


class Predator:
    rotation_cache: ClassVar[dict[int, pygame.Surface]] = {}

    def __init__(
        self,
        x: float,
        y: float,
        sprite: pygame.Surface,
        mouse_controlled: bool = True,
    ):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()
        self.angle = 0.0
        self.max_speed = 5.2
        self.max_force = 0.14
        self.sprite = sprite
        self.mouse_controlled = mouse_controlled

    def update(self, birds: list[Bird], predators: list[Predator]) -> None:
        if self.mouse_controlled:
            target = pygame.Vector2(pygame.mouse.get_pos())
            self.position.update(target)
            self.velocity.update(0, 0)
            return

        target = self.find_nearest_bird(birds)
        if target is not None:
            desired = toroidal_delta(self.position, target.position)
            if desired.length_squared() > 0:
                desired = desired.normalize() * self.max_speed
                steering = limit_vector(desired - self.velocity, self.max_force)
                self.acceleration += steering

        self.acceleration += self.separate_from_predators(predators)
        self.velocity += self.acceleration
        self.velocity = limit_vector(self.velocity, self.max_speed)
        if self.velocity.length_squared() > 0:
            self.angle = math.degrees(math.atan2(-self.velocity.y, self.velocity.x))
        self.position += self.velocity
        self.acceleration.update(0, 0)
        self.wrap_edges()

    def find_nearest_bird(self, birds: list[Bird]) -> Bird | None:
        if not birds:
            return None
        return min(
            birds,
            key=lambda bird: toroidal_delta(self.position, bird.position).length_squared(),
        )

    def separate_from_predators(self, predators: list[Predator]) -> pygame.Vector2:
        separation = pygame.Vector2()
        close_predators = 0

        for other in predators:
            if other is self:
                continue

            delta = toroidal_delta(self.position, other.position)
            distance_sq = delta.length_squared()
            if distance_sq == 0:
                angle = id(other) % 360
                separation += pygame.Vector2(1, 0).rotate(angle)
                close_predators += 1
                continue

            in_range = distance_sq < PREDATOR_SEPARATION_RANGE * PREDATOR_SEPARATION_RANGE
            if in_range:
                separation -= delta / max(distance_sq, 1.0)
                close_predators += 1

        if close_predators == 0 or separation.length_squared() == 0:
            return pygame.Vector2()

        separation /= close_predators
        separation = separation.normalize() * self.max_speed
        steering = limit_vector(separation - self.velocity, self.max_force)
        return steering * PREDATOR_SEPARATION_WEIGHT

    def wrap_edges(self) -> None:
        self.position.x %= WIDTH
        self.position.y %= HEIGHT

    def draw(self, screen: pygame.Surface) -> None:
        if self.mouse_controlled:
            mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
            if mouse_pos != self.position:
                delta = mouse_pos - self.position
                if delta.length_squared() > 0:
                    self.angle = math.degrees(math.atan2(-delta.y, delta.x))

        angle_key = round(self.angle) % 360
        rotated = self.rotation_cache.get(angle_key)
        if rotated is None:
            rotated = pygame.transform.rotate(self.sprite, angle_key)
            self.rotation_cache[angle_key] = rotated
        rect = rotated.get_rect(center=self.position)
        screen.blit(rotated, rect)
