from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, ClassVar

import pygame

from src.neural import NeuralNetwork
from src.settings import HEIGHT, WIDTH, BoidSettings, NeuralSettings
from src.vector_utils import limit_vector, toroidal_delta

if TYPE_CHECKING:
    from src.predator import Predator


class Bird:
    rotation_cache: ClassVar[dict[int, pygame.Surface]] = {}

    def __init__(
        self,
        x: float,
        y: float,
        sprite: pygame.Surface,
        brain: NeuralNetwork | None = None,
    ):
        self.position = pygame.Vector2(x, y)
        self.velocity = self.random_velocity()
        self.acceleration = pygame.Vector2()
        self.angle = 0.0
        self.sprite = sprite
        self.brain = brain
        self.fitness = 0.0
        self.captures = 0

    @staticmethod
    def random_velocity() -> pygame.Vector2:
        velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if velocity.length_squared() == 0:
            velocity = pygame.Vector2(1, 0)
        velocity.scale_to_length(random.uniform(1.5, 3.4))
        return velocity

    def apply_force(self, force: pygame.Vector2) -> None:
        self.acceleration += force

    def flock(self, birds: list[Bird], predator: Predator, settings: BoidSettings) -> None:
        separation = pygame.Vector2()
        alignment = pygame.Vector2()
        cohesion = pygame.Vector2()
        total_neighbors = 0
        close_neighbors = 0

        for other in birds:
            if other is self:
                continue

            delta = toroidal_delta(self.position, other.position)
            distance_sq = delta.length_squared()
            if distance_sq == 0:
                continue

            if distance_sq < settings.visual_range * settings.visual_range:
                alignment += other.velocity
                cohesion += self.position + delta
                total_neighbors += 1

            if distance_sq < settings.protected_range * settings.protected_range:
                separation -= delta / max(distance_sq, 1.0)
                close_neighbors += 1

        if total_neighbors:
            alignment /= total_neighbors
            if alignment.length_squared() > 0:
                alignment = alignment.normalize() * settings.max_speed
                alignment = limit_vector(alignment - self.velocity, settings.max_force)

            center = cohesion / total_neighbors
            cohesion_direction = center - self.position
            if cohesion_direction.length_squared() > 0:
                cohesion = cohesion_direction.normalize() * settings.max_speed
                cohesion = limit_vector(cohesion - self.velocity, settings.max_force)
        else:
            alignment = pygame.Vector2()
            cohesion = pygame.Vector2()

        if close_neighbors and separation.length_squared() > 0:
            separation /= close_neighbors
            separation = separation.normalize() * settings.max_speed
            separation = limit_vector(separation - self.velocity, settings.max_force)

        flee = self.flee_from_predator(predator, settings)

        self.apply_force(separation * settings.separation_weight)
        self.apply_force(alignment * settings.alignment_weight)
        self.apply_force(cohesion * settings.cohesion_weight)
        self.apply_force(flee * settings.flee_weight)

    def steer_with_brain(
        self,
        birds: list[Bird],
        predator: Predator,
        settings: BoidSettings,
        neural_settings: NeuralSettings,
    ) -> None:
        if self.brain is None:
            return

        inputs = self.neural_inputs(birds, predator, settings)
        turn_signal = self.brain.forward(inputs)[0]
        self.velocity.rotate_ip(turn_signal * neural_settings.max_turn_degrees)

        if self.velocity.length_squared() > 0:
            self.velocity.scale_to_length(settings.max_speed)

    def neural_inputs(
        self, birds: list[Bird], predator: Predator, settings: BoidSettings
    ) -> list[float]:
        predator_vector = toroidal_delta(self.position, predator.position)
        group_vector = self.vector_to_group(birds, settings.visual_range)

        predator_input = self.normalized_components(
            predator_vector, settings.predator_range
        )
        group_input = self.normalized_components(group_vector, settings.visual_range)
        return [*predator_input, *group_input]

    def vector_to_group(self, birds: list[Bird], visual_range: float) -> pygame.Vector2:
        center = pygame.Vector2()
        total_neighbors = 0

        for other in birds:
            if other is self:
                continue

            delta = toroidal_delta(self.position, other.position)
            if 0 < delta.length_squared() < visual_range * visual_range:
                center += self.position + delta
                total_neighbors += 1

        if total_neighbors == 0:
            return pygame.Vector2()

        return center / total_neighbors - self.position

    def count_neighbors(self, birds: list[Bird], visual_range: float) -> int:
        total_neighbors = 0
        for other in birds:
            if other is self:
                continue

            delta = toroidal_delta(self.position, other.position)
            if 0 < delta.length_squared() < visual_range * visual_range:
                total_neighbors += 1

        return total_neighbors

    @staticmethod
    def normalized_components(
        vector: pygame.Vector2, max_length: float
    ) -> tuple[float, float]:
        if max_length <= 0:
            return 0.0, 0.0

        scaled = vector / max_length
        return max(-1.0, min(1.0, scaled.x)), max(-1.0, min(1.0, scaled.y))

    def flee_from_predator(
        self, predator: Predator, settings: BoidSettings
    ) -> pygame.Vector2:
        delta_to_predator = toroidal_delta(self.position, predator.position)
        distance_sq = delta_to_predator.length_squared()
        if distance_sq == 0 or distance_sq > settings.predator_range * settings.predator_range:
            return pygame.Vector2()

        desired = -delta_to_predator.normalize() * settings.max_speed
        steering = desired - self.velocity
        return limit_vector(steering, settings.max_force * 1.8)

    def update(self, settings: BoidSettings) -> None:
        self.velocity += self.acceleration
        self.velocity = limit_vector(self.velocity, settings.max_speed)

        if self.velocity.length_squared() > 0:
            self.angle = math.degrees(math.atan2(-self.velocity.y, self.velocity.x))

        self.position += self.velocity
        self.acceleration.update(0, 0)
        self.wrap_edges()

    def update_neural_fitness(
        self,
        birds: list[Bird],
        predator: Predator,
        settings: BoidSettings,
        neural_settings: NeuralSettings,
    ) -> bool:
        predator_distance = toroidal_delta(self.position, predator.position).length()
        group_distance = self.vector_to_group(birds, settings.visual_range).length()
        neighbor_count = self.count_neighbors(birds, settings.visual_range)

        safety_score = min(predator_distance / settings.predator_range, 1.0)
        group_distance_score = max(0.0, 1.0 - group_distance / settings.visual_range)
        group_density_score = min(neighbor_count / 6.0, 1.0)
        group_score = group_distance_score * group_density_score
        speed_score = min(self.velocity.length() / settings.max_speed, 1.0)

        self.fitness += 0.012 * safety_score
        self.fitness += 0.006 * group_score
        self.fitness += 0.002 * speed_score

        if predator_distance < neural_settings.predator_catch_radius:
            self.fitness -= 1.0
            self.captures += 1
            self.respawn()
            return True

        return False

    def reset_fitness(self) -> None:
        self.fitness = 0.0
        self.captures = 0

    def respawn(self) -> None:
        self.position.update(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        self.velocity = self.random_velocity()
        self.acceleration.update(0, 0)

    def wrap_edges(self) -> None:
        self.position.x %= WIDTH
        self.position.y %= HEIGHT

    def draw(self, screen: pygame.Surface) -> None:
        # Base sprite points right; quantized rotation keeps transform cost low.
        angle_key = round(self.angle) % 360
        rotated = self.rotation_cache.get(angle_key)
        if rotated is None:
            rotated = pygame.transform.rotate(self.sprite, angle_key)
            self.rotation_cache[angle_key] = rotated
        rect = rotated.get_rect(center=self.position)
        screen.blit(rotated, rect)
