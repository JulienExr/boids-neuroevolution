from __future__ import annotations

import random

from src.checkpoints import CheckpointManager
from src.bird import Bird
from src.neural import NeuralNetwork
from src.settings import HEIGHT, WIDTH, NeuralSettings


class GeneticAlgorithm:
    def __init__(
        self,
        settings: NeuralSettings,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.settings = settings
        self.checkpoint_manager = checkpoint_manager
        self.generation = 0
        self.best_fitness = 0.0

    def create_brain(self) -> NeuralNetwork:
        return NeuralNetwork.random(
            self.settings.input_size,
            self.settings.hidden_size,
            self.settings.output_size,
        )

    def initialize_population(self, birds: list[Bird]) -> None:
        for bird in birds:
            if bird.brain is None:
                bird.brain = self.create_brain()
            bird.reset_fitness()

    def evolve(self, birds: list[Bird], bird_sprite) -> list[Bird]:
        if not birds:
            return birds

        ranked = sorted(birds, key=lambda bird: bird.fitness, reverse=True)
        self.best_fitness = ranked[0].fitness
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.maybe_save_top(self.generation, ranked)

        elite_count = max(2, int(len(ranked) * self.settings.elite_fraction))
        next_generation = [
            self._spawn_bird(bird_sprite, ranked[index].brain.clone())
            for index in range(elite_count)
            if ranked[index].brain is not None
        ]

        while len(next_generation) < len(birds):
            parent_a = self._select_parent(ranked)
            parent_b = self._select_parent(ranked)
            if parent_a.brain is None or parent_b.brain is None:
                child_brain = self.create_brain()
            else:
                child_brain = NeuralNetwork.crossover(parent_a.brain, parent_b.brain)
                child_brain.mutate(
                    self.settings.mutation_rate,
                    self.settings.mutation_strength,
                )
            next_generation.append(self._spawn_bird(bird_sprite, child_brain))

        self.generation += 1
        return next_generation

    def _select_parent(self, ranked_birds: list[Bird]) -> Bird:
        sample_size = min(self.settings.tournament_size, len(ranked_birds))
        competitors = random.sample(ranked_birds, sample_size)
        return max(competitors, key=lambda bird: bird.fitness)

    @staticmethod
    def _spawn_bird(bird_sprite, brain: NeuralNetwork) -> Bird:
        bird = Bird(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), bird_sprite, brain)
        bird.reset_fitness()
        return bird
