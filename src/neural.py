from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class NeuralNetwork:
    input_size: int
    hidden_size: int
    output_size: int
    genome: list[float]

    @classmethod
    def random(cls, input_size: int, hidden_size: int, output_size: int) -> NeuralNetwork:
        genome_size = (
            input_size * hidden_size
            + hidden_size
            + hidden_size * output_size
            + output_size
        )
        genome = [random.uniform(-1.0, 1.0) for _ in range(genome_size)]
        return cls(input_size, hidden_size, output_size, genome)

    def clone(self) -> NeuralNetwork:
        return NeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.genome.copy(),
        )

    def forward(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}.")

        cursor = 0
        hidden_outputs = []
        for _ in range(self.hidden_size):
            value = 0.0
            for input_value in inputs:
                value += input_value * self.genome[cursor]
                cursor += 1
            value += self.genome[cursor]
            cursor += 1
            hidden_outputs.append(math.tanh(value))

        outputs = []
        for _ in range(self.output_size):
            value = 0.0
            for hidden_value in hidden_outputs:
                value += hidden_value * self.genome[cursor]
                cursor += 1
            value += self.genome[cursor]
            cursor += 1
            outputs.append(math.tanh(value))

        return outputs

    @staticmethod
    def crossover(parent_a: NeuralNetwork, parent_b: NeuralNetwork) -> NeuralNetwork:
        child_genome = [
            gene_a if random.random() < 0.5 else gene_b
            for gene_a, gene_b in zip(parent_a.genome, parent_b.genome)
        ]
        return NeuralNetwork(
            parent_a.input_size,
            parent_a.hidden_size,
            parent_a.output_size,
            child_genome,
        )

    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        for index, gene in enumerate(self.genome):
            if random.random() < mutation_rate:
                self.genome[index] = gene + random.gauss(0.0, mutation_strength)
