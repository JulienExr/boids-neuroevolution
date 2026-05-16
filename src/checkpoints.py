from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from src.neural import NeuralNetwork
from src.settings import BoidSettings, NeuralSettings

if TYPE_CHECKING:
    from src.bird import Bird


class CheckpointManager:
    def __init__(
        self,
        directory: str,
        target_generations: Iterable[int],
        top_brains: int,
        neural_settings: NeuralSettings,
        boid_settings: BoidSettings,
        mirror_directory: str | None = None,
    ):
        self.directory = Path(directory)
        self.mirror_directory = Path(mirror_directory) if mirror_directory else None
        self.target_generations = set(target_generations)
        self.top_brains = top_brains
        self.neural_settings = neural_settings
        self.boid_settings = boid_settings
        self.saved_generations: set[int] = set()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.mirror_directory is not None:
            self.mirror_directory.mkdir(parents=True, exist_ok=True)

    def maybe_save_top(self, generation: int, birds: list[Bird]) -> bool:
        if generation not in self.target_generations:
            return False
        if generation in self.saved_generations:
            return False

        ranked = [bird for bird in birds if bird.brain is not None]
        if not ranked:
            return False

        ranked.sort(key=lambda bird: bird.fitness, reverse=True)
        self.save_brains(generation, ranked[: self.top_brains])
        self.saved_generations.add(generation)
        return True

    def save_brains(self, generation: int, birds: list[Bird]) -> None:
        top_entries = []
        for rank, bird in enumerate(birds, start=1):
            if bird.brain is None:
                continue

            top_entries.append(
                {
                    "rank": rank,
                    "fitness": bird.fitness,
                    "captures": bird.captures,
                    "genome": bird.brain.genome,
                }
            )

        first_brain = birds[0].brain
        if first_brain is None:
            return

        payload = {
            "generation": generation,
            "top_count": len(top_entries),
            "best_fitness": top_entries[0]["fitness"] if top_entries else 0.0,
            "architecture": {
                "input_size": first_brain.input_size,
                "hidden_size": first_brain.hidden_size,
                "output_size": first_brain.output_size,
            },
            "brains": top_entries,
            "neural_settings": asdict(self.neural_settings),
            "boid_settings": asdict(self.boid_settings),
        }

        content = json.dumps(payload, indent=2)
        file_name = f"top_gen_{generation:03d}.json"
        path = self.directory / file_name
        path.write_text(content, encoding="utf-8")

        if self.mirror_directory is not None:
            mirror_path = self.mirror_directory / file_name
            mirror_path.write_text(content, encoding="utf-8")
            self.write_embedded_web_checkpoints()

    def write_embedded_web_checkpoints(self) -> None:
        if self.mirror_directory is None:
            return

        payloads = {}
        for path in sorted(self.mirror_directory.glob("top_gen_*.json")):
            generation = str(int(path.stem.split("_")[-1]))
            payloads[generation] = json.loads(path.read_text(encoding="utf-8"))

        if not payloads:
            return

        content = "window.CHECKPOINT_DATA = "
        content += json.dumps(payloads, separators=(",", ":"))
        content += ";\n"
        embedded_path = self.mirror_directory.parent / "checkpoint_data.js"
        embedded_path.write_text(content, encoding="utf-8")

    def load_brains(self, generation: int) -> list[NeuralNetwork]:
        path = self.directory / f"top_gen_{generation:03d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        architecture = payload["architecture"]
        return [
            NeuralNetwork(
                architecture["input_size"],
                architecture["hidden_size"],
                architecture["output_size"],
                entry["genome"],
            )
            for entry in payload["brains"]
        ]

    def load_brain(self, generation: int, rank: int = 1) -> NeuralNetwork:
        brains = self.load_brains(generation)
        if not 1 <= rank <= len(brains):
            raise ValueError(f"Rank must be between 1 and {len(brains)}.")
        return brains[rank - 1]
