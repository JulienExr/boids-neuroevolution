from dataclasses import dataclass
from enum import Enum


WIDTH = 1200
HEIGHT = 800
FPS = 60
NUM_BIRDS = 60
NUM_PREDATORS = 3
PREDATOR_SEPARATION_RANGE = 72.0
PREDATOR_SEPARATION_WEIGHT = 1.9
FAST_MODE_STEPS = 100
FAST_EVENT_POLL_INTERVAL = 5
FAST_STATUS_INTERVAL_MS = 250
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_WEB_DIR = "web/checkpoints"
CHECKPOINT_GENERATIONS = (0, 25, 50, 75, 100)
CHECKPOINT_TOP_BRAINS = 10


class ControlMode(Enum):
    BOIDS = "boids"
    NEURAL = "neural"


@dataclass
class BoidSettings:
    visual_range: float = 85.0
    protected_range: float = 28.0
    predator_range: float = 170.0
    max_speed: float = 4.2
    max_force: float = 0.09
    separation_weight: float = 1.75
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.85
    flee_weight: float = 2.8


@dataclass
class NeuralSettings:
    input_size: int = 8
    hidden_size: int = 10
    output_size: int = 2
    max_turn_degrees: float = 7.0
    max_speed_change: float = 0.18
    min_speed: float = 1.2
    neighbor_count_scale: float = 8.0
    generation_frames: int = 900
    mutation_rate: float = 0.12
    mutation_strength: float = 0.35
    elite_fraction: float = 0.12
    tournament_size: int = 5
    predator_catch_radius: float = 22.0
