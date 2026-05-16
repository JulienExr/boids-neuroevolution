import os
import random
import sys

import pygame

from src.bird import Bird
from src.checkpoints import CheckpointManager
from src.genetic import GeneticAlgorithm
from src.predator import Predator
from src.settings import (
    FAST_EVENT_POLL_INTERVAL,
    FAST_MODE_STEPS,
    FAST_STATUS_INTERVAL_MS,
    CHECKPOINT_DIR,
    CHECKPOINT_GENERATIONS,
    CHECKPOINT_TOP_BRAINS,
    CHECKPOINT_WEB_DIR,
    FPS,
    HEIGHT,
    NUM_BIRDS,
    NUM_PREDATORS,
    WIDTH,
    BoidSettings,
    ControlMode,
    NeuralSettings,
)
from src.sprites import create_bird_sprite, create_predator_sprite


class Simulation:
    def __init__(self):
        self.configure_video_driver()
        print(
            "[pygame] starting display | "
            f"SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER', 'auto')} | "
            f"DISPLAY={os.environ.get('DISPLAY', '')} | "
            f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}",
            flush=True,
        )
        pygame.init()
        pygame.display.set_caption("Boids - Simulation d'essaim")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        print(
            f"[pygame] display ready | driver={pygame.display.get_driver()} | "
            f"window={WIDTH}x{HEIGHT}",
            flush=True,
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.settings = BoidSettings()
        self.neural_settings = NeuralSettings()
        self.checkpoint_manager = CheckpointManager(
            CHECKPOINT_DIR,
            CHECKPOINT_GENERATIONS,
            CHECKPOINT_TOP_BRAINS,
            self.neural_settings,
            self.settings,
            CHECKPOINT_WEB_DIR,
        )
        self.genetic_algorithm = GeneticAlgorithm(
            self.neural_settings,
            self.checkpoint_manager,
        )
        self.mode = ControlMode.BOIDS
        self.generation_frame = 0
        self.captures_this_generation = 0
        self.fast_mode = False
        self.fast_status_last_ticks = pygame.time.get_ticks()
        self.fast_status_steps = 0
        self.fast_steps_per_second = 0.0
        self.paused = False
        self.show_hud = True

        Bird.rotation_cache.clear()
        Predator.rotation_cache.clear()
        self.bird_sprite = create_bird_sprite()
        self.predator_sprite = create_predator_sprite()

        self.birds = [
            Bird(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), self.bird_sprite)
            for _ in range(NUM_BIRDS)
        ]
        self.predators = [
            Predator(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT),
                self.predator_sprite,
                mouse_controlled=False,
            )
            for index in range(NUM_PREDATORS)
        ]
        self.genetic_algorithm.initialize_population(self.birds)

    @staticmethod
    def configure_video_driver() -> None:
        requested_driver = os.environ.get("BOIDS_SDL_DRIVER")
        if requested_driver:
            os.environ["SDL_VIDEODRIVER"] = requested_driver

    def run(self) -> None:
        while True:
            self.handle_events()

            if self.fast_mode and not self.paused:
                self.run_fast_batch()
                continue

            if not self.paused:
                self.update()
            self.draw()
            self.clock.tick(FPS)

    def run_fast_batch(self) -> None:
        steps_done = 0
        for step in range(FAST_MODE_STEPS):
            if step and step % FAST_EVENT_POLL_INTERVAL == 0:
                self.handle_events()
                if not self.fast_mode or self.paused:
                    break
            self.update()
            steps_done += 1

        self.fast_status_steps += steps_done
        self.draw_fast_status_if_due()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_m and self.predators:
                    self.predators[0].mouse_controlled = not self.predators[0].mouse_controlled
                    for predator in self.predators[1:]:
                        predator.mouse_controlled = False
                if event.key == pygame.K_b:
                    self.mode = ControlMode.BOIDS
                if event.key == pygame.K_n:
                    self.mode = ControlMode.NEURAL
                    self.genetic_algorithm.initialize_population(self.birds)
                    self.generation_frame = 0
                    self.captures_this_generation = 0
                if event.key == pygame.K_g and self.mode == ControlMode.NEURAL:
                    self.evolve_neural_population()
                if event.key == pygame.K_f:
                    self.fast_mode = not self.fast_mode
                    if self.fast_mode:
                        self.reset_fast_status()
                        self.draw_fast_status()
                if event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                if event.key == pygame.K_r:
                    self.__init__()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                brain = None
                if self.mode == ControlMode.NEURAL:
                    brain = self.genetic_algorithm.create_brain()
                self.birds.append(
                    Bird(event.pos[0], event.pos[1], self.bird_sprite, brain)
                )

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and self.birds:
                mouse = pygame.Vector2(event.pos)
                self.birds.sort(key=lambda bird: (bird.position - mouse).length_squared())
                self.birds.pop(0)

    def update(self) -> None:
        for predator in self.predators:
            predator.update(self.birds, self.predators)

        if self.mode == ControlMode.NEURAL:
            self.update_neural_mode()
            return

        for bird in self.birds:
            bird.flock(self.birds, self.predators, self.settings)
        for bird in self.birds:
            bird.update(self.settings)

    def update_neural_mode(self) -> None:
        for bird in self.birds:
            bird.steer_with_brain(
                self.birds,
                self.predators,
                self.settings,
                self.neural_settings,
            )

        for bird in self.birds:
            bird.update(self.settings)

        for bird in self.birds:
            was_captured = bird.update_neural_fitness(
                self.birds,
                self.predators,
                self.settings,
                self.neural_settings,
            )
            if was_captured:
                self.captures_this_generation += 1

        self.generation_frame += 1
        if self.generation_frame >= self.neural_settings.generation_frames:
            self.evolve_neural_population()

    def evolve_neural_population(self) -> None:
        self.birds = self.genetic_algorithm.evolve(self.birds, self.bird_sprite)
        self.generation_frame = 0
        self.captures_this_generation = 0

    def reset_fast_status(self) -> None:
        self.fast_status_last_ticks = pygame.time.get_ticks()
        self.fast_status_steps = 0
        self.fast_steps_per_second = 0.0

    def draw_fast_status_if_due(self) -> None:
        now = pygame.time.get_ticks()
        elapsed = now - self.fast_status_last_ticks
        if elapsed < FAST_STATUS_INTERVAL_MS:
            return

        self.fast_steps_per_second = self.fast_status_steps * 1000 / max(elapsed, 1)
        self.fast_status_steps = 0
        self.fast_status_last_ticks = now
        self.draw_fast_status()

    def draw_fast_status(self) -> None:
        self.screen.fill((8, 11, 16))

        current_best = max((bird.fitness for bird in self.birds), default=0.0)
        progress = self.generation_frame / max(self.neural_settings.generation_frames, 1)
        progress = max(0.0, min(1.0, progress))

        lines = [
            "MODE RAPIDE - rendu minimal",
            f"Generation: {self.genetic_algorithm.generation}",
            f"Progression: {self.generation_frame}/{self.neural_settings.generation_frames}",
            f"Best courant: {current_best:.2f}",
            f"Best precedent: {self.genetic_algorithm.best_fitness:.2f}",
            f"Checkpoints: top {CHECKPOINT_TOP_BRAINS} @ {CHECKPOINT_GENERATIONS}",
            f"Captures: {self.captures_this_generation}",
            f"Simulation: {self.fast_steps_per_second:.0f} updates/s",
            "F: retour au rendu normal | Espace: pause | Esc: quitter",
        ]

        y = 80
        for index, line in enumerate(lines):
            color = (255, 218, 121) if index == 0 else (224, 232, 244)
            text = self.font.render(line, True, color)
            self.screen.blit(text, (80, y))
            y += 34

        bar_rect = pygame.Rect(80, y + 12, 520, 18)
        fill_rect = pygame.Rect(80, y + 12, int(520 * progress), 18)
        pygame.draw.rect(self.screen, (39, 48, 64), bar_rect, border_radius=4)
        pygame.draw.rect(self.screen, (88, 167, 255), fill_rect, border_radius=4)
        pygame.display.flip()

    def draw(self) -> None:
        self.screen.fill((12, 17, 24))
        self.draw_predator_radius()

        for bird in self.birds:
            bird.draw(self.screen)

        for predator in self.predators:
            predator.draw(self.screen)

        if self.show_hud:
            self.draw_hud()

        pygame.display.flip()

    def draw_predator_radius(self) -> None:
        surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for predator in self.predators:
            pygame.draw.circle(
                surface,
                (255, 95, 75, 22),
                (int(predator.position.x), int(predator.position.y)),
                int(self.settings.predator_range),
                2,
            )
        self.screen.blit(surface, (0, 0))

    def draw_hud(self) -> None:
        mouse_predator = bool(self.predators and self.predators[0].mouse_controlled)
        mode = "souris + chasse auto" if mouse_predator else "chasse auto"
        control = "Reynolds" if self.mode == ControlMode.BOIDS else "Neural + GA"
        speed = f"rapide sans rendu x{FAST_MODE_STEPS}" if self.fast_mode else "normal"
        lines = [
            (
                f"Mode: {control} | Vitesse: {speed} | Oiseaux: {len(self.birds)} | "
                f"Predateurs: {len(self.predators)} ({mode}) | "
                f"FPS: {self.clock.get_fps():.0f}"
            ),
            (
                "B: Reynolds | N: neural/GA | F: rapide | G: generation | "
                "M: predateur | Espace: pause | Clic gauche/droit: +/- | "
                "H: HUD | R: reset | Esc: quitter"
            ),
        ]
        if self.mode == ControlMode.NEURAL:
            current_best = max((bird.fitness for bird in self.birds), default=0.0)
            lines.append(
                (
                    f"Generation: {self.genetic_algorithm.generation} | "
                    f"Frame: {self.generation_frame}/"
                    f"{self.neural_settings.generation_frames} | "
                    f"Best courant: {current_best:.2f} | "
                    f"Best precedent: {self.genetic_algorithm.best_fitness:.2f} | "
                    f"Captures: {self.captures_this_generation}"
                )
            )
        y = 12
        for line in lines:
            text = self.font.render(line, True, (224, 232, 244))
            shadow = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(shadow, (13, y + 1))
            self.screen.blit(text, (12, y))
            y += 24
