import random
import sys

import pygame

from src.bird import Bird
from src.genetic import GeneticAlgorithm
from src.predator import Predator
from src.settings import (
    FPS,
    HEIGHT,
    NUM_BIRDS,
    WIDTH,
    BoidSettings,
    ControlMode,
    NeuralSettings,
)
from src.sprites import create_bird_sprite, create_predator_sprite


class Simulation:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Boids - Simulation d'essaim")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.settings = BoidSettings()
        self.neural_settings = NeuralSettings()
        self.genetic_algorithm = GeneticAlgorithm(self.neural_settings)
        self.mode = ControlMode.BOIDS
        self.generation_frame = 0
        self.captures_this_generation = 0
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
        self.predator = Predator(WIDTH / 2, HEIGHT / 2, self.predator_sprite)
        self.genetic_algorithm.initialize_population(self.birds)

    def run(self) -> None:
        while True:
            self.handle_events()
            if not self.paused:
                self.update()
            self.draw()
            self.clock.tick(FPS)

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
                if event.key == pygame.K_m:
                    self.predator.mouse_controlled = not self.predator.mouse_controlled
                if event.key == pygame.K_b:
                    self.mode = ControlMode.BOIDS
                if event.key == pygame.K_n:
                    self.mode = ControlMode.NEURAL
                    self.genetic_algorithm.initialize_population(self.birds)
                    self.generation_frame = 0
                    self.captures_this_generation = 0
                if event.key == pygame.K_g and self.mode == ControlMode.NEURAL:
                    self.evolve_neural_population()
                if event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                if event.key == pygame.K_r:
                    self.__init__()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                brain = None
                if self.mode == ControlMode.NEURAL:
                    brain = self.genetic_algorithm.create_brain()
                self.birds.append(Bird(event.pos[0], event.pos[1], self.bird_sprite, brain))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and self.birds:
                mouse = pygame.Vector2(event.pos)
                self.birds.sort(key=lambda bird: (bird.position - mouse).length_squared())
                self.birds.pop(0)

    def update(self) -> None:
        self.predator.update(self.birds)

        if self.mode == ControlMode.NEURAL:
            self.update_neural_mode()
            return

        for bird in self.birds:
            bird.flock(self.birds, self.predator, self.settings)
        for bird in self.birds:
            bird.update(self.settings)

    def update_neural_mode(self) -> None:
        for bird in self.birds:
            bird.steer_with_brain(
                self.birds,
                self.predator,
                self.settings,
                self.neural_settings,
            )

        for bird in self.birds:
            bird.update(self.settings)

        for bird in self.birds:
            was_captured = bird.update_neural_fitness(
                self.birds,
                self.predator,
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

    def draw(self) -> None:
        self.screen.fill((12, 17, 24))
        self.draw_predator_radius()

        for bird in self.birds:
            bird.draw(self.screen)

        self.predator.draw(self.screen)

        if self.show_hud:
            self.draw_hud()

        pygame.display.flip()

    def draw_predator_radius(self) -> None:
        surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(
            surface,
            (255, 95, 75, 22),
            (int(self.predator.position.x), int(self.predator.position.y)),
            int(self.settings.predator_range),
            2,
        )
        self.screen.blit(surface, (0, 0))

    def draw_hud(self) -> None:
        mode = "souris" if self.predator.mouse_controlled else "chasse auto"
        control = "Reynolds" if self.mode == ControlMode.BOIDS else "Neural + GA"
        lines = [
            f"Mode: {control} | Oiseaux: {len(self.birds)} | Predateur: {mode} | FPS: {self.clock.get_fps():.0f}",
            "B: Reynolds | N: neural/GA | G: nouvelle generation | M: predateur | Espace: pause | Clic gauche/droit: +/- | H: HUD | R: reset | Esc: quitter",
        ]
        if self.mode == ControlMode.NEURAL:
            lines.append(
                f"Generation: {self.genetic_algorithm.generation} | Frame: {self.generation_frame}/{self.neural_settings.generation_frames} | Best fitness: {self.genetic_algorithm.best_fitness:.2f} | Captures: {self.captures_this_generation}"
            )
        y = 12
        for line in lines:
            text = self.font.render(line, True, (224, 232, 244))
            shadow = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(shadow, (13, y + 1))
            self.screen.blit(text, (12, y))
            y += 24
