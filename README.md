# Boids-neuroevolution

An interactive 2D flocking simulation built in Python using Pygame. This project combines emergent behavior with artificial intelligence to simulate complex evolutionary dynamics.

## Key Features
* **Boids Simulation:** Implementation of Craig Reynolds' classic rules: Separation, Alignment, and Cohesion.
* **Neuroevolution:** Every bird features a simple neural network controlling its steering and speed based on its surroundings.
  Neural inputs: predator vector, group vector, velocity vector, neighbor density, and predator distance. Outputs: turn and speed control.
* **Natural Selection:** Birds use a genetic algorithm to evolve survival behaviors generation after generation to escape a dynamic predator.
* **Optimized Sprites:** Smooth rendering of bird assets with pre-calculated rotations for optimal performance.

## Controls
* `B`: switch to classic Reynolds mode.
* `N`: switch to neural + genetic algorithm mode.
* `F`: toggle fast mode, replacing animation with a lightweight evolution dashboard.
* `G`: force a new generation in neural mode.
* `M`: toggle predator mouse control / automatic chase.
* `Space`: pause or resume.
* Left click / right click: add or remove a bird.
* `H`: show or hide the HUD.
* `R`: reset the simulation.
* `Esc`: quit.
