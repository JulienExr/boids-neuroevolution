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


## Checkpoints
During neuroevolution, the top 10 bird brains are saved as JSON for generations `0`, `25`, `50`, `75`, and `100`. Files are written to `checkpoints/top_gen_XXX.json` and include the neural architecture, genomes, fitness values, captures, and settings used for the run.


## Web Viewer
A static Canvas viewer is available in `docs/`. It supports the classic Reynolds mode and a genetic replay mode that loads `top_gen_000`, `025`, `050`, `075`, and `100` checkpoints.

Run it locally with:

```bash
cd web
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765/`. The `docs/checkpoints/` folder is mirrored when new checkpoints are saved so it can be published directly with GitHub Pages. `docs/checkpoint_data.js` embeds the same checkpoints so the viewer also works when opened directly from disk. The speed slider starts slower by default (`0.5x`) but still goes up to `8x`.
