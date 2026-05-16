const WIDTH = 1200;
const HEIGHT = 800;
const NUM_BIRDS = 60;
const NUM_PREDATORS = 3;
const VISUAL_RANGE = 85;
const PROTECTED_RANGE = 28;
const PREDATOR_RANGE = 170;
const MAX_SPEED = 4.2;
const MIN_NEURAL_SPEED = 1.2;
const MAX_FORCE = 0.09;
const MAX_TURN_DEGREES = 7.0;
const MAX_SPEED_CHANGE = 0.18;
const NEIGHBOR_COUNT_SCALE = 8.0;
const PREDATOR_CATCH_RADIUS = 22.0;
const PREDATOR_SEPARATION_RANGE = 72;
const PREDATOR_SEPARATION_WEIGHT = 1.9;
const GENERATIONS = [0, 25, 50, 75, 100];

const canvas = document.getElementById("simulationCanvas");
const ctx = canvas.getContext("2d");
const classicModeButton = document.getElementById("classicMode");
const geneticModeButton = document.getElementById("geneticMode");
const generationSelect = document.getElementById("generationSelect");
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");
const resetButton = document.getElementById("resetButton");
const pauseButton = document.getElementById("pauseButton");
const statusLine = document.getElementById("statusLine");
const modeStat = document.getElementById("modeStat");
const checkpointStat = document.getElementById("checkpointStat");
const brainStat = document.getElementById("brainStat");
const fitnessStat = document.getElementById("fitnessStat");
const captureStat = document.getElementById("captureStat");
const fpsStat = document.getElementById("fpsStat");

const sprites = {
  bird: loadSprite("assets/bird.svg"),
  falcon: loadSprite("assets/falcon.svg"),
};

const state = {
  mode: "classic",
  generation: 0,
  checkpoint: null,
  birds: [],
  predators: [],
  paused: false,
  mouse: { x: WIDTH / 2, y: HEIGHT / 2, inside: false },
  captures: 0,
  frames: 0,
  fps: 0,
  speedAccumulator: 0,
  lastFpsTime: performance.now(),
};

function loadSprite(src) {
  const image = new Image();
  image.src = src;
  return image;
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function vec(x = 0, y = 0) {
  return { x, y };
}

function add(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function mul(v, scalar) {
  return { x: v.x * scalar, y: v.y * scalar };
}

function div(v, scalar) {
  return { x: v.x / scalar, y: v.y / scalar };
}

function lenSq(v) {
  return v.x * v.x + v.y * v.y;
}

function length(v) {
  return Math.sqrt(lenSq(v));
}

function normalize(v) {
  const l = length(v);
  if (l === 0) return vec();
  return div(v, l);
}

function setLength(v, size) {
  const n = normalize(v);
  return mul(n, size);
}

function limit(v, max) {
  if (lenSq(v) > max * max) return setLength(v, max);
  return v;
}

function rotate(v, degrees) {
  const radians = degrees * Math.PI / 180;
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  return { x: v.x * c - v.y * s, y: v.x * s + v.y * c };
}

function toroidalDelta(source, target) {
  let dx = target.x - source.x;
  let dy = target.y - source.y;
  if (dx > WIDTH / 2) dx -= WIDTH;
  if (dx < -WIDTH / 2) dx += WIDTH;
  if (dy > HEIGHT / 2) dy -= HEIGHT;
  if (dy < -HEIGHT / 2) dy += HEIGHT;
  return { x: dx, y: dy };
}

function wrapPosition(position) {
  position.x = ((position.x % WIDTH) + WIDTH) % WIDTH;
  position.y = ((position.y % HEIGHT) + HEIGHT) % HEIGHT;
}

function randomVelocity() {
  let velocity = vec(rand(-1, 1), rand(-1, 1));
  if (lenSq(velocity) === 0) velocity = vec(1, 0);
  return setLength(velocity, rand(1.5, 3.4));
}

class NeuralNetwork {
  constructor(architecture, genome) {
    this.inputSize = architecture.input_size;
    this.hiddenSize = architecture.hidden_size;
    this.outputSize = architecture.output_size;
    this.genome = genome;
  }

  forward(inputs) {
    let cursor = 0;
    const hidden = [];
    for (let h = 0; h < this.hiddenSize; h += 1) {
      let value = 0;
      for (const input of inputs) {
        value += input * this.genome[cursor];
        cursor += 1;
      }
      value += this.genome[cursor];
      cursor += 1;
      hidden.push(Math.tanh(value));
    }

    const outputs = [];
    for (let o = 0; o < this.outputSize; o += 1) {
      let value = 0;
      for (const hiddenValue of hidden) {
        value += hiddenValue * this.genome[cursor];
        cursor += 1;
      }
      value += this.genome[cursor];
      cursor += 1;
      outputs.push(Math.tanh(value));
    }
    return outputs;
  }
}

class Bird {
  constructor(brain = null) {
    this.position = vec(rand(0, WIDTH), rand(0, HEIGHT));
    this.velocity = randomVelocity();
    this.acceleration = vec();
    this.brain = brain;
    this.fitness = 0;
    this.rank = null;
  }

  flock(birds, predators) {
    let separation = vec();
    let alignment = vec();
    let cohesion = vec();
    let totalNeighbors = 0;
    let closeNeighbors = 0;

    for (const other of birds) {
      if (other === this) continue;
      const delta = toroidalDelta(this.position, other.position);
      const d2 = lenSq(delta);
      if (d2 === 0) continue;

      if (d2 < VISUAL_RANGE * VISUAL_RANGE) {
        alignment = add(alignment, other.velocity);
        cohesion = add(cohesion, add(this.position, delta));
        totalNeighbors += 1;
      }

      if (d2 < PROTECTED_RANGE * PROTECTED_RANGE) {
        separation = sub(separation, div(delta, Math.max(d2, 1)));
        closeNeighbors += 1;
      }
    }

    if (totalNeighbors > 0) {
      alignment = div(alignment, totalNeighbors);
      if (lenSq(alignment) > 0) {
        alignment = limit(sub(setLength(alignment, MAX_SPEED), this.velocity), MAX_FORCE);
      }
      const center = div(cohesion, totalNeighbors);
      let cohesionDirection = sub(center, this.position);
      if (lenSq(cohesionDirection) > 0) {
        cohesion = limit(sub(setLength(cohesionDirection, MAX_SPEED), this.velocity), MAX_FORCE);
      }
    } else {
      alignment = vec();
      cohesion = vec();
    }

    if (closeNeighbors > 0 && lenSq(separation) > 0) {
      separation = div(separation, closeNeighbors);
      separation = limit(sub(setLength(separation, MAX_SPEED), this.velocity), MAX_FORCE);
    }

    const flee = this.fleeFromPredators(predators);
    this.acceleration = add(this.acceleration, mul(separation, 1.75));
    this.acceleration = add(this.acceleration, mul(alignment, 1.0));
    this.acceleration = add(this.acceleration, mul(cohesion, 0.85));
    this.acceleration = add(this.acceleration, mul(flee, 2.8));
  }

  steerWithBrain(birds, predators) {
    if (!this.brain) return;
    const outputs = this.brain.forward(this.neuralInputs(birds, predators));
    const turnSignal = outputs[0] ?? 0;
    const speedSignal = outputs[1] ?? 0;
    if (lenSq(this.velocity) === 0) this.velocity = randomVelocity();
    this.velocity = rotate(this.velocity, turnSignal * MAX_TURN_DEGREES);
    const speed = clamp(length(this.velocity) + speedSignal * MAX_SPEED_CHANGE, MIN_NEURAL_SPEED, MAX_SPEED);
    this.velocity = setLength(this.velocity, speed);
  }

  neuralInputs(birds, predators) {
    let predatorVector = vec();
    let predatorDistance = PREDATOR_RANGE;
    if (predators.length > 0) {
      let closest = predators[0];
      let best = lenSq(toroidalDelta(this.position, closest.position));
      for (const predator of predators.slice(1)) {
        const d2 = lenSq(toroidalDelta(this.position, predator.position));
        if (d2 < best) {
          best = d2;
          closest = predator;
        }
      }
      predatorVector = toroidalDelta(this.position, closest.position);
      predatorDistance = length(predatorVector);
    }

    const groupVector = this.vectorToGroup(birds);
    const neighborCount = this.countNeighbors(birds);
    return [
      ...normalizedComponents(predatorVector, PREDATOR_RANGE),
      ...normalizedComponents(groupVector, VISUAL_RANGE),
      ...normalizedComponents(this.velocity, MAX_SPEED),
      clamp(neighborCount / NEIGHBOR_COUNT_SCALE, 0, 1),
      clamp(predatorDistance / PREDATOR_RANGE, 0, 1),
    ];
  }

  vectorToGroup(birds) {
    let center = vec();
    let total = 0;
    for (const other of birds) {
      if (other === this) continue;
      const delta = toroidalDelta(this.position, other.position);
      const d2 = lenSq(delta);
      if (d2 > 0 && d2 < VISUAL_RANGE * VISUAL_RANGE) {
        center = add(center, add(this.position, delta));
        total += 1;
      }
    }
    if (total === 0) return vec();
    return sub(div(center, total), this.position);
  }

  countNeighbors(birds) {
    let total = 0;
    for (const other of birds) {
      if (other === this) continue;
      const delta = toroidalDelta(this.position, other.position);
      const d2 = lenSq(delta);
      if (d2 > 0 && d2 < VISUAL_RANGE * VISUAL_RANGE) total += 1;
    }
    return total;
  }

  fleeFromPredators(predators) {
    let flee = vec();
    let total = 0;
    for (const predator of predators) {
      const delta = toroidalDelta(this.position, predator.position);
      const d2 = lenSq(delta);
      if (d2 > 0 && d2 <= PREDATOR_RANGE * PREDATOR_RANGE) {
        flee = sub(flee, normalize(delta));
        total += 1;
      }
    }
    if (total === 0 || lenSq(flee) === 0) return vec();
    flee = setLength(normalize(flee), MAX_SPEED);
    return limit(sub(flee, this.velocity), MAX_FORCE * 1.8);
  }

  update(predators) {
    this.velocity = limit(add(this.velocity, this.acceleration), MAX_SPEED);
    this.position = add(this.position, this.velocity);
    this.acceleration = vec();
    wrapPosition(this.position);

    for (const predator of predators) {
      if (length(toroidalDelta(this.position, predator.position)) < PREDATOR_CATCH_RADIUS) {
        this.respawn();
        state.captures += 1;
        break;
      }
    }
  }

  respawn() {
    this.position = vec(rand(0, WIDTH), rand(0, HEIGHT));
    this.velocity = randomVelocity();
    this.acceleration = vec();
  }

  draw(context) {
    const angle = Math.atan2(this.velocity.y, this.velocity.x);
    context.save();
    context.translate(this.position.x, this.position.y);
    context.rotate(angle);

    if (sprites.bird.complete && sprites.bird.naturalWidth > 0) {
      context.drawImage(sprites.bird, -18, -9, 36, 18);
      if (this.brain && this.rank <= 3) {
        context.strokeStyle = rankColor(this.rank);
        context.lineWidth = 1.5;
        context.beginPath();
        context.arc(0, 0, 14, 0, Math.PI * 2);
        context.stroke();
      }
      context.restore();
      return;
    }

    context.beginPath();
    context.moveTo(13, 0);
    context.lineTo(-10, -7);
    context.lineTo(-5, 0);
    context.lineTo(-10, 7);
    context.closePath();
    context.fillStyle = this.brain ? rankColor(this.rank) : "#e9f1ff";
    context.strokeStyle = this.brain ? "#ffda79" : "#58a7ff";
    context.lineWidth = 1.5;
    context.fill();
    context.stroke();
    context.restore();
  }
}

class Predator {
  constructor(mouseControlled = false) {
    this.position = vec(rand(0, WIDTH), rand(0, HEIGHT));
    this.velocity = vec();
    this.acceleration = vec();
    this.mouseControlled = mouseControlled;
    this.maxSpeed = 5.2;
    this.maxForce = 0.14;
  }

  update(birds, predators) {
    if (this.mouseControlled && state.mouse.inside) {
      this.position = { ...state.mouse };
      this.velocity = vec();
      return;
    }

    const target = this.findNearestBird(birds);
    if (target) {
      let desired = toroidalDelta(this.position, target.position);
      if (lenSq(desired) > 0) {
        desired = setLength(desired, this.maxSpeed);
        this.acceleration = add(this.acceleration, limit(sub(desired, this.velocity), this.maxForce));
      }
    }

    this.acceleration = add(this.acceleration, this.separate(predators));
    this.velocity = limit(add(this.velocity, this.acceleration), this.maxSpeed);
    this.position = add(this.position, this.velocity);
    this.acceleration = vec();
    wrapPosition(this.position);
  }

  findNearestBird(birds) {
    let target = null;
    let best = Infinity;
    for (const bird of birds) {
      const d2 = lenSq(toroidalDelta(this.position, bird.position));
      if (d2 < best) {
        best = d2;
        target = bird;
      }
    }
    return target;
  }

  separate(predators) {
    let separation = vec();
    let total = 0;
    for (const other of predators) {
      if (other === this) continue;
      const delta = toroidalDelta(this.position, other.position);
      const d2 = lenSq(delta);
      if (d2 === 0) {
        separation = add(separation, rotate(vec(1, 0), rand(0, 360)));
        total += 1;
      } else if (d2 < PREDATOR_SEPARATION_RANGE * PREDATOR_SEPARATION_RANGE) {
        separation = sub(separation, div(delta, Math.max(d2, 1)));
        total += 1;
      }
    }
    if (total === 0 || lenSq(separation) === 0) return vec();
    separation = setLength(normalize(div(separation, total)), this.maxSpeed);
    return mul(limit(sub(separation, this.velocity), this.maxForce), PREDATOR_SEPARATION_WEIGHT);
  }

  draw(context) {
    const angle = Math.atan2(this.velocity.y, this.velocity.x);
    context.save();
    context.translate(this.position.x, this.position.y);
    context.rotate(angle);

    if (sprites.falcon.complete && sprites.falcon.naturalWidth > 0) {
      context.drawImage(sprites.falcon, -28, -17, 56, 34);
      context.restore();
      return;
    }

    context.beginPath();
    context.moveTo(22, 0);
    context.lineTo(-16, -12);
    context.lineTo(-8, 0);
    context.lineTo(-16, 12);
    context.closePath();
    context.fillStyle = "#ff6752";
    context.strokeStyle = "#52131c";
    context.lineWidth = 3;
    context.fill();
    context.stroke();
    context.restore();
  }
}

function normalizedComponents(vector, maxLength) {
  if (maxLength <= 0) return [0, 0];
  return [clamp(vector.x / maxLength, -1, 1), clamp(vector.y / maxLength, -1, 1)];
}

function rankColor(rank) {
  if (rank === 1) return "#ffda79";
  if (rank && rank <= 3) return "#b8f07a";
  if (rank && rank <= 6) return "#7dd3fc";
  return "#e9f1ff";
}

function createBirds() {
  const checkpointBrains = state.checkpoint?.brains ?? [];
  state.birds = [];
  for (let i = 0; i < NUM_BIRDS; i += 1) {
    let brain = null;
    let rank = null;
    if (state.mode === "genetic" && checkpointBrains.length > 0) {
      const entry = checkpointBrains[i % checkpointBrains.length];
      brain = new NeuralNetwork(state.checkpoint.architecture, entry.genome);
      rank = entry.rank;
    }
    const bird = new Bird(brain);
    bird.rank = rank;
    state.birds.push(bird);
  }
  state.captures = 0;
}

function createPredators() {
  state.predators = [];
  for (let i = 0; i < NUM_PREDATORS; i += 1) {
    state.predators.push(new Predator(false));
  }
}

function resetSimulation() {
  createBirds();
  createPredators();
}

async function loadCheckpoint(generation) {
  const embeddedCheckpoint = window.CHECKPOINT_DATA?.[String(generation)];
  if (embeddedCheckpoint) {
    state.checkpoint = embeddedCheckpoint;
    state.generation = generation;
    statusLine.textContent = `Checkpoint chargé: données embarquées gen ${generation}`;
    resetSimulation();
    updateStats();
    return;
  }

  const fileName = `top_gen_${String(generation).padStart(3, "0")}.json`;
  const candidates = [`checkpoints/${fileName}`, `../checkpoints/${fileName}`, `/checkpoints/${fileName}`];
  let lastError = null;
  for (const path of candidates) {
    try {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const checkpoint = await response.json();
      state.checkpoint = checkpoint;
      state.generation = generation;
      statusLine.textContent = `Checkpoint chargé: ${path}`;
      resetSimulation();
      updateStats();
      return;
    } catch (error) {
      lastError = error;
    }
  }
  state.checkpoint = null;
  statusLine.textContent = `Checkpoint ${generation} introuvable (${lastError?.message ?? "fetch failed"}).`;
  resetSimulation();
  updateStats();
}

function update() {
  for (const predator of state.predators) predator.update(state.birds, state.predators);

  if (state.mode === "classic") {
    for (const bird of state.birds) bird.flock(state.birds, state.predators);
  } else {
    for (const bird of state.birds) bird.steerWithBrain(state.birds, state.predators);
  }

  for (const bird of state.birds) bird.update(state.predators);
}

function drawGrid() {
  ctx.strokeStyle = "rgba(104, 130, 166, 0.08)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= WIDTH; x += 60) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, HEIGHT);
    ctx.stroke();
  }
  for (let y = 0; y <= HEIGHT; y += 60) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(WIDTH, y);
    ctx.stroke();
  }
}

function draw() {
  ctx.fillStyle = "#0c1118";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  drawGrid();

  ctx.strokeStyle = "rgba(255, 103, 82, 0.18)";
  ctx.lineWidth = 2;
  for (const predator of state.predators) {
    ctx.beginPath();
    ctx.arc(predator.position.x, predator.position.y, PREDATOR_RANGE, 0, Math.PI * 2);
    ctx.stroke();
  }

  for (const bird of state.birds) bird.draw(ctx);
  for (const predator of state.predators) predator.draw(ctx);
}

function updateStats() {
  modeStat.textContent = state.mode === "classic" ? "Classique" : "Génétique";
  checkpointStat.textContent = state.mode === "genetic" ? `Gen ${state.generation}` : "-";
  brainStat.textContent = state.checkpoint ? `${state.checkpoint.top_count} top` : "-";
  fitnessStat.textContent = state.checkpoint ? state.checkpoint.best_fitness.toFixed(2) : "-";
  captureStat.textContent = String(state.captures);
  fpsStat.textContent = String(state.fps);
}

function tick(now) {
  if (!state.paused) {
    state.speedAccumulator += Number(speedSlider.value);
    const steps = Math.floor(state.speedAccumulator);
    state.speedAccumulator -= steps;
    for (let i = 0; i < steps; i += 1) update();
  }
  draw();
  state.frames += 1;
  if (now - state.lastFpsTime >= 500) {
    state.fps = Math.round(state.frames * 1000 / (now - state.lastFpsTime));
    state.frames = 0;
    state.lastFpsTime = now;
    updateStats();
  }
  requestAnimationFrame(tick);
}

function setMode(mode) {
  state.mode = mode;
  classicModeButton.classList.toggle("active", mode === "classic");
  geneticModeButton.classList.toggle("active", mode === "genetic");
  generationSelect.disabled = mode !== "genetic";
  if (mode === "genetic") {
    loadCheckpoint(Number(generationSelect.value));
  } else {
    statusLine.textContent = "Mode classique Reynolds actif.";
    resetSimulation();
  }
  updateStats();
}

function canvasMousePosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * WIDTH / rect.width,
    y: (event.clientY - rect.top) * HEIGHT / rect.height,
  };
}

classicModeButton.addEventListener("click", () => setMode("classic"));
geneticModeButton.addEventListener("click", () => setMode("genetic"));
generationSelect.addEventListener("change", () => loadCheckpoint(Number(generationSelect.value)));
speedSlider.addEventListener("input", () => {
  speedValue.textContent = `${Number(speedSlider.value).toFixed(2).replace(/\.00$/, "")}x`;
});
resetButton.addEventListener("click", resetSimulation);
pauseButton.addEventListener("click", () => {
  state.paused = !state.paused;
  pauseButton.textContent = state.paused ? "▶" : "Ⅱ";
});
canvas.addEventListener("mousemove", (event) => {
  state.mouse = { ...canvasMousePosition(event), inside: true };
});
canvas.addEventListener("mouseenter", (event) => {
  state.mouse = { ...canvasMousePosition(event), inside: true };
});
canvas.addEventListener("mouseleave", () => {
  state.mouse.inside = false;
});

for (const generation of GENERATIONS) {
  const option = [...generationSelect.options].find((item) => Number(item.value) === generation);
  if (option) option.textContent = `Gen ${generation}`;
}

generationSelect.disabled = true;
speedValue.textContent = `${Number(speedSlider.value).toFixed(2).replace(/\.00$/, "")}x`;
resetSimulation();
statusLine.textContent = "Mode classique Reynolds actif. Passe en génétique pour charger un checkpoint.";
updateStats();
requestAnimationFrame(tick);
