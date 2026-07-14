# hal-cpp

A C++/Vulkan port of the **Hybrid Automata Library (HAL)** — a framework for
hybrid modeling: spatial simulations that combine agent-based models with
partial-differential-equation (PDE) fields, originally developed for
mathematical oncology (tumor growth, drug/resource diffusion, population models).

This port keeps HAL's library-not-an-engine design (your model owns `main()` and
calls HAL each tick) while adding what the Java original lacks: a **GPU compute
backend** for the per-tick agent and PDE loops, and a data-oriented layout chosen
upfront for that purpose. See [`../docs/DESIGN.md`](../docs/DESIGN.md) for the
rationale behind each decision.

## Relationship to the original HAL

- Original (Java): <https://github.com/MathOnco/HAL> · <https://halloworld.org/>
- Paper: Bravo et al., *PLOS Comp. Biol.* (2020),
  [10.1371/journal.pcbi.1007635](https://doi.org/10.1371/journal.pcbi.1007635)
- **License: MIT** (Copyright © 2017 Rafael Ramon Bravo). This port is
  distributed under the same license — see [`LICENSE`](LICENSE).

If you use this in academic work, please cite the original HAL paper above.

## What's here

| Area | Headers | Description |
|------|---------|-------------|
| Grids | `include/Hal/Grids` | Value grids (`Grid1D/2D/3D`), PDE grids, agent grids, population grids |
| Agents | `include/Hal/Agents` | On-lattice (`AgentSQ2D/3D`) and off-lattice (`AgentPT`, `SphericalAgent`) agents + CPU force backend |
| PDE | `include/Hal/Pde` | Explicit stencil diffusion/advection, implicit ADI (Thomas solver) |
| GPU | `include/Hal/Gpu` | Vulkan compute kernels (diffusion, ADI, agent forces, reaction-diffusion, wave, 3D flow), plus 2D/3D render contexts |
| GUI | `include/Hal/Gui` | Dear ImGui / ImPlot layer |
| Tools | `include/Hal/Tools` | `ModuleSet`, `VarSetManager`, ODE solver, FFT, phylogeny tracker, Lenia |
| IO / Video | `include/Hal/IO`, `include/Hal/Video` | Versioned binary save/load, PNG export, MPEG-1/MP4 video |
| Concurrency | `include/Hal/Concurrency` | `MultiThread`, triple-buffered sim/render handoff |

Backends follow a strategy pattern (`IDiffusionBackend2D`, `IAgentForceBackend2D`,
`IAdiDiffusionBackend2D`) with matched CPU and Vulkan implementations, so model
code is identical whichever runs.

## Requirements

- CMake ≥ 3.20 and a C++17 compiler
- A Vulkan SDK / loader (the GPU compute backend is always built)
- Everything else (volk, VMA, GLFW, Dear ImGui, ImPlot) is fetched automatically
  via CMake `FetchContent`

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

Build options (all default `ON`):

| Option | Effect |
|--------|--------|
| `HAL_BUILD_TESTS` | Unit test suite (`tests/`) |
| `HAL_BUILD_WINDOW` | GLFW + Vulkan windowed rendering |
| `HAL_BUILD_GUI` | Dear ImGui + ImPlot GUI layer |
| `HAL_BUILD_EXAMPLES` | Ported example models |

## Run the examples

```sh
./build/examples/GameOfLife            # live Vulkan window
./build/examples/GameOfLife headless   # no window; writes a PNG
./build/examples/Gradient2D            # prints to stdout
./build/HalBackendBenchmark            # CPU vs GPU backend timing
```

Windowed models take an optional `headless` argument to render to a PNG/MP4
instead of a window. Nine GPU showcases (`GpuReactionDiffusion`, `GpuInkFlow`,
`GpuSwarm`, `GpuSlimeMold`, `GpuWaveField`, `GpuAttractor3D`, …) demonstrate the
GPU-resident compute loop. See [`examples/README.md`](examples/README.md).

## Test

```sh
ctest --test-dir build        # or: ./build/tests/<Name>
```

The suite (`tests/`) covers grids, agents, PDE/ADI numerics, every GPU kernel
against its CPU reference, serialization, and rendering contexts.

## Documentation

- [`../docs/DESIGN.md`](../docs/DESIGN.md) — design rationale, Java→C++ mapping,
  GPU-residency findings, open decisions
- [`../docs/coding_standards.md`](../docs/coding_standards.md) — code style
- [`examples/README.md`](examples/README.md) — example catalog and GPU showcases
