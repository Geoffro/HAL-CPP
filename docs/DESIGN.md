# HAL C++/Vulkan — Design

## Attribution

This project is a C++/Vulkan port of the **Hybrid Automata Library (HAL)**, a
framework for hybrid (agent-based + PDE) simulation developed for mathematical
oncology at the H. Lee Moffitt Cancer Center.

- Original project: <https://github.com/MathOnco/HAL> · <https://halloworld.org/>
- Paper: Bravo et al., *PLOS Computational Biology* (2020),
  [10.1371/journal.pcbi.1007635](https://doi.org/10.1371/journal.pcbi.1007635)
- License: MIT, Copyright (c) 2017 Rafael Ramon Bravo (see `LICENSE`)

## Goals

The per-tick agent and PDE loops are the expensive part of a HAL model, and
they are data-parallel. The library is built so those loops can run on the GPU
(Vulkan compute) as well as the CPU, without changing model code.

HAL is a **library, not an engine**: a model owns a plain `main()` and calls
HAL each tick. There is no framework-owned loop or callback lifecycle.

## Core data structures

- **Value grids** (`Grid1D/2D/3D<T>`): flat arrays with per-axis wrap flags.
  Index `i = x * yDim + y` in 2D and `(x * yDim + y) * zDim + z` in 3D.
- **PDE grids**: the committed field plus a delta buffer; operators and
  `Set`/`Add` write deltas, and `Update()` commits them. Each cell reads only
  the committed field and writes only its own delta, which is the shape of a
  compute-shader stencil (one invocation per cell).
- **Population grids**: counts with the same delta/commit pattern, iterating
  occupied cells through a linked list when occupancy is sparse.
- **Agent neighbor search**: agents are bucketed into grid squares; radius
  queries walk the squares overlapping the query circle (a uniform-grid cell
  list). The GPU builds the same buckets per dispatch with a count, prefix
  sum, and scatter.

## Agents

An agent grid is a template on the agent type, and new agents are constructed
as `TAgent(TGrid*)`. Agents live in a `std::deque` pool with a dead-agent free
list and in-place reuse, so agent pointers stay valid as the pool grows.
Iteration skips agents born after it started (each agent records the pool's
iteration state id at birth).

Agents are written against their *concrete* grid, so the grid takes itself as
a CRTP parameter: `class GOLGrid : public AgentGrid2D<GOLAgent, GOLGrid>`.
Because agent and grid refer to each other, `TGrid` is incomplete when an
agent base class is instantiated. The rule in `include/Hal/Agents/`: **never
name `TGrid::AgentType` in a member declaration, field type, or return
type.** Store base-typed pointers (`AgentPT2D<TGrid>*`) and downcast inside
function bodies; template methods on the pointer type (parameters) or on a
defaulted `TSelf = TGrid` parameter (return types) so the lookup happens at
the first call. Examples: `GOLAgent`/`GOLGrid` in `tests/AgentGridTests.cpp`
and `SphericalAgent2D::Divide`.

## Modules

`Hal::ModuleSet<TModule, THook>` composes optional model behavior from
modules. Each module declares the hooks it implements when added
(`AddModule(module, {Hook::Setup, Hook::OnDiv, ...})`), and `Iter(hook)`
visits only those modules. `Hal::VarSetManager` / `Hal::IVarSet` give each
module its own slots in a per-agent parameter array.

## Save/load

Save/load is a raw native-endian binary dump with no format header or
version: grids write their dimensions, wrap flags, and arrays; `Rand` writes
its generator state; `AgentGrid2D::SaveAgents` writes the tick and each
agent's square and birth tick around a per-agent callback for model fields.
Files are meant to be reloaded on the same platform. `BinaryReader` logs any
failed read, and grid and agent loads then throw rather than building from
garbage, as they also do for invalid dimensions or square indices.

## CPU/GPU backends

Compute-heavy operations sit behind small strategy interfaces with CPU and
Vulkan implementations that take flat, caller-owned arrays:

- `IDiffusionBackend2D`: explicit stencil
- `IAgentForceBackend2D`: cell-bucket neighborhood forces (the CPU version
  uses the same four passes as the GPU kernel)
- `IAdiDiffusionBackend2D`: implicit ADI (Thomas sweeps on the CPU, parallel
  cyclic reduction on the GPU; both handle periodic axes via Sherman-Morrison)

`HalBackendBenchmark` times each CPU vs. GPU.

GPU memory goes through VMA (owned by `ComputeContext`): device-local storage
buffers plus a persistently mapped staging buffer per kernel, with upload,
compute, and readback in one fence-synchronized submission.

Because successive time steps depend on each other, overlapping frames in
flight doesn't help; keeping the loop **on the GPU** does.
`DiffusionKernel2D::RunSteps` chains N stencil + commit passes in one command
buffer, measured at 18–30× over the CPU, versus ~0.6× for the same stencil
stepped one call at a time through the round-tripping interface.
`AdiKernel2D`, `GrayScottKernel2D`, `WaveKernel2D`, and the 3D kernels
(`GrayScott3DKernel`, `FlowFieldKernel3D`, `SlimeMold3DKernel`,
`FluidKernel3D`) follow the same pattern. `FluidKernel3D` is a stable-fluids
smoke/fire solver (Stam 1999: semi-Lagrangian advection, buoyancy, Jacobi
pressure projection) whose velocity and pressure never leave the device; only
the density and temperature round-trip. These drive the GPU showcases in
`examples/README.md`.

## RNG

`Hal::Rng` is xoshiro256** seeded by splitmix64, with `Split()` deriving
independent streams. `Hal::Rand` adds distribution sampling (Gaussian via the
polar method; binomial via Bernoulli sums, inversion, or BTPE depending on n
and p).

## Threading

`Hal::MultiThread` runs independent replicates or sweep points across
threads. For simulation/render separation, `Hal::TripleBuffer` hands state
from the simulation thread to the render loop: the producer writes and
publishes a back buffer, the consumer acquires the newest published one, and
neither blocks or touches the other's slot.

## Rendering

Two separate render classes, since they share almost no pipeline state:

- `Hal::Gpu::WindowRenderContext`: a fixed-size texture drawn across the
  window (the pixel-grid view), with an overlay hook for the ImGui layer.
- `Hal::Gpu::Scene3DRenderContext`: depth-tested 3D with an instanced
  unit-sphere mesh, line lists, additive glow, and Lambert shading. It mirrors
  the software `examples/common/Scene3D` API and camera defaults.

Each owns its own Vulkan instance and device, separate from `ComputeContext`.
Matrix math is a small `Mat4`/`Vec3` (`Math3D.h`) with Vulkan clip-space
conventions (depth in [0,1], y down).

## Callback style

- **Per-call hot-loop callbacks** (neighborhood hooks like
  `MapHood`/`ApplyHood`, PDE boundary conditions, force laws, ODE
  derivatives, agent predicates) are template parameters deduced at the call
  site, so captured lambdas are inlined with no type erasure.
- **Stored, runtime-replaceable callbacks** (Lenia kernel and growth
  functions) are `std::function`.

## Logging and error handling

Vulkan, file IO, and export failures are never swallowed: they go through the
severity-leveled `Hal::Logger`, then either throw (Vulkan setup, corrupt grid
or agent loads) or return false (PNG/video export). API misuse, such as
mismatched array sizes, throws.

## Numerical methods

- Explicit finite-difference diffusion (3/5/7-point stencils, uniform or
  per-cell rates) and first-order upwind advection (uniform velocity, or
  conservative flux form for per-cell velocity fields), with CFL checks.
- Implicit ADI diffusion via the Thomas algorithm, one tridiagonal solve per
  line. Dirichlet boundaries use a reflecting ghost cell on both the
  explicit and implicit halves, so a field equal to the boundary value is a
  fixed point.
- ODEs: Euler, classical RK4, and adaptive Runge-Kutta-Fehlberg 4(5).

## Video export

`Hal::VideoWriter` writes MPEG-1 through the vendored single-header
`jo_mpeg`, keeping the core free of codec dependencies. That stream doesn't
play in QuickTime (use VLC/ffmpeg), so the example-layer recorder
(`examples/common/Mp4Recorder`) writes H.264 MP4 through `ffmpeg` when it is
on the PATH and falls back to `VideoWriter` otherwise. The 3D renderer's
opt-in frame capture (`SetCaptureEnabled`/`ReadCapturedFrame`) feeds the same
recorder.

## Open decisions

- Python bindings: pybind11 is planned for Python-driven parameter sweeps.
- Sharing one Vulkan instance/device between compute and rendering.
