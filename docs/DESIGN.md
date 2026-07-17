# HAL C++/Vulkan Port — Design Doc

## Attribution

This project is a C++/Vulkan port of the **Hybrid Automata Library (HAL)**, a
Java framework for hybrid (agent-based + PDE) simulation developed for
mathematical oncology at the H. Lee Moffitt Cancer Center.

- Original project: <https://github.com/MathOnco/HAL> · <https://halloworld.org/>
- Paper: Bravo et al., *PLOS Computational Biology* (2020),
  [10.1371/journal.pcbi.1007635](https://doi.org/10.1371/journal.pcbi.1007635)
- License: MIT, Copyright (c) 2017 Rafael Ramon Bravo (see `hal-cpp/LICENSE`)

## Why

HAL's core simulation loops are single-threaded — Java's only parallelism is
coarse, replicate-level task parallelism (`Util.MultiThread`); the per-tick
agent/PDE loops never got fine-grained parallelism. Adding real GPU compute for
those loops is the point of this rewrite, not just a language change (a C++
preference is the secondary motive).

HAL is a **library, not an engine**: models own a plain `main()` and call HAL
APIs procedurally each tick — no inversion of control, no framework-owned loop.
This fits C++ well and is preserved exactly; the port introduces no
engine/callback lifecycle that didn't already exist.

## What ports directly vs. what needs redesign

Most of HAL's core data structures are already data-oriented and embarrassingly
parallel, so they translate to C++ (and later, compute shaders) with little
structural change:

- **Value grids** (`Grid1D/2D/3D<T>`): flattened arrays with wraparound flags —
  a direct port.
- **PDE grids**: double-buffered "current field + delta accumulator, commit on
  `Update()`". Each cell reads only neighbor values from the read-only current
  field and writes only its own delta slot — already the shape of a
  compute-shader stencil kernel, one invocation per cell.
- **Agent neighbor search**: agents are bucketed into square-grid cells (each a
  linked list); radius queries walk the bounding box of overlapping cells. This
  is a standard cell-list / uniform spatial hash — the same technique used in GPU
  N-body/molecular-dynamics codes — porting directly to CPU and later to a
  GPU-built bucket structure (counting sort + prefix sum per tick).

Three Java-specific mechanisms have **no direct C++ equivalent** and are
deliberately redesigned:

1. **Reflection-based agent construction.** Java grids take a `Class<T>` and
   instantiate agents via `newInstance()`, reflectively injecting the owning
   grid. The port makes the agent type a compile-time template parameter of
   `AgentGrid<T>`, constructed by an explicit `T(Grid*)` constructor. Java needed
   reflection here only because generics are erased at runtime; C++ templates
   aren't, so the mechanism is simply unnecessary. The dead-agent free-list /
   in-place-reuse pooling is kept (good for cache locality and stable GPU upload
   buffers), with ownership by raw pointer/index into the pool. Agent storage
   uses `std::deque<TAgent>` — it never invalidates references on growth (vector
   would), matching the stability Java gets from an `ArrayList` of references.

   *Implementation note (CRTP + C++ completeness):* a HAL agent is written
   against its *concrete* grid type (reaching grid-specific fields), so
   `AgentGrid2D<TAgent, TDerived>` takes the derived grid as a CRTP parameter
   (`class GOLGrid : public AgentGrid2D<GOLAgent, GOLGrid>`). Because the agent
   and grid reference each other by value/pointer, C++ hits a real
   circular-completeness requirement that Java (reference semantics, order-
   independent) never does. The rule followed throughout
   `hal-cpp/include/Hal/Agents/`: **never spell `TGrid::AgentType` in a member
   declaration, field, or return type** — those force nested-type lookup at
   class-instantiation time, when `TGrid` is still incomplete. Store
   self-referential `AgentPT2D<TGrid>*` for linked-list fields and downcast
   inside bodies; template methods on the pointer type (params) or on a
   `TSelf = TGrid` defaulted parameter (return types) so lookup defers to the
   first call site. Worked examples: `GOLAgent`/`GOLGrid` in
   `tests/AgentGridTests.cpp`; `SphericalAgent2D::Divide` for the return-type case.

2. **Reflection-driven plugin dispatch (`ModuleSetManager`).** Java inspects a
   module's overridden methods via `getDeclaredMethods()` to decide which
   lifecycle hooks to invoke. The port (`Hal::ModuleSet<TModule, THook>`) instead
   *declares* the hooks a module implements when it is added
   (`AddModule(module, {Hook::Setup, Hook::OnDiv, ...})`), bucketing modules per
   hook so `Iter(hook)` still visits only the relevant ones — same behavior,
   explicit instead of reflected. The module base is an ordinary class with named
   virtual hooks. `Hal::VarSetManager` / `Hal::IVarSet` port the per-module
   parameter-slot allocator unchanged.

3. **Save/load via Java object serialization.** Java serializes the live object
   graph with `ObjectOutputStream`, then needs `SetupConstructors()` to rebuild
   reflection handles that can't survive serialization. Since the port never uses
   reflection for construction, this disappears — save/load is a plain versioned
   binary dump (raw arrays + small header).

The GUI (Swing + immediate-mode OpenGL, ~5.2K LOC) is not portable and is
replaced wholesale by Vulkan + GLFW + Dear ImGui/ImPlot. Only its *API shape* is
replicated: pixel-writable grid surface, line plotting, palettes/heatmaps,
parameter controls, and PNG/GIF export.

## CPU/GPU backend abstraction

Compute-heavy operations sit behind small strategy interfaces with CPU and
Vulkan-compute implementations, so model code doesn't change with the active
backend and CPU-vs-GPU comparison/timing is straightforward. As built these are
three sibling interfaces (flat caller-owned arrays), not one god-interface:

- `IDiffusionBackend2D` — explicit stencil (CPU + GPU)
- `IAgentForceBackend2D` — cell-bucket neighborhood forces (CPU + GPU; the CPU
  path mirrors the GPU kernel's four bucket passes)
- `IAdiDiffusionBackend2D` — implicit ADI (CPU Thomas sweeps + GPU PCR, wrapped
  axes via the same Sherman-Morrison decomposition)

`HalBackendBenchmark` times any of them CPU-vs-GPU through the tests' call sites.

GPU memory goes through VMA (owned by `ComputeContext`): device-local SSBOs plus
a persistently mapped staging buffer per kernel, with upload/compute/readback in
a single fence-synchronized submission. For time-stepped compute, frames-in-
flight pipelining is replaced by **GPU residency**: since successive steps depend
on each other, `DiffusionKernel2D::RunSteps` chains N stencil+commit pass pairs
in one command buffer — measured at 18–30× over CPU here, versus ~0.6× for the
same stencil stepped one-at-a-time through the round-tripping interface. That
gap, not raw kernel speed, is the design lesson: **the GPU pays off when the loop
lives there.**

The same principle drives `AdiKernel2D::RunSteps`, `GrayScottKernel2D::RunSteps`
(reaction-diffusion), `WaveKernel2D::RunSteps` (leapfrog wave equation), and the
3D siblings `GrayScott3DKernel`, `FlowFieldKernel3D`, and `FluidKernel3D` — the
last an incompressible stable-fluids smoke/fire solver (Stam 1999:
semi-Lagrangian advection, buoyancy, Jacobi pressure projection) whose velocity
and pressure fields never leave the device across calls; only the density/
temperature scalars round-trip for seeding and rendering. These back a set of
visual showcases (`GpuReactionDiffusion`, `GpuInkFlow`, `GpuSwarm`,
`GpuSlimeMold`, `GpuWaveField`, and hardware-rendered `GpuAttractor3D`,
`Gpu3DReactionDiffusion`, `GpuFlowField3D`, `GpuSupernova`; see
`examples/README.md`) — the first examples designed around the GPU rather than
ported from Java, leaning on residency for compute and, for 3D, on tens of
thousands of depth-tested instanced spheres plus an additive depth-read-only
glow pipeline (`Scene3DRenderContext::SphereGlow`) for volumetric fog and fire.

## RNG

Java wraps a vendored `SplittableRandom` for cheap independent streams (important
for per-agent generation). The port keeps that *design goal* but implements it
with a standard modern algorithm (splitmix64 / xoshiro256**). Bit-exact Java
parity is not a goal; statistical correctness is verified instead.

## Threading: simulation vs. rendering

Java mutates Swing components directly from the simulation thread — unsafe per
Swing's own contract, but works for HAL's simple examples. The port defines this
properly: the sim thread writes double-buffered state; the render loop reads a
consistent snapshot.

## Rendering contexts: parallel classes, not one generalized renderer

Two deliberately parallel render classes rather than one abstracted renderer:

- `Hal::Gpu::WindowRenderContext` — 2D pixel-grid blitter (the `UIGrid`
  equivalent): fixed textured quad, no depth, no MVP.
- `Hal::Gpu::Scene3DRenderContext` — hardware 3D renderer (`OpenGL3DWindow`
  successor): depth testing, an instanced unit-sphere mesh + line-list geometry,
  an MVP push constant, Lambert shading.

They share almost no pipeline state, so a shared abstraction would be all
conditionals; each owns its own instance/device/swapchain (unification remains a
legitimate follow-up). The 3D context mirrors the software `examples/common/Scene3D`
API and camera defaults so a model can swap renderers without rereading scene
code. Matrix math is a small hand-rolled `Mat4`/`Vec3` (`Math3D.h`), not GLM,
with Vulkan clip-space conventions (depth [0,1], y down) baked in.

## Callback style

Java leans on ~89 tiny `@FunctionalInterface` types for callbacks. Rather than a
generic `std::function` everywhere, the port uses named function-pointer typedefs
or small single-purpose callable structs per callback category (boundary
condition, force response, agent predicate), mirroring the Java categories. Plain
lambdas only where there's a clear local advantage.

*Amendment (Phase 2):* neighborhood-iteration hooks (`MapHood`/`ApplyHood`/…)
need to capture local state, which function pointers can't. These are instead
templated on the caller's callable (deduced, not type-erased) — the compiler
inlines a dedicated instantiation per call site, so it costs nothing at runtime
and isn't the generic-`std::function` abstraction the standards warn against
(same technique as `std::sort`'s comparator). Named function pointers remain
right for callbacks *stored* across many calls (e.g. a PDE grid's boundary
condition held as a field).

## Logging and error handling

Vulkan, file IO, and export failures must never fail silently. A small
severity-leveled logger (Phase 0, since Phase 4's Vulkan setup depends on it) is
used at every failure path instead of swallowing errors or relying on assertions.

## Numerical methods carried over

- Explicit finite-difference diffusion (5/7-point stencil) and first-order upwind
  advection, same CFL constraints as Java.
- ADI implicit diffusion via the Thomas algorithm (tridiagonal solve per
  row/column/pencil) — unconditionally stable, serial per pencil but parallel
  across pencils, and the candidate for GPU cyclic reduction / PCR.

*Correction (Phase 3):* an earlier research pass claimed Java's 3D ADI was dead
code. It conflated a genuinely abandoned commented-out attempt in
`PDEequations.java` with the real, live `Diffusion3DADI` in `ADIequations.java`
(called by `PDEGrid3D.DiffusionADI`). The port ported the real one — 3D ADI did
not need fresh design. A reminder to re-verify "dead code" claims against the
call graph before scoping around them.

## Improvements over the Java version

- Structure-of-arrays / data-oriented layout chosen upfront for GPU upload.
- Compile-time-typed agents via templates — no runtime reflection, no erasure.
- Real GPU compute for the per-tick agent/PDE loops (new capability).
- Simpler save/load format (a consequence of not needing reflection).
- Explicit, correct sim/render threading contract.
- pybind11 instead of Py4J's socket bridge for Python-driven parameter sweeps.

## Open design decisions

- Whether ADI gets a GPU cyclic-reduction solver or stays CPU-only —
  **deferred to end of project** per user decision (with the CPU/GPU diffusion
  timing comparison).
- ~~GIF vs. MP4/frame-dump export~~ **Decided**: built-in MPEG-1 via a vendored
  single-header encoder (`third_party/jo_mpeg`) for the zero-dependency core
  `Hal::VideoWriter`. Caveat: that raw MPEG-1 elementary stream won't play in
  QuickTime (use VLC/ffmpeg). The example-layer recorder
  (`examples/common/Mp4Recorder`) therefore prefers a real H.264 MP4 when
  `ffmpeg` is on the PATH and falls back to the built-in encoder (`.mpg`)
  otherwise — the core keeps its zero-dependency guarantee; shareable video is
  opt-in. The hardware 3D renderer exposes opt-in frame capture
  (`SetCaptureEnabled`/`ReadCapturedFrame`) to feed the same recorder.
- CRTP vs. virtual dispatch for the agent hierarchy — pick whichever keeps the
  neighbor-search hot path non-virtual (Phase 2).
