# HAL C++ Examples

Ports of the Java `Examples/` and `LEARN_HERE/` models. Windowed models use
the reusable `common/GridWindow` helper (a `UIGrid`/`GridWindow` analogue):
run them with no arguments for a live Vulkan window, or pass `headless` to
run without a window and dump a PNG/MP4 instead. Plot demos use
`common/PlotWindow` (a pixel-grid line plotter, the `UIPlot` analogue). 3D
demos use `common/Scene3D` (a software 3D renderer — orbit camera + depth
sort over `GridWindow`, the `OpenGL3DWindow` analogue; drag the mouse to
orbit, up/down arrows to zoom). Headless-only demos just print to stdout.

## Building & running

```sh
cmake -S . -B build && cmake --build build -j4
./build/examples/GameOfLife            # live window
./build/examples/GameOfLife headless   # writes gameoflife.png
./build/examples/Gradient2D            # prints to stdout
```

Examples are built when `HAL_BUILD_EXAMPLES` and `HAL_BUILD_WINDOW` are ON
(both default ON). One extra tool lives alongside them:
`./build/HalBackendBenchmark` (headless — built with `HAL_BUILD_EXAMPLES`
alone) times the CPU vs GPU compute backends on shared diffusion and
agent-force workloads.

## GPU showcases

Nine examples exist to *show off* the GPU rather than to port a Java model —
large, deliberately artistic pieces. Most are GPU compute, where the whole
time-stepped loop stays resident on the device (the shape that actually
makes the GPU pay off; see `docs/DESIGN.md`); several also drive the hardware
3D renderer. Each 2D piece opens a live window, or writes a video plus a
final PNG with `headless`. If `ffmpeg` is on the PATH the video is a real
QuickTime-playable H.264 `.mp4` (upscaled with nearest-neighbor to keep the
crisp look); otherwise it falls back to the built-in MPEG-1 encoder written
as `.mpg` (play that one in VLC/ffmpeg). The 3D pieces need a live window (it
*is* the Vulkan surface) but record a clip when given a frame count, via the
renderer's opt-in frame capture.

| Example | GPU kernel | What you see |
|---|---|---|
| GpuReactionDiffusion | `GrayScottKernel2D` | 384² Gray-Scott reaction-diffusion: coral / maze / mitosis / spot patterns, many GPU-resident steps per frame. Presets `coral`, `maze`, `mitosis`, `spots`, `fingerprint`, `bubbles`, or it cycles. |
| GpuInkFlow | `AdiKernel2D` (implicit ADI) | Luminous "ink in water": three colored dye fields spread by the unconditionally-stable implicit solver, orbiting emitters braid them into a slow mandala. |
| GpuSwarm | `AgentForceKernel2D` | 150k particles: GPU bucketed neighbor repulsion gives the cloud texture, orbiting attractors gather it into bright braiding knots. Pass a number to change the count. |
| GpuSlimeMold | `DiffusionKernel2D` | Physarum transport network: 340k agents sense/steer/deposit a chemical trail that the GPU blurs and fades, self-organizing a living web. Palettes `gold`, `ember`, `spore`, `ice`. |
| GpuWaveField | `WaveKernel2D` | Ripple tank: driven point sources on a real 2D wave equation (leapfrog on the GPU), their wavefronts crossing into interference fringes. Patterns `duet`, `orbit`, `grid`, `chaos`. |
| GpuAttractor3D | (3D render) | A strange attractor as a 9k-particle 3D point cloud on the depth-tested Vulkan renderer, auto-orbiting. Systems `lorenz`, `aizawa`, `thomas`, `halvorsen`. |
| Gpu3DReactionDiffusion | `GrayScott3DKernel` + 3D render | VOLUMETRIC Gray-Scott: a 64³ reaction-diffusion evolved on the GPU, its V iso-surface drawn as tens of thousands of depth-tested spheres — a living 3D coral that slowly turns. Presets `coral`, `sponge`, `bubbles`, `maze`. |
| GpuFlowField3D | `FlowFieldKernel3D` + 3D render | A 120k-particle nebula advected through a chaotic 3D ABC flow on the GPU, drawn as depth-tested spheres tinted by speed — swirling luminous tubes and sheets. |
| GpuSlimeMold3D | `SlimeMold3DKernel` + 3D render | VOLUMETRIC Physarum — the 3D hybrid: 200k agents sense a diffusible chemical field in a 3D cone, steer up it, step, and deposit; the field diffuses/decays on the GPU and stays resident, so both the agents *and* the field live on the device. Grows a living 3D branching transport network. Palettes `gold`, `ember`, `spore`, `ice`. |

```sh
./build/examples/GpuReactionDiffusion coral        # live window, one preset
./build/examples/GpuReactionDiffusion headless      # gpu_reaction_diffusion.mp4 + .png
./build/examples/GpuInkFlow                         # live window
./build/examples/GpuSwarm 250000                    # live window, 250k particles
./build/examples/GpuSlimeMold headless ice          # gpu_slime_mold.mp4 + .png
./build/examples/GpuWaveField duet                  # live window, two-source interference
./build/examples/GpuAttractor3D aizawa              # live window (3D, self-orbiting)
./build/examples/GpuAttractor3D lorenz 300          # 3D window + records gpu_attractor3d.mp4 + .png
./build/examples/Gpu3DReactionDiffusion coral       # live 3D volumetric coral
./build/examples/Gpu3DReactionDiffusion coral 300   # + records gpu_3d_reaction_diffusion.mp4 + .png
./build/examples/GpuFlowField3D 300                 # 3D nebula, records gpu_flow_field_3d.mp4 + .png
./build/examples/GpuSlimeMold3D spore               # live 3D volumetric Physarum network
./build/examples/GpuSlimeMold3D spore 300           # + records gpu_slime_mold_3d.mp4 + .png
```

## Porting status

All 50 example programs are ported — every model in `Examples/` and
`LEARN_HERE/`:

| Example | Origin | Kind |
|---|---|---|
| GameOfLife | Examples/GameOfLife | windowed (on-lattice agents) |
| ClinicianSim | Examples/ClinicianSim | interactive (treatable tumor + plot) |
| SaveLoadModel | LEARN_HERE/Agents | windowed (agent-pool save/load) |
| GuiExample | LEARN_HERE/Other | interactive (parameter menu) |
| Example3D | Examples/_5Example3D | 3D (tumor + oxygen PDE) |
| Mover3D | LEARN_HERE/Agents | 3D (lattice random walk) |
| Mover3DoffLattice | LEARN_HERE/Agents | 3D (off-lattice walk) |
| TemplateMultiGrid | Examples/Template_Models | 3D (multi-grid + PDE) |
| Vis3Dtest | LEARN_HERE/Other | 3D (sphere/box smoke test) |
| Vis3DtestGpu | (hardware-renderer demo) | 3D (Vulkan `Scene3DRenderContext`) |
| LineSegmentTest | LEARN_HERE/Other | 3D (line segment) |
| PopTumor | Examples/PopulationTumor | windowed (population diffusion) |
| OrbiumNPlayer | Examples/Lenia | windowed (N-channel Lenia) |
| MultiOutbreak | Examples/MultiwellOutbreak | windowed (multi-well SIR) |
| OutbreakWorld | Examples/MultiwellOutbreak | windowed (SIR model) |
| MultiwellExample | LEARN_HERE/Agents | windowed (multi-well BirthDeath) |
| DiffusionAdvection3D | LEARN_HERE/Diffusibles | windowed (3D PDE slice) |
| GOLSweepEx | Examples/GameOfLife | headless (parallel sweep, CSV) |
| ColorBarGenerator | Examples/ColorBarGenerator | windowed (color bars) |
| GIFtest | LEARN_HERE/Other | headless (MP4 + PNG export) |
| FileIOTest | LEARN_HERE/Other | headless (mass file IO) |
| LockExample | LEARN_HERE/Other | headless (threads + lock) |
| Orbium | Examples/Lenia | windowed (Lenia / FFT) |
| Chemotaxis | Examples/Chemotaxis | windowed (off-lattice + PDE) |
| OffLatticeExample | Examples/_3OffLatticeExample | windowed (off-lattice, CSV) |
| Gas | Examples/GasPopulationGrid | windowed (off-lattice density) |
| BirthDeathOffLattice | LEARN_HERE/Agents | windowed (off-lattice agents) |
| TemplateOffLattice2D | Examples/Template_Models | windowed (off-lattice skeleton) |
| StemCellExample | Examples/_2StemCellExample | windowed (on-lattice agents) |
| PersianCarpets | Examples/PersianModel | windowed (evolutionary game) |
| PhylogenyModel | Examples/PhylogenyTracker | windowed (phylogeny + CSV export) |
| ModuleSetExample | Examples/ModuleSetExample | windowed (ModuleSet / VarSet) |
| CompetitiveRelease | Examples/_6CompetitiveRelease | windowed (agents + drug PDE, CSV) |
| TumorGrowth | Examples/ODEmodel | plot (ODE + PlotWindow) |
| PlotExample | LEARN_HERE/Other | plot (PlotWindow) |
| BirthDeath | LEARN_HERE/Agents | windowed (on-lattice agents) |
| DivisionDeathMutation | Examples/_1DivisionDeathMutation | windowed (on-lattice agents) |
| TemplateOnLattice2D | Examples/Template_Models | windowed (on-lattice skeleton) |
| BasicCellularAutomata | Examples/BasicCellularAutomata | windowed (elementary CA) |
| ReactionDiffusion2D | LEARN_HERE/Diffusibles | windowed (PDE) |
| PDEexample | Examples/_4PDEexample | windowed (agents + PDE) |
| PopGrid | LEARN_HERE/Agents | windowed (population grid) |
| HeatMaps | LEARN_HERE/Other | windowed (color palettes) |
| ColorInterpolation | LEARN_HERE/Other | windowed (color) |
| PNGexample | LEARN_HERE/Other | headless (PNG export) |
| YCbCrTest | LEARN_HERE/Other | windowed (color) |
| Gradient2D | LEARN_HERE/Diffusibles | headless (PDE) |
| Gradient3D | LEARN_HERE/Diffusibles | headless (PDE) |
| BinomialTest | LEARN_HERE/Other | headless (RNG) |
| ColorTest | LEARN_HERE/Other | headless (color) |
| NonSpatialABM | LEARN_HERE/Other | headless (mean-field ABM) |

Nothing left to port. The last three each needed a supporting piece that was
added with them: `SaveLoadModel` uses agent-pool serialization
(`AgentGrid2D::SaveAgents`/`LoadAgents`); `GuiExample` uses new `GuiContext`
string/combo widgets; `ClinicianSim` uses a reduced software port of Java's
`InteractiveModel` (`common/InteractiveModel` + `common/TreatableTumor`) —
keyboard-toggled treatment plus a live population plot, in place of the full
mouse-driven treatment-timeline editor.

## Library gaps found while porting

- `Hal::CircleHood(includeOrigin, radius)` — added to `Neighborhood`.
- `AgentGrid2D::MapEmptyHood` / `MapOccupiedHood` — added.
- `Hal::MultinomialCalc` — added to `Tools` (also unblocks
  `PopulationGrid2D::Diffusion` and `PopTumor`).
- Named color constants `Hal::RED/GREEN/BLUE/BLACK/WHITE/YELLOW/CYAN/MAGENTA`
  (from Java `Util`) — added to `Color`. (Note: `HalColorSchemes.jar` in the
  Java repo is unrelated — it's a bundle of IntelliJ editor themes, not part
  of the simulation color API.)
- `Hal::ProbScale(prob, duration)` — added to `MathUtil` (per-timestep rate
  rescaling used by CompetitiveRelease).
- `AgentGrid2D::PopAt(i)` / `PopAt(x, y)` — per-square occupancy count, and
  `GridWindow::Circle(x, y, r, color)` — a filled-disc rasterizer (the
  `OpenGL2DWindow.Circle` analogue) for drawing off-lattice agents.
- `Hal::RectangleHood(includeOrigin, radX, radY)` — added to `Neighborhood`
  (the travel neighborhood in the outbreak model). `GridWindow::SetPixels`
  blits a pre-composited buffer (e.g. `MultiWellExperiment::Pixels()`).
- `AgentGrid2D::SaveAgents` / `LoadAgents` — reflection-free agent-pool
  serialization (per-agent field callbacks), the last deferred serialization
  piece; unblocks `SaveLoadModel`.
- `GuiContext::StringInput` / `ComboBox` — text and dropdown widgets for
  `GuiExample`. `common/Scene3D` (software 3D) and `common/InteractiveModel`
  + `common/TreatableTumor` (interactive treatment driver) round out the
  reusable helpers.
- `PopulationGrid2D::Diffusion(diffRate, MultinomialCalc&)` — mean-field
  multinomial diffusion (needed by PopTumor), the last previously-deferred
  `PopulationGrid2D` method.
- `AgentGrid3D::MapEmptyHood` (parallel to the 2D version).
- `common/Scene3D` — a software 3D renderer (orbit camera + painter's-
  algorithm depth sort over `GridWindow`) for the 3D examples. The
  hardware-accelerated Vulkan version now exists as
  `Hal::Gpu::Scene3DRenderContext` (depth-tested instanced spheres +
  lines, Lambert lighting, drag-to-orbit / scroll-to-zoom via `Window`) —
  see the `Vis3DtestGpu` demo, which mirrors `Vis3Dtest` on the hardware
  path.
