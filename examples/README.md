# HAL C++ Examples

Example models built on the library. Windowed models use the reusable
`common/GridWindow` helper: run them with no arguments for a live Vulkan
window, or pass `headless` to run without a window and dump a PNG/MP4
instead. Plot demos use `common/PlotWindow` (a pixel-grid line plotter). 3D
demos use `common/Scene3D` (a software 3D renderer with an orbit camera and
depth sort over `GridWindow`; drag the mouse to orbit, up/down arrows to
zoom). Headless-only demos print to stdout.

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

Twelve examples exist to *show off* the GPU: large, deliberately artistic
pieces. Most are GPU compute, where the whole time-stepped loop stays resident
on the device (the shape that actually makes the GPU pay off; see
`docs/DESIGN.md`); several also drive the hardware 3D renderer. Each 2D piece opens a live window, or writes a video plus a
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
| GpuPlasmaVortex | `AgentForceKernel2D` + `DiffusionKernel2D` | Abstract electric-plasma storm, not biological: 200k charged sparks torn between four counter-spinning neon vortices (magenta/cyan/gold/violet), GPU neighbor repulsion for texture, three independent GPU-resident color fields blooming and fading where filaments cross into white-hot cores. The 2D hybrid pushed hardest — both agents *and* field (x3 channels) are GPU-resident every frame. |
| GpuTempest | `WaveKernel2D` + `AgentForceKernel2D` + `DiffusionKernel2D` | A three-kernel hybrid flipped from the others: the PDE drives the agents, not the reverse. A real 2D wave equation is driven by point sources that spawn and die at random (unseeded RNG, no preset, no two runs alike) into wild concentric/maze interference; 150k amber motes climb the local wave gradient on the GPU-spaced host loop, each firing a decaying white flash — chains of lightning — whenever a wavefront sweeps past. |
| GpuSupernova | `FlowFieldKernel3D` + `FluidKernel3D` + 3D render | A planet-killer explosion: blinding core flash with star-filter spikes, a soft blue-white shock shell, the signature flat blast ring on a randomly tilted plane, and 200k star-like debris sparks fracturing outward in clustered fiery plumes — churned by GPU ABC turbulence, streaking flame licks, cooling white → orange → ember. The smoke/fire is a real GPU Navier-Stokes fluid (`FluidKernel3D`, Stam's stable fluids: semi-Lagrangian advection + Jacobi pressure projection) rendered as additive soft-glow fog (the renderer's `SphereGlow` pipeline), and every subsystem's radius rides one shared drag-decelerated ballistic front so the whole blast expands as one object. Camera shake on detonation; re-detonates on a cycle, differently every time (unseeded RNG). |

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
./build/examples/GpuPlasmaVortex                    # live window
./build/examples/GpuPlasmaVortex 300000             # live window, 300k sparks
./build/examples/GpuTempest                         # live window
./build/examples/GpuTempest headless                # gpu_tempest.mp4 + .png
./build/examples/GpuSupernova                       # live 3D window (re-detonates forever)
./build/examples/GpuSupernova 430                   # one full blast, records gpu_supernova.mp4 + .png
```

## Example catalog

| Example | Kind |
|---|---|
| GameOfLife | windowed (on-lattice agents) |
| ClinicianSim | interactive (treatable tumor + plot) |
| SaveLoadModel | windowed (agent-pool save/load) |
| GuiExample | interactive (parameter menu) |
| Example3D | 3D (tumor + oxygen PDE) |
| Mover3D | 3D (lattice random walk) |
| Mover3DoffLattice | 3D (off-lattice walk) |
| TemplateMultiGrid | 3D (multi-grid + PDE) |
| Vis3Dtest | 3D (sphere/box smoke test) |
| Vis3DtestGpu | 3D (Vulkan `Scene3DRenderContext`) |
| LineSegmentTest | 3D (line segment) |
| PopTumor | windowed (population diffusion) |
| OrbiumNPlayer | windowed (N-channel Lenia) |
| MultiOutbreak | windowed (multi-well SIR) |
| OutbreakWorld | windowed (SIR model) |
| MultiwellExample | windowed (multi-well BirthDeath) |
| DiffusionAdvection3D | windowed (3D PDE slice) |
| GOLSweepEx | headless (parallel sweep, CSV) |
| ColorBarGenerator | windowed (color bars) |
| GIFtest | headless (MP4 + PNG export) |
| FileIOTest | headless (mass file IO) |
| LockExample | headless (threads + lock) |
| Orbium | windowed (Lenia / FFT) |
| Chemotaxis | windowed (off-lattice + PDE) |
| OffLatticeExample | windowed (off-lattice, CSV) |
| Gas | windowed (off-lattice density) |
| BirthDeathOffLattice | windowed (off-lattice agents) |
| TemplateOffLattice2D | windowed (off-lattice skeleton) |
| StemCellExample | windowed (on-lattice agents) |
| PersianCarpets | windowed (evolutionary game) |
| PhylogenyModel | windowed (phylogeny + CSV export) |
| ModuleSetExample | windowed (ModuleSet / VarSet) |
| CompetitiveRelease | windowed (agents + drug PDE, CSV) |
| TumorGrowth | plot (ODE + PlotWindow) |
| PlotExample | plot (PlotWindow) |
| BirthDeath | windowed (on-lattice agents) |
| DivisionDeathMutation | windowed (on-lattice agents) |
| TemplateOnLattice2D | windowed (on-lattice skeleton) |
| BasicCellularAutomata | windowed (elementary CA) |
| ReactionDiffusion2D | windowed (PDE) |
| PDEexample | windowed (agents + PDE) |
| PopGrid | windowed (population grid) |
| HeatMaps | windowed (color palettes) |
| ColorInterpolation | windowed (color) |
| PNGexample | headless (PNG export) |
| YCbCrTest | windowed (color) |
| Gradient2D | headless (PDE) |
| Gradient3D | headless (PDE) |
| BinomialTest | headless (RNG) |
| ColorTest | headless (color) |
| NonSpatialABM | headless (mean-field ABM) |
