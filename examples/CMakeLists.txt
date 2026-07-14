# Ported HAL example models. The visual ones use the reusable GridWindow
# helper (a UIGrid analogue) and can be run live or, with a "headless"
# argument, dumped to PNG/MP4. Headless-only demos link just the core libs.

# Shared window/render helper for the visual examples.
add_library(HalExamplesCommon
    common/GridWindow.cpp
    common/Mp4Recorder.cpp
    common/PlotWindow.cpp
    common/Scene3D.cpp
    common/InteractiveModel.cpp
)
target_include_directories(HalExamplesCommon
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(HalExamplesCommon
    PUBLIC
        HalWindow
)

# hal_add_example(<target> <source> [WINDOWED])
# WINDOWED examples link the GridWindow helper; headless ones link core only.
function(hal_add_example target source)
    cmake_parse_arguments(ARG "WINDOWED" "" "" ${ARGN})
    add_executable(${target} ${source})
    if (ARG_WINDOWED)
        target_link_libraries(${target} PRIVATE HalExamplesCommon)
    else()
        target_link_libraries(${target} PRIVATE HalCore)
    endif()
endfunction()

# --- Windowed model examples (live window or PNG/MP4 with "headless") ---
hal_add_example(GameOfLife GameOfLife.cpp WINDOWED)
hal_add_example(BirthDeath BirthDeath.cpp WINDOWED)
hal_add_example(ReactionDiffusion2D ReactionDiffusion2D.cpp WINDOWED)
hal_add_example(DivisionDeathMutation DivisionDeathMutation.cpp WINDOWED)
hal_add_example(TemplateOnLattice2D TemplateOnLattice2D.cpp WINDOWED)
hal_add_example(PopGrid PopGrid.cpp WINDOWED)
hal_add_example(BasicCellularAutomata BasicCellularAutomata.cpp WINDOWED)
hal_add_example(PDEexample PDEexample.cpp WINDOWED)
hal_add_example(HeatMaps HeatMaps.cpp WINDOWED)
hal_add_example(ColorInterpolation ColorInterpolation.cpp WINDOWED)
hal_add_example(PNGexample PNGexample.cpp WINDOWED)
hal_add_example(YCbCrTest YCbCrTest.cpp WINDOWED)
hal_add_example(Orbium Orbium.cpp WINDOWED)
hal_add_example(StemCellExample StemCellExample.cpp WINDOWED)
hal_add_example(PersianCarpets PersianCarpets.cpp WINDOWED)
hal_add_example(PhylogenyModel PhylogenyModel.cpp WINDOWED)
hal_add_example(ModuleSetExample ModuleSetExample.cpp WINDOWED)
hal_add_example(CompetitiveRelease CompetitiveRelease.cpp WINDOWED)
hal_add_example(TumorGrowth TumorGrowth.cpp WINDOWED)
hal_add_example(PlotExample PlotExample.cpp WINDOWED)
hal_add_example(BirthDeathOffLattice BirthDeathOffLattice.cpp WINDOWED)
hal_add_example(TemplateOffLattice2D TemplateOffLattice2D.cpp WINDOWED)
hal_add_example(Gas Gas.cpp WINDOWED)
hal_add_example(Chemotaxis Chemotaxis.cpp WINDOWED)
hal_add_example(OffLatticeExample OffLatticeExample.cpp WINDOWED)
hal_add_example(GIFtest GIFtest.cpp WINDOWED)
hal_add_example(ColorBarGenerator ColorBarGenerator.cpp WINDOWED)
hal_add_example(DiffusionAdvection3D DiffusionAdvection3D.cpp WINDOWED)
hal_add_example(MultiwellExample MultiwellExample.cpp WINDOWED)
hal_add_example(OutbreakWorld OutbreakWorld.cpp WINDOWED)
hal_add_example(MultiOutbreak MultiOutbreak.cpp WINDOWED)
hal_add_example(PopTumor PopTumor.cpp WINDOWED)
hal_add_example(OrbiumNPlayer OrbiumNPlayer.cpp WINDOWED)
hal_add_example(Mover3D Mover3D.cpp WINDOWED)
hal_add_example(Mover3DoffLattice Mover3DoffLattice.cpp WINDOWED)
hal_add_example(Vis3Dtest Vis3Dtest.cpp WINDOWED)
hal_add_example(Vis3DtestGpu Vis3DtestGpu.cpp WINDOWED)
hal_add_example(LineSegmentTest LineSegmentTest.cpp WINDOWED)
hal_add_example(TemplateMultiGrid TemplateMultiGrid.cpp WINDOWED)
hal_add_example(Example3D Example3D.cpp WINDOWED)
hal_add_example(SaveLoadModel SaveLoadModel.cpp WINDOWED)
hal_add_example(GuiExample GuiExample.cpp WINDOWED)
hal_add_example(ClinicianSim ClinicianSim.cpp WINDOWED)

# --- GPU compute showcases (visually striking, GPU-resident kernels) ---
hal_add_example(GpuReactionDiffusion GpuReactionDiffusion.cpp WINDOWED)
hal_add_example(GpuInkFlow GpuInkFlow.cpp WINDOWED)
hal_add_example(GpuSwarm GpuSwarm.cpp WINDOWED)
hal_add_example(GpuSlimeMold GpuSlimeMold.cpp WINDOWED)
hal_add_example(GpuWaveField GpuWaveField.cpp WINDOWED)
hal_add_example(GpuAttractor3D GpuAttractor3D.cpp WINDOWED)
hal_add_example(Gpu3DReactionDiffusion Gpu3DReactionDiffusion.cpp WINDOWED)
hal_add_example(GpuFlowField3D GpuFlowField3D.cpp WINDOWED)
hal_add_example(GpuSlimeMold3D GpuSlimeMold3D.cpp WINDOWED)

# --- Headless demos ---
hal_add_example(Gradient2D Gradient2D.cpp)
hal_add_example(Gradient3D Gradient3D.cpp)
hal_add_example(BinomialTest BinomialTest.cpp)
hal_add_example(ColorTest ColorTest.cpp)
hal_add_example(NonSpatialABM NonSpatialABM.cpp)
hal_add_example(FileIOTest FileIOTest.cpp)
hal_add_example(LockExample LockExample.cpp)
hal_add_example(GOLSweepEx GOLSweepEx.cpp)
