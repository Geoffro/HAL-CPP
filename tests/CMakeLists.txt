include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)

FetchContent_MakeAvailable(doctest)

add_executable(HalTests
    LoggerTests.cpp
    MathUtilTests.cpp
    ColorTests.cpp
    NeighborhoodTests.cpp
    RandTests.cpp
    GridTests.cpp
    AgentGridTests.cpp
    AgentPT2DTests.cpp
    SphericalAgentTests.cpp
    AgentGrid3DTests.cpp
    GridGeometry3DTests.cpp
    SphericalAgent3DTests.cpp
    PopulationGridTests.cpp
    MultinomialCalcTests.cpp
    PDEGridTests.cpp
    PDEADITests.cpp
    IDiffusionBackend2DTests.cpp
    SerializationTests.cpp
    VideoExportTests.cpp
    TripleBufferTests.cpp
    OdeSolverTests.cpp
    PhylogenyTrackerTests.cpp
    MultiThreadTests.cpp
    FftTests.cpp
    FftGridTests.cpp
    Lenia1PlayerTests.cpp
    LeniaNPlayerTests.cpp
    ModularityTests.cpp
    MultiWellExperimentTests.cpp
    ComputeContextTests.cpp
    DiffusionKernel2DTests.cpp
    AgentForceKernel2DTests.cpp
    IAgentForceBackend2DTests.cpp
    AdiKernel2DTests.cpp
    GrayScottKernel2DTests.cpp
    GrayScott3DKernelTests.cpp
    WaveKernel2DTests.cpp
    FlowFieldKernel3DTests.cpp
    SlimeMold3DKernelTests.cpp
    Math3DTests.cpp
)

find_package(Threads REQUIRED)

target_link_libraries(HalTests
    PRIVATE
        HalCore
        HalGpu
        doctest::doctest
        Threads::Threads
)

target_include_directories(HalTests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
)

if (HAL_BUILD_WINDOW)
    target_sources(HalTests PRIVATE WindowRenderContextTests.cpp Scene3DRenderContextTests.cpp)
    target_link_libraries(HalTests PRIVATE HalWindow)
    target_compile_definitions(HalTests PRIVATE HAL_BUILD_WINDOW)
endif()

if (HAL_BUILD_GUI)
    target_sources(HalTests PRIVATE GuiContextTests.cpp)
    target_link_libraries(HalTests PRIVATE HalGui)
    target_compile_definitions(HalTests PRIVATE HAL_BUILD_GUI)
endif()

add_test(NAME HalTests COMMAND HalTests)
