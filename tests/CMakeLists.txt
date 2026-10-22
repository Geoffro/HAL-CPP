include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)

FetchContent_MakeAvailable(doctest)

add_executable(HalTests
    TestSupport.cpp
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
    GpuTestSupport.cpp
    DiffusionKernel2DTests.cpp
    AgentForceKernel2DTests.cpp
    IAgentForceBackend2DTests.cpp
    AdiKernel2DTests.cpp
    GrayScottKernel2DTests.cpp
    GrayScott3DKernelTests.cpp
    WaveKernel2DTests.cpp
    FlowFieldKernel3DTests.cpp
    FluidKernel3DTests.cpp
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

# --- Coding standards checks -----------------------------------------------
# docs/coding_standards.md as ctest tests: the Python linter for rules
# clang-format/clang-tidy can't express, plus those tools where available.
set(HAL_STYLE_CHECK_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/examples
)

find_package(Python3 COMPONENTS Interpreter)

if (Python3_Interpreter_FOUND)
    add_test(NAME CodingStandardsCheck
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/check_coding_standards.py
                check ${HAL_STYLE_CHECK_DIRS}
                --root ${CMAKE_SOURCE_DIR}
    )
else()
    message(STATUS "python3 interpreter not found - CodingStandardsCheck test will not be registered")
endif()

find_program(CLANG_FORMAT_EXE NAMES clang-format)
find_program(CLANG_TIDY_EXE NAMES clang-tidy)

if (CLANG_FORMAT_EXE)
    file(GLOB_RECURSE HAL_CLANG_FORMAT_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.h
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/tests/*.cpp ${CMAKE_SOURCE_DIR}/tests/*.h
        ${CMAKE_SOURCE_DIR}/examples/*.cpp ${CMAKE_SOURCE_DIR}/examples/*.h
    )

    add_test(NAME ClangFormatCheck
        COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror -style=file
                ${HAL_CLANG_FORMAT_SOURCES}
    )
    # Disabled by default: the tree isn't clang-format clean. Run explicitly
    # with `ctest -R ClangFormatCheck --test-action Test`.
    set_tests_properties(ClangFormatCheck PROPERTIES DISABLED TRUE)
else()
    message(STATUS "clang-format not found - ClangFormatCheck test will not be registered")
endif()

if (CLANG_TIDY_EXE)
    add_test(NAME ClangTidyCheck
        COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
                ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
    )
    # Disabled by default: needs compile_commands.json from a Makefile/Ninja
    # configure, and the tree isn't clang-tidy clean.
    set_tests_properties(ClangTidyCheck PROPERTIES DISABLED TRUE)
else()
    message(STATUS "clang-tidy not found - ClangTidyCheck test will not be registered")
endif()
