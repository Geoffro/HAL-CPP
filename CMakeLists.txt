cmake_minimum_required(VERSION 3.20)

project(HalCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile_commands.json for clang-tidy (tests/CMakeLists.txt's ClangTidyCheck).
# Generators that don't support it (e.g. Xcode) silently ignore this.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HAL_BUILD_TESTS "Build the Hal unit test suite" ON)
option(HAL_BUILD_WINDOW "Build the GLFW + Vulkan windowed rendering backend" ON)
option(HAL_BUILD_GUI "Build the Dear ImGui + ImPlot GUI layer" ON)
option(HAL_BUILD_EXAMPLES "Build the HAL example models" ON)

add_library(HalCore
    src/Log/Logger.cpp
    src/IO/BinaryWriter.cpp
    src/IO/BinaryReader.cpp
    src/MathUtil.cpp
    src/Color.cpp
    src/Neighborhood.cpp
    src/Rng.cpp
    src/GaussianSampler.cpp
    src/BinomialSampler.cpp
    src/Rand.cpp
    src/Grids/GridGeometry1D.cpp
    src/Grids/GridGeometry2D.cpp
    src/Grids/GridGeometry3D.cpp
    src/Grids/GridSerialization.cpp
    src/Grids/PDEGridSerialization.cpp
    src/Pde/PDEequations.cpp
    src/Pde/ADIequations.cpp
    src/Pde/IDiffusionBackend2D.cpp
    src/Pde/CpuDiffusionBackend2D.cpp
    src/Pde/IAdiDiffusionBackend2D.cpp
    src/Pde/CpuAdiDiffusionBackend2D.cpp
    src/Agents/IAgentForceBackend2D.cpp
    src/Agents/CpuAgentForceBackend2D.cpp
    src/Tools/OdeSolver.cpp
    src/Tools/MultinomialCalc.cpp
    src/Tools/Fft.cpp
    src/Tools/FftGrid.cpp
    src/Tools/Lenia1Player.cpp
    src/Tools/LeniaNPlayer.cpp
    src/Tools/VarSetManager.cpp
    src/Video/PngExport.cpp
    src/Video/StbImageWriteImpl.cpp
    src/Video/VideoWriter.cpp
    src/Video/JoMpegImpl.cpp
)

target_include_directories(HalCore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Vulkan is required — the headless GPU compute backend (HalGpu) is always
# built.
find_package(Vulkan REQUIRED)

include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 1.3.295
)
FetchContent_MakeAvailable(volk)

# VMA (Vulkan Memory Allocator) for the compute kernels' buffers. Its one
# implementation TU is src/Gpu/Internal/VmaImpl.cpp.
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# Shaders compile GLSL -> SPIR-V at build time into a fixed directory under
# the build tree. HAL_SHADER_DIR bakes its absolute path into HalGpu, so
# shader lookup doesn't depend on the working directory (e.g. under Xcode).
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(HAL_SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Diffusion2D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/DiffusionUpdate2D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/AdiSweep2D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/GrayScott2D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/GrayScott3D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Wave2D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FlowField3D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SlimeMold3DAgents.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SlimeMold3DField.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FluidAdvect3D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FluidDivergence3D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FluidJacobi3D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FluidProject3D.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/AgentCellCount.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/AgentPrefixSum.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/AgentScatter.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/AgentForceAccum.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FullscreenQuad.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FullscreenQuad.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SphereLambert.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SphereLambert.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SphereGlow.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Line3D.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Line3D.frag
)
set(HAL_SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${HAL_SHADER_OUTPUT_DIR})

set(HAL_SHADER_SPV_FILES "")
foreach (shaderSource ${HAL_SHADER_SOURCES})
    # Keep the source extension (FullscreenQuad.vert -> FullscreenQuad.vert.spv)
    # so stages sharing a stem don't collide.
    get_filename_component(shaderName ${shaderSource} NAME)
    set(shaderOutput ${HAL_SHADER_OUTPUT_DIR}/${shaderName}.spv)
    add_custom_command(
        OUTPUT ${shaderOutput}
        COMMAND ${GLSLANG_VALIDATOR} -V ${shaderSource} -o ${shaderOutput}
        DEPENDS ${shaderSource}
        COMMENT "Compiling shader ${shaderName} -> ${shaderName}.spv"
        VERBATIM
    )
    list(APPEND HAL_SHADER_SPV_FILES ${shaderOutput})
endforeach ()

add_custom_target(HalShaders DEPENDS ${HAL_SHADER_SPV_FILES})

add_library(HalGpu
    src/Gpu/Internal/VulkanUtil.cpp
    src/Gpu/Internal/VmaImpl.cpp
    src/Gpu/Math3D.cpp
    src/Gpu/ComputeContext.cpp
    src/Gpu/ShaderPaths.cpp
    src/Gpu/DiffusionKernel2D.cpp
    src/Gpu/AgentForceKernel2D.cpp
    src/Gpu/AdiKernel2D.cpp
    src/Gpu/GrayScottKernel2D.cpp
    src/Gpu/GrayScott3DKernel.cpp
    src/Gpu/WaveKernel2D.cpp
    src/Gpu/FlowFieldKernel3D.cpp
    src/Gpu/SlimeMold3DKernel.cpp
    src/Gpu/FluidKernel3D.cpp
    src/Gpu/GpuDiffusionBackend2D.cpp
    src/Gpu/GpuAgentForceBackend2D.cpp
    src/Gpu/GpuAdiDiffusionBackend2D.cpp
)

add_dependencies(HalGpu HalShaders)

target_include_directories(HalGpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(HalGpu
    PUBLIC
        HalCore
        volk::volk
    PRIVATE
        GPUOpen::VulkanMemoryAllocator
)

# volk loads Vulkan dynamically, so VMA must not link static Vulkan
# functions; it fetches them through the two loader entry points
# ComputeContext hands it. PRIVATE: vk_mem_alloc.h is only included by
# Hal::Gpu .cpp files (public headers forward-declare the VMA handles).
target_compile_definitions(HalGpu
    PRIVATE
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

target_compile_definitions(HalGpu
    PUBLIC
        HAL_SHADER_DIR="${HAL_SHADER_OUTPUT_DIR}"
)

if (HAL_BUILD_WINDOW)
    include(FetchContent)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glfw)

    add_library(HalWindow
        src/Gpu/Window.cpp
        src/Gpu/WindowRenderContext.cpp
        src/Gpu/Scene3DRenderContext.cpp
    )

    target_include_directories(HalWindow
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(HalWindow
        PUBLIC
            HalGpu
            glfw
    )
endif()

if (HAL_BUILD_GUI)
    include(FetchContent)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.9
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(implot)

    # The GUI core uses only the Dear ImGui / ImPlot core (no GLFW/Vulkan
    # backends), so it builds and is tested without a window. The windowed
    # integration lives in HalWindow.
    add_library(HalGui
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
        src/Gui/GuiContext.cpp
    )

    target_include_directories(HalGui
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${imgui_SOURCE_DIR}
            ${implot_SOURCE_DIR}
    )

    target_link_libraries(HalGui
        PUBLIC
            HalCore
    )
endif()

# On-screen GUI: wire ImGui's GLFW + Vulkan backends into HalWindow so the
# GuiContext can be drawn over the window. Needs both the window (GLFW +
# Vulkan) and the GUI (ImGui/ImPlot) subsystems. IMGUI_IMPL_VULKAN_USE_VOLK
# makes imgui_impl_vulkan use volk's dynamically-loaded entry points, the
# same loader the rest of HalGpu/HalWindow uses.
if (HAL_BUILD_WINDOW AND HAL_BUILD_GUI)
    target_sources(HalWindow
        PRIVATE
            ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
            src/Gui/WindowGuiRenderer.cpp
    )

    target_include_directories(HalWindow
        PRIVATE
            ${imgui_SOURCE_DIR}/backends
    )

    target_compile_definitions(HalWindow
        PRIVATE
            IMGUI_IMPL_VULKAN_USE_VOLK
    )

    target_link_libraries(HalWindow
        PUBLIC
            HalGui
    )

    # Manual-verification demo (rendering needs a live display, so it is
    # not part of the automated test suite).
    add_executable(HalGuiDemo examples/GuiDemo.cpp)
    target_link_libraries(HalGuiDemo
        PRIVATE
            HalWindow
            HalGui
    )
endif()

if (HAL_BUILD_EXAMPLES AND HAL_BUILD_WINDOW)
    add_subdirectory(examples)
endif()

# CPU/GPU backend timing comparison. Headless, so it doesn't need the window
# layer.
if (HAL_BUILD_EXAMPLES)
    add_executable(HalBackendBenchmark examples/BackendBenchmark.cpp)
    target_link_libraries(HalBackendBenchmark
        PRIVATE
            HalGpu
    )
endif()

if (HAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
