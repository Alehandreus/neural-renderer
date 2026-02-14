cmake_minimum_required(VERSION 3.21)
project(renderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "" FORCE)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/glfw)

# ImGui (header-only, manually integrated below)
set(imgui_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui)

# BVH
set(BVH_BUILD_C_API OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/bvh)

# Tiny-CUDA-NN
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)
add_subdirectory(ext/tiny-cuda-nn)
if (TARGET tiny-cuda-nn)
    set_target_properties(tiny-cuda-nn PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()

# Assimp
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/assimp)

# TinyEXR (header-only)
set(tinyexr_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/tinyexr)

# STB (header-only)
set(stb_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/stb)

# NFD
add_subdirectory(ext/nfd)

# FLIP library
set(FLIP_LIBRARY ON CACHE BOOL "" FORCE)
add_subdirectory(ext/flip-cuda)

# TinyGLTF (header-only)
include_directories(ext/tinygltf)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw)
target_compile_definitions(imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)

find_package(OpenGL REQUIRED)

add_executable(imgui_renderer
    src/bvh_data.h
    src/config_loader.cpp
    src/config_loader.h
    src/cuda_renderer_neural.cu
    src/cuda_renderer_neural.h
    src/input_controller.cpp
    src/input_controller.h
    src/main.cu
    src/mesh.cu
    src/mesh.h
    src/mesh_bvh.cpp
    src/mesh_loader.cpp
    src/mesh_loader.h
    src/material.h
    src/ray.h
    src/renderer.h
    src/scene.cu
    src/scene.h
    src/stb_image.cpp
    src/stb_image_write.cpp
    src/tinyexr.cpp
    src/tinygltf.cpp
    src/triangle.h
    src/vec3.h
)
target_include_directories(imgui_renderer PRIVATE
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${stb_SOURCE_DIR}
)
target_link_libraries(imgui_renderer PRIVATE imgui glfw OpenGL::GL bvh tiny-cuda-nn nfd)
if (TARGET assimp::assimp)
    target_link_libraries(imgui_renderer PRIVATE assimp::assimp)
else()
    target_link_libraries(imgui_renderer PRIVATE assimp)
endif()

# ---------------------------------------------------------------------------
# OptiX hardware ray tracing (optional — set OPTIX_ROOT_DIR to SDK root)
# ---------------------------------------------------------------------------
option(USE_OPTIX "Enable OptiX hardware ray tracing" OFF)

if(USE_OPTIX)
    find_package(CUDAToolkit REQUIRED)

    # Locate OptiX headers
    find_path(OPTIX_INCLUDE_DIR optix.h
        PATHS
            $ENV{OPTIX_ROOT_DIR}/include
            /usr/local/optix/include
            /opt/optix/include
        DOC "OptiX SDK include directory"
    )
    if(NOT OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR
            "OptiX headers not found. "
            "Set OPTIX_ROOT_DIR to the OptiX SDK root (e.g. ~/NVIDIA-OptiX-SDK-8.1.0).")
    endif()
    message(STATUS "OptiX include dir: ${OPTIX_INCLUDE_DIR}")

    # Virtual CUDA architecture for OptiX PTX compilation.
    # We deliberately default to compute_89 (Ada Lovelace) rather than the
    # real device arch (e.g. compute_120/Blackwell): CUDA 12.5+ runs ptxas as
    # a validation step even for -ptx builds, and ptxas rejects OptiX intrinsics
    # (_optix_trace_typed_32 etc.) that are only resolved by the OptiX driver at
    # runtime.  PTX ISA 8.9 is fully supported by OptiX 9 and the OptiX JIT will
    # compile it to the actual GPU's ISA, so no functionality is lost.
    set(OPTIX_PTX_ARCH "compute_89"
        CACHE STRING "Virtual CUDA arch for OptiX PTX compilation (e.g. compute_89)")
    message(STATUS "OptiX PTX arch: ${OPTIX_PTX_ARCH}")

    # -----------------------------------------------------------------------
    # Step 1: compile optix_programs.cu → PTX
    # Use CMake's CUDA_PTX_COMPILATION property so CMake skips ptxas assembly.
    # A raw nvcc -ptx command would still invoke ptxas for validation, which
    # fails for OptiX intrinsics (_optix_trace_typed_32 etc.) that are only
    # resolved by the OptiX driver at runtime.
    # -----------------------------------------------------------------------
    set(_ptx_src ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/optix_programs.cu)

    add_library(optix_programs_ptx OBJECT ${_ptx_src})
    set_property(TARGET optix_programs_ptx PROPERTY CUDA_PTX_COMPILATION ON)
    set_property(TARGET optix_programs_ptx PROPERTY CUDA_STANDARD 17)
    # Disable CMake's automatic -gencode flags (from CMAKE_CUDA_ARCHITECTURES):
    # -ptx disallows multiple architectures.  We set a single virtual arch manually.
    set_property(TARGET optix_programs_ptx PROPERTY CUDA_ARCHITECTURES OFF)
    target_include_directories(optix_programs_ptx PRIVATE
        ${OPTIX_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_options(optix_programs_ptx PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=${OPTIX_PTX_ARCH}>
    )
    # Must define USE_OPTIX so MeshDeviceView gets the OptixTraversableHandle gas
    # field — otherwise sizeof(OptixLaunchParams) differs between host and device.
    target_compile_definitions(optix_programs_ptx PRIVATE USE_OPTIX)

    # -----------------------------------------------------------------------
    # Step 2: embed PTX text as a C string header
    # CMake places OBJECT-library PTX files at a predictable path inside the
    # build tree.  We derive it explicitly so we can list it as a file DEPENDS
    # (generator expressions in DEPENDS don't create build-system rules).
    # -----------------------------------------------------------------------
    set(_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/optix_programs_ptx.h)
    # CMake convention: <binary_dir>/CMakeFiles/<target>.dir/<src_rel>.ptx
    set(_ptx_out
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_programs_ptx.dir/src/rt/optix_programs.ptx)

    add_custom_command(
        OUTPUT  ${_ptx_header}
        COMMAND ${CMAKE_COMMAND}
                -DPTX_FILE=${_ptx_out}
                -DOUTPUT_FILE=${_ptx_header}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake
        DEPENDS ${_ptx_out}
        COMMENT "Embedding OptiX PTX into C string header"
        VERBATIM
    )

    add_custom_target(optix_ptx_header DEPENDS ${_ptx_header})
    # Ensure the OBJECT library is built before the embed step runs
    add_dependencies(optix_ptx_header optix_programs_ptx)

    # -----------------------------------------------------------------------
    # Wire into imgui_renderer
    # -----------------------------------------------------------------------
    target_sources(imgui_renderer PRIVATE
        src/rt/optix_state.cu
        src/rt/optix_state.h
        src/rt/optix_launch_params.h
    )

    add_dependencies(imgui_renderer optix_ptx_header)

    target_include_directories(imgui_renderer PRIVATE
        ${OPTIX_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}   # for optix_programs_ptx.h
    )
    target_compile_definitions(imgui_renderer PRIVATE USE_OPTIX)
    target_link_libraries(imgui_renderer PRIVATE CUDA::cuda_driver)
endif()

# Evaluation tool (ground truth vs neural comparison with PSNR and FLIP)
add_executable(evaluate
    src/bvh_data.h
    src/config_loader.cpp
    src/config_loader.h
    src/cuda_renderer_neural.cu
    src/cuda_renderer_neural.h
    src/main_comparison.cu
    src/mesh.cu
    src/mesh.h
    src/mesh_bvh.cpp
    src/mesh_loader.cpp
    src/mesh_loader.h
    src/material.h
    src/ray.h
    src/renderer.h
    src/scene.cu
    src/scene.h
    src/tinygltf.cpp
    src/triangle.h
    src/vec3.h
)
target_include_directories(evaluate PRIVATE
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${stb_SOURCE_DIR}
    ext/flip-cuda
    ext/flip-cuda/common
)
target_link_libraries(evaluate PRIVATE bvh tiny-cuda-nn flip-cuda)
if (TARGET assimp::assimp)
    target_link_libraries(evaluate PRIVATE assimp::assimp)
else()
    target_link_libraries(evaluate PRIVATE assimp)
endif()

# Image comparison tool (PSNR and FLIP between two existing images)
add_executable(compare_images
    src/main_compare_images.cu
)
target_include_directories(compare_images PRIVATE
    ${stb_SOURCE_DIR}
    ext/flip-cuda
    ext/flip-cuda/common
)
target_link_libraries(compare_images PRIVATE flip-cuda)
