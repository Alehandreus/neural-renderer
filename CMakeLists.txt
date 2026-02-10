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

set(TINYGLTF_HEADER_ONLY OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/tinygltf)

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
    src/triangle.h
    src/vec3.h
)
target_include_directories(imgui_renderer PRIVATE
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${stb_SOURCE_DIR}
)
target_link_libraries(imgui_renderer PRIVATE imgui glfw OpenGL::GL bvh tiny-cuda-nn nfd)
if (TARGET tinygltf)
    target_link_libraries(imgui_renderer PRIVATE tinygltf)
endif()
if (TARGET assimp::assimp)
    target_link_libraries(imgui_renderer PRIVATE assimp::assimp)
else()
    target_link_libraries(imgui_renderer PRIVATE assimp)
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
    # src/stb_image.cpp
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
if (TARGET tinygltf)
    target_link_libraries(evaluate PRIVATE tinygltf)
endif()
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
