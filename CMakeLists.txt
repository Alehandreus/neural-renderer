cmake_minimum_required(VERSION 3.21)
project(renderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "" FORCE)
endif()

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.4
)
FetchContent_GetProperties(imgui)
if (NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

set(BVH_BUILD_C_API OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    bvh
    GIT_REPOSITORY https://github.com/madmann91/bvh.git
    GIT_TAG master
)
FetchContent_MakeAvailable(bvh)

set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    tiny-cuda-nn
    GIT_REPOSITORY https://github.com/julcst/tiny-cuda-nn.git
    GIT_TAG fix-cutlass
)
FetchContent_MakeAvailable(tiny-cuda-nn)
if (TARGET tiny-cuda-nn)
    set_target_properties(tiny-cuda-nn PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()

set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG v5.3.1
)
FetchContent_MakeAvailable(assimp)

FetchContent_Declare(
    tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG v1.0.7
)
FetchContent_GetProperties(tinyexr)
if (NOT tinyexr_POPULATED)
    FetchContent_Populate(tinyexr)
endif()

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_GetProperties(stb)
if (NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

FetchContent_Declare(
    nfd
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG v1.1.1
)
FetchContent_MakeAvailable(nfd)

# FLIP library
set(FLIP_LIBRARY ON CACHE BOOL "" FORCE)
add_subdirectory(ext/flip-cuda)

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
    src/stb_image.cpp
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

