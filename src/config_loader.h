#pragma once

#include <string>
#include "vec3.h"

struct MeshConfig {
    std::string path;
    float scale = 1.0f;
};

struct EnvironmentConfig {
    std::string hdri_path;
    float rotation = 0.0f;
    float strength = 1.0f;
};

struct CameraConfig {
    float matrix[16];  // 4x4 matrix in column-major order (OpenGL/GLM convention)
    float yfov = 1.047198f;  // ~60 degrees in radians
};

struct RenderingConfig {
    bool normalize_meshes = false;
    bool nearest_texture_sampling = true;
};

struct RendererConfig {
    MeshConfig original_mesh;
    MeshConfig inner_shell;
    MeshConfig outer_shell;
    std::string checkpoint_path;
    EnvironmentConfig environment;
    CameraConfig camera;
    RenderingConfig rendering;
};

// Load config from JSON file
bool LoadConfigFromFile(const char* configPath, RendererConfig* config, std::string* error);

// Convert camera matrix to position/yaw/pitch representation
void MatrixToCameraState(const float matrix[16], Vec3* position, float* yaw, float* pitch);

// Convert position/yaw/pitch to camera matrix
void CameraStateToMatrix(Vec3 position, float yaw, float pitch, float matrix[16]);
