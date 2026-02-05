#pragma once

#include <string>
#include "vec3.h"

struct MeshConfig {
    std::string path;
    float scale = 1.0f;
    bool use_texture_color = false;
};

struct EnvironmentConfig {
    std::string hdri_path;
    float rotation = 0.0f;
    float strength = 1.0f;
};

struct CameraConfig {
    float matrix[16];  // 4x4 matrix in column-major order (OpenGL/GLM convention)
    float yfov = 1.047198f;  // ~60 degrees in radians
    float move_speed = 0.0f;  // 0 means auto-calculate based on mesh bounds
};

struct RenderingConfig {
    bool normalize_meshes = false;
    bool nearest_texture_sampling = true;
};

struct MaterialConfig {
    Vec3 base_color{1.0f, 1.0f, 1.0f};
    float roughness = 1.0f;
    float metallic = 0.0f;
    float specular = 0.0f;
    float specular_tint = 0.0f;
    float anisotropy = 0.0f;
    float sheen = 0.0f;
    float sheen_tint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoat_gloss = 0.0f;
};

struct NeuralNetworkConfig {
    int log2_hashmap_size = 14;
    bool use_neural_query = false;
};

struct RendererConfig {
    MeshConfig original_mesh;
    MeshConfig inner_shell;
    MeshConfig outer_shell;
    MeshConfig additional_mesh;
    std::string checkpoint_path;
    EnvironmentConfig environment;
    CameraConfig camera;
    RenderingConfig rendering;
    MaterialConfig material;
    NeuralNetworkConfig neural_network;
};

// Load config from JSON file
bool LoadConfigFromFile(const char* configPath, RendererConfig* config, std::string* error);

// Convert camera matrix to position/yaw/pitch representation
void MatrixToCameraState(const float matrix[16], Vec3* position, float* yaw, float* pitch);

// Convert position/yaw/pitch to camera matrix
void CameraStateToMatrix(Vec3 position, float yaw, float pitch, float matrix[16]);
