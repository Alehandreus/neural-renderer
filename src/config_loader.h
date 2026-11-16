#pragma once

#include <string>
#include "render_params.h"
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
    float move_speed = 0.0f;  // 0 means auto-calculate based on mesh bounds
};

struct RenderingConfig {
    // Deterministic stratified sub-pixel sampling instead of random jitter,
    // matching LiteRT's RAY_GEN_MODE_REGULAR: a floor(sqrt(total_samples))^2 grid
    // per pixel (set by evaluate via setStrataPerAxis).  Use the same spp as the
    // LiteRT reference for identical sample positions.
    bool center_rays = false;
    // Trace shells / meshes with OptiX hardware RT when the build supports it
    // (USE_OPTIX); false falls back to the software BVH.
    bool use_hardware_rt = true;
    int total_samples = 2048;
    int bounce_count = 3;
    int width = 1920;
    int height = 1080;

    // Where `evaluate` writes the neural render.  Empty keeps the historical
    // behaviour of writing comparison_output/neural.png relative to the CWD;
    // set it so several renders (e.g. one per benchmark camera) can run from a
    // single working directory without overwriting each other.
    std::string output_path;

    // The ground-truth path-traced pass is only needed for evaluate's own
    // PSNR/FLIP report.  Driven from the LiteRT benchmark it is dead weight -
    // the comparison there is against LiteRT's own MESH reference - and at high
    // sample counts it is the more expensive of the two passes.
    bool render_ground_truth = true;
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
    bool use_constant_neural_color = false;
    Vec3 constant_neural_color{0.8f, 0.8f, 0.8f};
};

// Shading settings.  With `lambert = false` (the default) the renderer path
// traces with the Disney BRDF and the environment map, as in the paper.  Setting
// `lambert = true` together with `litert_mode` reproduces LiteRT's
// MULTI_RENDER_MODE_LAMBERT_NO_TEX so that PSNR/FLIP against the other
// representations in the LiteRT benchmark is meaningful.
struct ShadingConfig {
    bool lambert = false;
    bool litert_mode = true;   // only consulted when lambert is on
    Vec3 light_dir{1.0f, 1.0f, 1.0f};
    float dir_intensity = 2.0f / 3.0f;
    float ambient = 0.25f;
    Vec3 background{1.0f, 1.0f, 1.0f};
    bool white_base_color = true;
    bool encode_srgb = false;
};

struct NeuralNetworkConfig {
    int log2_hashmap_size = 14;
    int base_resolution = 16;
    bool use_neural_query = false;
    int n_neurons = 128;
    int n_hidden_layers = 4;
    int n_levels = 8;
    std::string mlp_otype = "FullyFusedMLP";
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
    ShadingConfig shading;
    NeuralNetworkConfig neural_network;
};

// Translate the JSON-facing shading config into the GPU-facing settings struct.
inline LambertSettings ToLambertSettings(const ShadingConfig& shading) {
    LambertSettings s;
    s.litertMode = shading.litert_mode;
    s.lightDir = shading.light_dir;
    s.dirIntensity = shading.dir_intensity;
    s.ambient = shading.ambient;
    s.background = shading.background;
    s.whiteBaseColor = shading.white_base_color;
    s.encodeSrgb = shading.encode_srgb;
    return s;
}

// Load config from JSON file
bool LoadConfigFromFile(const char* configPath, RendererConfig* config, std::string* error);

// Convert camera matrix to position/yaw/pitch representation
void MatrixToCameraState(const float matrix[16], Vec3* position, float* yaw, float* pitch);

// Convert position/yaw/pitch to camera matrix
void CameraStateToMatrix(Vec3 position, float yaw, float pitch, float matrix[16]);
