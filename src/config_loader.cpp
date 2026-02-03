#include "config_loader.h"

#include <json/json.hpp>
#include <fstream>
#include <cmath>

using json = nlohmann::json;

bool LoadConfigFromFile(const char* configPath, RendererConfig* config, std::string* error) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        *error = std::string("Failed to open config file: ") + configPath;
        return false;
    }

    json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        *error = std::string("Failed to parse JSON: ") + e.what();
        return false;
    }

    try {
        // Parse scene meshes
        if (j.contains("scene")) {
            auto scene = j["scene"];

            if (scene.contains("original_mesh")) {
                auto& orig = scene["original_mesh"];
                config->original_mesh.path = orig.value("path", "");
                config->original_mesh.scale = orig.value("scale", 1.0f);
            }

            if (scene.contains("inner_shell")) {
                auto& inner = scene["inner_shell"];
                config->inner_shell.path = inner.value("path", "");
                config->inner_shell.scale = inner.value("scale", 1.0f);
            }

            if (scene.contains("outer_shell")) {
                auto& outer = scene["outer_shell"];
                config->outer_shell.path = outer.value("path", "");
                config->outer_shell.scale = outer.value("scale", 1.0f);
            }
        }

        // Parse checkpoint path
        config->checkpoint_path = j.value("checkpoint_path", "");

        // Parse environment
        if (j.contains("environment")) {
            auto& env = j["environment"];
            config->environment.hdri_path = env.value("hdri_path", "");
            config->environment.rotation = env.value("rotation", 0.0f);
            config->environment.strength = env.value("strength", 1.0f);
        }

        // Parse camera
        if (j.contains("camera")) {
            auto& cam = j["camera"];

            if (cam.contains("matrix") && cam["matrix"].is_array() && cam["matrix"].size() == 16) {
                for (size_t i = 0; i < 16; i++) {
                    config->camera.matrix[i] = cam["matrix"][i].get<float>();
                }
            } else {
                *error = "Camera matrix missing or invalid (must be array of 16 floats)";
                return false;
            }

            config->camera.yfov = cam.value("yfov", 1.047198f);
        } else {
            *error = "Camera configuration missing";
            return false;
        }

        // Parse rendering settings
        if (j.contains("rendering")) {
            auto& render = j["rendering"];
            config->rendering.normalize_meshes = render.value("normalize_meshes", false);
            config->rendering.nearest_texture_sampling = render.value("nearest_texture_sampling", true);
        }

    } catch (const std::exception& e) {
        *error = std::string("Error parsing config: ") + e.what();
        return false;
    }

    return true;
}

void MatrixToCameraState(const float matrix[16], Vec3* position, float* yaw, float* pitch) {
    // The matrix from nbvh config is a world-to-camera (view) matrix.
    // To extract camera position, we need to compute inverse(matrix) * (0,0,0,1).
    // For a rigid transformation matrix (rotation + translation), we can use the formula:
    // camera_pos = -R^T * t, where R is the 3x3 rotation part and t is the translation column

    // Extract rotation part (first 3x3)
    // Note: For world-to-camera matrix, R^T gives us the camera orientation in world space

    // Column-major matrix layout:
    // [0 4  8 12]
    // [1 5  9 13]
    // [2 6 10 14]
    // [3 7 11 15]

    // The translation part of world-to-camera matrix
    float tx = matrix[12];
    float ty = matrix[13];
    float tz = matrix[14];

    // Rotation part transposed (which is the camera-to-world rotation)
    float r00 = matrix[0], r01 = matrix[4], r02 = matrix[8];
    float r10 = matrix[1], r11 = matrix[5], r12 = matrix[9];
    float r20 = matrix[2], r21 = matrix[6], r22 = matrix[10];

    // Camera position in world space: -R^T * t
    position->x = -(r00 * tx + r10 * ty + r20 * tz);
    position->y = -(r01 * tx + r11 * ty + r21 * tz);
    position->z = -(r02 * tx + r12 * ty + r22 * tz);

    // Camera forward direction in world space
    // Third row of view matrix is -forward, so negate to get forward
    Vec3 forward(-r20, -r21, -r22);

    // Compute yaw (horizontal angle around Y axis)
    float hLen = std::sqrt(forward.x * forward.x + forward.z * forward.z);
    constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
    *yaw = std::atan2(forward.z, forward.x) * kRadToDeg;
    *pitch = std::atan2(forward.y, hLen) * kRadToDeg;
}

void CameraStateToMatrix(Vec3 position, float yaw, float pitch, float matrix[16]) {
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    float yawRad = yaw * kDegToRad;
    float pitchRad = pitch * kDegToRad;

    // Compute camera forward vector from yaw and pitch (in world space)
    Vec3 forward(
        std::cos(pitchRad) * std::cos(yawRad),
        std::sin(pitchRad),
        std::cos(pitchRad) * std::sin(yawRad)
    );
    float fwdLen = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    if (fwdLen > 1e-6f) {
        forward.x /= fwdLen;
        forward.y /= fwdLen;
        forward.z /= fwdLen;
    }

    // Compute right vector: cross(forward, worldUp)
    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    Vec3 right(
        forward.y * worldUp.z - forward.z * worldUp.y,
        forward.z * worldUp.x - forward.x * worldUp.z,
        forward.x * worldUp.y - forward.y * worldUp.x
    );
    float rLen = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
    if (rLen < 1e-4f) {
        // Forward is parallel to worldUp, use fallback
        right = Vec3(1.0f, 0.0f, 0.0f);
    } else {
        right.x /= rLen;
        right.y /= rLen;
        right.z /= rLen;
    }

    // Compute up vector: cross(right, forward)
    Vec3 up(
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    );
    float upLen = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
    if (upLen > 1e-6f) {
        up.x /= upLen;
        up.y /= upLen;
        up.z /= upLen;
    }

    // Build world-to-camera (view) matrix in column-major order
    // This is the transpose of the camera-to-world rotation, plus transformed translation

    // The rotation part is R^T (transpose of camera orientation)
    // Column 0: right vector becomes row 0
    matrix[0] = right.x;
    matrix[1] = up.x;
    matrix[2] = -forward.x;  // Camera looks down -Z, so negate forward
    matrix[3] = 0.0f;

    // Column 1: up vector becomes row 1
    matrix[4] = right.y;
    matrix[5] = up.y;
    matrix[6] = -forward.y;
    matrix[7] = 0.0f;

    // Column 2: -forward vector becomes row 2
    matrix[8] = right.z;
    matrix[9] = up.z;
    matrix[10] = -forward.z;
    matrix[11] = 0.0f;

    // Column 3: -R^T * position (transformed translation)
    matrix[12] = -(right.x * position.x + right.y * position.y + right.z * position.z);
    matrix[13] = -(up.x * position.x + up.y * position.y + up.z * position.z);
    matrix[14] = -(-forward.x * position.x + -forward.y * position.y + -forward.z * position.z);
    matrix[15] = 1.0f;
}
