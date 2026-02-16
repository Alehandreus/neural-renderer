#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <string>
#include <vector>

#include "config_loader.h"
#include "cuda_renderer_neural.h"
#include "image_utils.h"
#include "input_controller.h"
#include "mesh_loader.h"
#include "scene.h"
#include "vec3.h"

// Configuration.
namespace {

const int kWidth = 1920;
const int kHeight = 1080;
const int kBatchSizeGT = 8;
const int kBatchSizeNeural = 8;

const char* kOutputFolder = "comparison_output";
const char* kGroundTruthOutput = "ground_truth.png";
const char* kNeuralOutput = "neural.png";
const char* kFlipOutput = "flip_error.png";

}  // namespace

struct ProgressBar {
    const char* label = "";
    int total = 0;
    int width = 40;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int lastPrinted = -1;

    void begin(const char* newLabel, int totalIters) {
        label = newLabel;
        total = totalIters > 0 ? totalIters : 1;
        start = std::chrono::steady_clock::now();
        lastPrinted = -1;
        update(0);
    }

    static void formatDuration(double seconds, char* out, size_t outSize) {
        if (seconds < 0.0) {
            std::snprintf(out, outSize, "--:--");
            return;
        }
        int sec = static_cast<int>(seconds + 0.5);
        int mins = sec / 60;
        int hrs = mins / 60;
        sec %= 60;
        mins %= 60;
        if (hrs > 0) {
            std::snprintf(out, outSize, "%d:%02d:%02d", hrs, mins, sec);
        } else {
            std::snprintf(out, outSize, "%02d:%02d", mins, sec);
        }
    }

    void update(int current) {
        if (current < 0) current = 0;
        if (current > total) current = total;
        if (current == lastPrinted) return;
        lastPrinted = current;

        double progress = static_cast<double>(current) / static_cast<double>(total);
        int filled = static_cast<int>(progress * width);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double eta = (current > 0) ? (elapsed / current) * (total - current) : -1.0;
        char etaBuf[32];
        char elapsedBuf[32];
        formatDuration(eta, etaBuf, sizeof(etaBuf));
        formatDuration(elapsed, elapsedBuf, sizeof(elapsedBuf));

        std::printf("\r%s [", label);
        for (int i = 0; i < width; ++i) {
            std::printf(i < filled ? "=" : " ");
        }
        std::printf("] %d/%d ETA %s Elapsed %s", current, total, etaBuf, elapsedBuf);
        std::fflush(stdout);
        if (current == total) std::printf("\n");
    }
};

// Compute camera basis from yaw/pitch.
void computeCameraBasis(const CameraState& camera, RenderBasis& basis) {
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    float yawRad = camera.yaw * kDegToRad;
    float pitchRad = camera.pitch * kDegToRad;

    Vec3 forward(
        std::cos(pitchRad) * std::cos(yawRad),
        std::sin(pitchRad),
        std::cos(pitchRad) * std::sin(yawRad));
    forward = normalize(forward);

    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    Vec3 right = cross(forward, worldUp);
    float rLen = length(right);
    if (rLen < 1e-4f) {
        right = Vec3(1.0f, 0.0f, 0.0f);
    } else {
        right = right / rLen;
    }
    Vec3 up = normalize(cross(right, forward));

    basis.forward = forward;
    basis.right = right;
    basis.up = up;
    basis.fovY = camera.fovY;
}

int main(int argc, char** argv) {
    std::printf("=== Comparison Renderer ===\n");

    // Load configuration.
    const char* kDefaultConfigPath = "configs/statuette_obj.json";
    const char* configPath = (argc > 1) ? argv[1] : kDefaultConfigPath;

    RendererConfig config;
    std::string configError;
    if (!LoadConfigFromFile(configPath, &config, &configError)) {
        std::fprintf(stderr, "Failed to load config: %s\n", configError.c_str());
        return 1;
    }
    const int kTotalSamples = config.rendering.total_samples;
    const int kBounceCount = config.rendering.bounce_count;

    // Create output directory.
    std::filesystem::create_directories(kOutputFolder);

    // Extract camera from config.
    CameraState camera{};
    MatrixToCameraState(config.camera.matrix, &camera.position, &camera.yaw, &camera.pitch);
    camera.position = camera.position * config.original_mesh.scale;
    camera.fovY = config.camera.yfov;
    std::printf("Loaded camera: pos=(%.2f, %.2f, %.2f), yaw=%.2f, pitch=%.2f, fovY=%.2f\n",
                camera.position.x, camera.position.y, camera.position.z,
                camera.yaw, camera.pitch, camera.fovY);

    // Load scene.
    Scene scene;
    Mesh& originalMesh = scene.originalMesh();
    Mesh& innerShell = scene.innerShell();
    Mesh& outerShell = scene.outerShell();
    Mesh& additionalMesh = scene.additionalMesh();

    if (!LoadMeshLabeled(config.original_mesh.path.c_str(), &originalMesh, "original",
                         false, true, config.original_mesh.scale)) {
        std::fprintf(stderr, "Failed to load original mesh: %s\n", config.original_mesh.path.c_str());
        return 1;
    }
    std::printf("Loaded original mesh: %d triangles\n", originalMesh.numTriangles());

    if (!LoadMeshLabeled(config.inner_shell.path.c_str(), &innerShell, "inner shell",
                         false, false, config.inner_shell.scale)) {
        std::fprintf(stderr, "Failed to load inner shell: %s\n", config.inner_shell.path.c_str());
        return 1;
    }
    std::printf("Loaded inner shell: %d triangles\n", innerShell.numTriangles());

    if (!LoadMeshLabeled(config.outer_shell.path.c_str(), &outerShell, "outer shell",
                         false, false, config.outer_shell.scale)) {
        std::fprintf(stderr, "Failed to load outer shell: %s\n", config.outer_shell.path.c_str());
        return 1;
    }
    std::printf("Loaded outer shell: %d triangles\n", outerShell.numTriangles());

    if (!config.additional_mesh.path.empty() &&
        LoadMeshLabeled(config.additional_mesh.path.c_str(), &additionalMesh, "additional mesh",
                        false, true, config.additional_mesh.scale)) {
        std::printf("Loaded additional mesh: %d triangles\n", additionalMesh.numTriangles());
    }

    // Apply material config to scene.
    auto applyMaterialConfig = [&](Material& mat) {
        mat.base_color = MaterialParamVec3::constant(config.material.base_color);
        mat.roughness = MaterialParam::constant(config.material.roughness);
        mat.metallic = MaterialParam::constant(config.material.metallic);
        mat.specular = MaterialParam::constant(config.material.specular);
        mat.specular_tint = MaterialParam::constant(config.material.specular_tint);
        mat.anisotropy = MaterialParam::constant(config.material.anisotropy);
        mat.sheen = MaterialParam::constant(config.material.sheen);
        mat.sheen_tint = MaterialParam::constant(config.material.sheen_tint);
        mat.clearcoat = MaterialParam::constant(config.material.clearcoat);
        mat.clearcoat_gloss = MaterialParam::constant(config.material.clearcoat_gloss);
    };
    // Override only non-texture material params (preserve base_color textures).
    auto applyMaterialParamsOnly = [&](Material& mat) {
        mat.roughness = MaterialParam::constant(config.material.roughness);
        mat.metallic = MaterialParam::constant(config.material.metallic);
        mat.specular = MaterialParam::constant(config.material.specular);
        mat.specular_tint = MaterialParam::constant(config.material.specular_tint);
        mat.anisotropy = MaterialParam::constant(config.material.anisotropy);
        mat.sheen = MaterialParam::constant(config.material.sheen);
        mat.sheen_tint = MaterialParam::constant(config.material.sheen_tint);
        mat.clearcoat = MaterialParam::constant(config.material.clearcoat);
        mat.clearcoat_gloss = MaterialParam::constant(config.material.clearcoat_gloss);
    };
    applyMaterialConfig(scene.globalMaterial());
    for (auto& mat : originalMesh.materials_) applyMaterialParamsOnly(mat);
    for (auto& mat : innerShell.materials_) applyMaterialParamsOnly(mat);
    for (auto& mat : outerShell.materials_) applyMaterialParamsOnly(mat);
    for (auto& mat : additionalMesh.materials_) applyMaterialParamsOnly(mat);

    // Load environment.
    std::string envError;
    if (!scene.environment().loadFromFile(config.environment.hdri_path.c_str(), &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", config.environment.hdri_path.c_str(), envError.c_str());
    }
    scene.environment().setRotation(config.environment.rotation);
    scene.environment().setStrength(config.environment.strength);
    std::printf("Loaded environment: %s\n", config.environment.hdri_path.c_str());

    // Create renderer.
    RendererNeural renderer(scene, &config.neural_network);
    renderer.setConstantNeuralColor(config.material.use_constant_neural_color, config.material.constant_neural_color);
    renderer.resize(kWidth, kHeight);
    renderer.setBounceCount(kBounceCount);
    renderer.setLambertView(false);
    renderer.setEnvmapRotation(config.environment.rotation);
    renderer.setUseNeuralQuery(config.neural_network.use_neural_query);

    // Load checkpoint.
    if (!renderer.loadWeightsFromFile(config.checkpoint_path.c_str())) {
        std::fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path.c_str());
        return 1;
    }
    std::printf("Loaded checkpoint: %s\n", config.checkpoint_path.c_str());

    // Compute camera basis.
    RenderBasis basis;
    computeCameraBasis(camera, basis);
    renderer.setCameraBasis(basis);

    // Allocate output buffers.
    size_t pixelCount = static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight);
    std::vector<uchar4> groundTruthPixels(pixelCount);
    std::vector<uchar4> neuralPixels(pixelCount);

    // Render ground truth (classic path tracing).
    {
        std::printf("\n=== Rendering ground truth (%d samples) ===\n", kTotalSamples);
        renderer.setUseNeuralQuery(false);
        renderer.setClassicMeshIndex(0);
        renderer.resetSamples();

        int remainingSamples = kTotalSamples;
        int totalIters = (kTotalSamples + kBatchSizeGT - 1) / kBatchSizeGT;
        int iter = 0;
        ProgressBar bar;
        bar.begin("Ground truth", totalIters);
        while (remainingSamples > 0) {
            int batchSamples = std::min(remainingSamples, kBatchSizeGT);
            renderer.setSamplesPerPixel(batchSamples);
            renderer.render(camera.position);
            remainingSamples -= batchSamples;
            bar.update(++iter);
        }

        cudaMemcpy(groundTruthPixels.data(), renderer.devicePixels(),
                   groundTruthPixels.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);
        savePng((std::string(kOutputFolder) + "/" + kGroundTruthOutput).c_str(),
                groundTruthPixels, kWidth, kHeight);
    }

    // Render neural.
    {
        std::printf("\n=== Rendering neural (%d samples) ===\n", kTotalSamples);
        renderer.setUseNeuralQuery(true);
        renderer.setClassicMeshIndex(0);
        renderer.resetSamples();

        int remainingSamples = kTotalSamples;
        int totalIters = (kTotalSamples + kBatchSizeNeural - 1) / kBatchSizeNeural;
        int iter = 0;
        ProgressBar bar;
        bar.begin("Neural", totalIters);
        while (remainingSamples > 0) {
            int batchSamples = std::min(remainingSamples, kBatchSizeNeural);
            renderer.setSamplesPerPixel(batchSamples);
            renderer.render(camera.position);
            remainingSamples -= batchSamples;
            bar.update(++iter);
        }

        cudaMemcpy(neuralPixels.data(), renderer.devicePixels(),
                   neuralPixels.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);
        savePng((std::string(kOutputFolder) + "/" + kNeuralOutput).c_str(),
                neuralPixels, kWidth, kHeight);
    }

    // Compute metrics.
    float psnr = computePsnr(groundTruthPixels, neuralPixels, kWidth, kHeight);
    std::printf("\n=== Metrics ===\n");
    std::printf("PSNR: %.2f dB\n", psnr);

    std::printf("Computing FLIP error...\n");
    std::string flipPath = std::string(kOutputFolder) + "/" + kFlipOutput;
    float flipError = computeFlip(groundTruthPixels, neuralPixels, kWidth, kHeight, flipPath.c_str());
    std::printf("FLIP: %.4f (mean)\n", flipError);

    std::printf("\nComparison complete.\n");
    return 0;
}
