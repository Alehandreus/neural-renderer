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

// Camera basis straight from the column-major view matrix (rows of its rotation
// part are right, up and -forward).  Going through yaw/pitch would lose roll.
RenderBasis cameraBasisFromMatrix(const float m[16], float fovY) {
    RenderBasis basis;
    basis.right = normalize(Vec3(m[0], m[4], m[8]));
    basis.up = normalize(Vec3(m[1], m[5], m[9]));
    basis.forward = normalize(Vec3(-m[2], -m[6], -m[10]));
    basis.fovY = fovY;
    return basis;
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
    const int kWidth = config.rendering.width;
    const int kHeight = config.rendering.height;
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
    renderer.setLambertView(config.shading.lambert);
    renderer.setLambertSettings(ToLambertSettings(config.shading));
    renderer.setCenterRays(config.rendering.center_rays);
#ifdef USE_OPTIX
    renderer.setUseHardwareRT(config.rendering.use_hardware_rt);
    std::printf("Hardware RT (OptiX): %s\n", config.rendering.use_hardware_rt ? "on" : "off");
#endif
    renderer.setStrataPerAxis(static_cast<int>(std::sqrt(static_cast<double>(config.rendering.total_samples))));
    if (config.shading.lambert) {
        std::printf("Shading: lambert%s\n",
                    config.shading.litert_mode ? " (LiteRT-compatible)" : " (headlight)");
    }
    renderer.setEnvmapRotation(config.environment.rotation);
    renderer.setUseNeuralQuery(config.neural_network.use_neural_query);

    // Load checkpoint.
    if (!renderer.loadWeightsFromFile(config.checkpoint_path.c_str())) {
        std::fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path.c_str());
        return 1;
    }
    std::printf("Loaded checkpoint: %s\n", config.checkpoint_path.c_str());

    // Compute camera basis.
    RenderBasis basis = cameraBasisFromMatrix(config.camera.matrix, camera.fovY);
    renderer.setCameraBasis(basis);

    // Allocate output buffers.
    size_t pixelCount = static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight);
    std::vector<uchar4> groundTruthPixels(pixelCount);
    std::vector<uchar4> neuralPixels(pixelCount);
    double neuralTimeMs = 0.0;
#ifdef PROFILE_KERNELS
    KernelTimings neuralTimings;  // summed over every render() call of the neural pass
#endif

    // Resolve where the neural render goes.  Without an explicit output_path we
    // keep writing comparison_output/neural.png next to the CWD, as before.
    std::string neuralPath = config.rendering.output_path;
    if (neuralPath.empty()) {
        neuralPath = std::string(kOutputFolder) + "/" + kNeuralOutput;
    } else {
        std::filesystem::path parent = std::filesystem::path(neuralPath).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    }

    // Render ground truth (classic path tracing).
    if (config.rendering.render_ground_truth) {
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
        // Untimed warm-up: the first render() lazily builds BVHs and uploads every
        // mesh, allocates network buffers and initializes tcnn.
        renderer.setSamplesPerPixel(std::min(kTotalSamples, kBatchSizeNeural));
        renderer.render(camera.position);
        cudaDeviceSynchronize();
        renderer.resetSamples();

        int remainingSamples = kTotalSamples;
        int totalIters = (kTotalSamples + kBatchSizeNeural - 1) / kBatchSizeNeural;
        int iter = 0;
        ProgressBar bar;
        bar.begin("Neural", totalIters);
        cudaDeviceSynchronize();
        auto neuralStart = std::chrono::steady_clock::now();
        while (remainingSamples > 0) {
            int batchSamples = std::min(remainingSamples, kBatchSizeNeural);
            renderer.setSamplesPerPixel(batchSamples);
            renderer.render(camera.position);
#ifdef PROFILE_KERNELS
            neuralTimings += renderer.lastFrameTimings();
#endif
            remainingSamples -= batchSamples;
            bar.update(++iter);
        }
        cudaDeviceSynchronize();
        neuralTimeMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - neuralStart).count();

        cudaMemcpy(neuralPixels.data(), renderer.devicePixels(),
                   neuralPixels.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);
        savePng(neuralPath.c_str(), neuralPixels, kWidth, kHeight);
    }

    // Compute metrics.  These compare our own neural render against our own path
    // traced ground truth; the LiteRT benchmark ignores them and scores the
    // saved PNG against its MESH reference instead.
    if (config.rendering.render_ground_truth) {
        float psnr = computePsnr(groundTruthPixels, neuralPixels, kWidth, kHeight);
        std::printf("\n=== Metrics ===\n");
        std::printf("PSNR: %.2f dB\n", psnr);

        std::printf("Computing FLIP error...\n");
        std::string flipPath = std::string(kOutputFolder) + "/" + kFlipOutput;
        float flipError = computeFlip(groundTruthPixels, neuralPixels, kWidth, kHeight, flipPath.c_str());
        std::printf("FLIP: %.4f (mean)\n", flipError);
    }

    // Machine-readable summary for the LiteRT benchmark, which parses this as a
    // blk block (same contract as nglod's sdf_renderer.py).  Size = network weights
    // file + the two shells.  Shells are counted as what tracing them needs:
    //   - OptiX on:  their compacted GAS bytes (the GAS contains the geometry);
    //   - OptiX off: 1.5 x (welded positions at 16 B + indices at 12 B per triangle),
    //                i.e. mesh data plus a BVH budget of half of it.
    // Time covers the neural pass only.
    {
        double sizeMb = 0.0;
        std::error_code ec;
        auto bytes = std::filesystem::file_size(config.checkpoint_path, ec);
        if (!ec) sizeMb = double(bytes) / (1024.0 * 1024.0);

        bool gasUsed = false;
#ifdef USE_OPTIX
        gasUsed = config.rendering.use_hardware_rt &&
                  innerShell.gasBufferSize() > 0 && outerShell.gasBufferSize() > 0;
#endif
        auto shellBytes = [&](const Mesh& shell) -> double {
#ifdef USE_OPTIX
            if (gasUsed) return double(shell.gasBufferSize());
#endif
            return 1.5 * (double(shell.weldedVertexCount()) * 16.0 + double(shell.numTriangles()) * 12.0);
        };
        const double innerB = shellBytes(innerShell), outerB = shellBytes(outerShell);
        sizeMb += (innerB + outerB) / (1024.0 * 1024.0);
        std::fprintf(stderr, "[DBRT size] weights=%ju B inner_shell=%.0f B outer_shell=%.0f B (%s) inner_tris=%d outer_tris=%d inner_welded_verts=%zu outer_welded_verts=%zu\n",
                     (uintmax_t)(ec ? 0 : bytes), innerB, outerB,
                     gasUsed ? "compacted OptiX GAS" : "1.5 x (welded positions 16 B + indices 12 B)",
                     innerShell.numTriangles(), outerShell.numTriangles(),
                     innerShell.weldedVertexCount(), outerShell.weldedVertexCount());

        std::fflush(stdout);
        std::fprintf(stderr, "{ size:r = %.6f time:r = %.4f }\n", sizeMb, neuralTimeMs);
#ifdef PROFILE_KERNELS
        // Per-bucket GPU time (ms) summed over the whole neural pass.
        // frame_ms spans each render() call, so frame_ms - sum(buckets) is the
        // time spent in untimed kernels, memsets and host syncs.
        std::fprintf(stderr, "{ prof_frame_ms:r = %.4f", neuralTimings.frameMs);
        for (int i = 0; i < KID_COUNT; ++i) {
            std::fprintf(stderr, " prof_%s_ms:r = %.4f", KernelTimings::key(i), neuralTimings.ms[i]);
        }
        std::fprintf(stderr, " prof_rays:r = %llu prof_neural_ray_calls:r = %llu spp:i = %d }\n",
                     static_cast<unsigned long long>(neuralTimings.rayCount),
                     static_cast<unsigned long long>(neuralTimings.neuralRayCalls), kTotalSamples);
#endif
        std::fflush(stderr);
    }

    std::printf("\nComparison complete.\n");
    return 0;
}
