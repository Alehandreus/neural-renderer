#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "FLIP.h"

#include "config_loader.h"
#include "cuda_renderer_neural.h"
#include "input_controller.h"
#include "mesh.h"
#include "mesh_loader.h"
#include "scene.h"
#include "vec3.h"

#include "stb_image_write.h"

// Configuration.
namespace {

const int kWidth = 1920;
const int kHeight = 1080;
const int kTotalSamples = 2048;
const int kBatchSizeGT = 8;
const int kBatchSizeNeural = 8;
const int kBounceCount = 3;

const char* kOutputFolder = "comparison_output";
const char* kGroundTruthOutput = "ground_truth.png";
const char* kNeuralOutput = "neural.png";
const char* kFlipOutput = "flip_error.png";

}  // namespace

// Create directory if it doesn't exist.
void ensureDirectory(const char* path) {
    std::filesystem::create_directories(path);
}

// Helper to load mesh (same as main.cu).
bool loadMesh(const char* path, Mesh* mesh, const char* label, bool normalize, bool nearestTex, float scale = 1.0f) {
    if (!path || path[0] == '\0') return false;
    std::string loadError;
    // Use LoadMeshAuto which auto-detects format
    bool loaded = LoadMeshAuto(path, mesh, &loadError, normalize, scale);
    if (loaded) {
        mesh->setTextureNearest(nearestTex);
    } else {
        std::fprintf(stderr, "Failed to load %s mesh '%s': %s\n", label, path, loadError.c_str());
    }
    return loaded;
}

// Forward declarations.
bool savePng(const char* path, const std::vector<uchar4>& pixels, int width, int height);

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

// Helper to calculate pixels per degree.
float calculatePPD(float monitorDistance, float resolutionX, float monitorWidth) {
    return monitorDistance * (resolutionX / monitorWidth) * (static_cast<float>(FLIP::PI) / 180.0f);
}

// Compute FLIP error between two images and save visualization.
float computeFlip(const std::vector<uchar4>& ref, const std::vector<uchar4>& test, int width, int height, const char* outputPath) {
    // FLIP parameters (same as nbvh defaults).
    struct {
        float PPD                = 0.0f;     // If PPD==0.0, then it will be computed from the parameters below.
        float monitorDistance    = 0.7f;     // Unit: meters.
        float monitorWidth       = 0.7f;     // Unit: meters.
        float monitorResolutionX = 3840.0f;  // Unit: pixels.
    } flipOptions;

    // Create FLIP images.
    FLIP::image<FLIP::color3> reference(width, height);
    FLIP::image<FLIP::color3> testImage(width, height);
    FLIP::image<float> errorMapFLIP(width, height, 0.0f);

    // Convert uchar4 to FLIP::color3 (normalize to [0,1]).
    // Rendered images are already in sRGB space (encodeSrgb applied in renderer).
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            reference.set(x, y, FLIP::color3(
                ref[idx].x / 255.0f,
                ref[idx].y / 255.0f,
                ref[idx].z / 255.0f));
            testImage.set(x, y, FLIP::color3(
                test[idx].x / 255.0f,
                test[idx].y / 255.0f,
                test[idx].z / 255.0f));
        }
    }

    // Images from renderer are already in sRGB space.
    // DO NOT call LinearRGB2sRGB() - that would apply sRGB curve twice!

    // Calculate PPD.
    flipOptions.PPD = calculatePPD(flipOptions.monitorDistance, flipOptions.monitorResolutionX, flipOptions.monitorWidth);

    // Compute FLIP.
    errorMapFLIP.FLIP(reference, testImage, flipOptions.PPD);

    // Compute mean error and find max.
    pooling<float> pooledValues;
    float maxError = 0.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float err = errorMapFLIP.get(x, y);
            pooledValues.update(x, y, err);
            if (err > maxError) maxError = err;
        }
    }

    // Save FLIP visualization using Magma colormap (black to violet, same as nbvh).
    FLIP::image<FLIP::color3> magmaMap(FLIP::MapMagma, 256);
    FLIP::image<FLIP::color3> ldr_flip(width, height);

    ldr_flip.copyFloat2Color3(errorMapFLIP);
    ldr_flip.colorMap(errorMapFLIP, magmaMap);

    // Convert FLIP::color3 to uchar4 for PNG saving.
    std::vector<uchar4> flipVis(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            FLIP::color3 c = ldr_flip.get(x, y);
            flipVis[idx] = {
                static_cast<unsigned char>(c.x * 255.0f),
                static_cast<unsigned char>(c.y * 255.0f),
                static_cast<unsigned char>(c.z * 255.0f),
                255
            };
        }
    }

    savePng(outputPath, flipVis, width, height);
    std::printf("FLIP max error: %.4f\n", maxError);

    return pooledValues.getMean();
}

// Compute PSNR between two images.
float computePsnr(const std::vector<uchar4>& ref, const std::vector<uchar4>& test, int width, int height) {
    double mse = 0.0;
    size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);

    for (size_t i = 0; i < count; ++i) {
        double dr = static_cast<double>(ref[i].x) - static_cast<double>(test[i].x);
        double dg = static_cast<double>(ref[i].y) - static_cast<double>(test[i].y);
        double db = static_cast<double>(ref[i].z) - static_cast<double>(test[i].z);
        mse += (dr * dr + dg * dg + db * db) / 3.0;
    }

    mse /= static_cast<double>(count);

    if (mse < 1e-10) {
        return 100.0f;  // Images are identical.
    }

    double psnr = 10.0 * std::log10((255.0 * 255.0) / mse);
    return static_cast<float>(psnr);
}

// Save image to PNG.
bool savePng(const char* path, const std::vector<uchar4>& pixels, int width, int height) {
    // Convert uchar4 to RGB.
    std::vector<unsigned char> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
        rgb[i * 3 + 0] = pixels[i].x;
        rgb[i * 3 + 1] = pixels[i].y;
        rgb[i * 3 + 2] = pixels[i].z;
    }

    if (stbi_write_png(path, width, height, 3, rgb.data(), width * 3) == 0) {
        std::fprintf(stderr, "Failed to write PNG: %s\n", path);
        return false;
    }

    std::printf("Saved: %s\n", path);
    return true;
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

    // Create output directory.
    ensureDirectory(kOutputFolder);

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

    if (!loadMesh(config.original_mesh.path.c_str(), &originalMesh, "original",
                  config.rendering.normalize_meshes, config.rendering.nearest_texture_sampling,
                  config.original_mesh.scale)) {
        std::fprintf(stderr, "Failed to load original mesh: %s\n", config.original_mesh.path.c_str());
        return 1;
    }
    std::printf("Loaded original mesh: %d triangles\n", originalMesh.numTriangles());

    if (!loadMesh(config.inner_shell.path.c_str(), &innerShell, "inner shell",
                  config.rendering.normalize_meshes, false, config.inner_shell.scale)) {
        std::fprintf(stderr, "Failed to load inner shell: %s\n", config.inner_shell.path.c_str());
        return 1;
    }
    std::printf("Loaded inner shell: %d triangles\n", innerShell.numTriangles());

    if (!loadMesh(config.outer_shell.path.c_str(), &outerShell, "outer shell",
                  config.rendering.normalize_meshes, false, config.outer_shell.scale)) {
        std::fprintf(stderr, "Failed to load outer shell: %s\n", config.outer_shell.path.c_str());
        return 1;
    }
    std::printf("Loaded outer shell: %d triangles\n", outerShell.numTriangles());

    if (!config.additional_mesh.path.empty() &&
        loadMesh(config.additional_mesh.path.c_str(), &additionalMesh, "additional mesh",
                 config.rendering.normalize_meshes, config.rendering.nearest_texture_sampling,
                 config.additional_mesh.scale)) {
        std::printf("Loaded additional mesh: %d triangles\n", additionalMesh.numTriangles());
    }

    // Apply material config to scene (use global material for non-GLTF meshes)
    scene.globalMaterial().base_color = MaterialParamVec3::constant(config.material.base_color);
    scene.globalMaterial().roughness = MaterialParam::constant(config.material.roughness);
    scene.globalMaterial().metallic = MaterialParam::constant(config.material.metallic);
    scene.globalMaterial().specular = MaterialParam::constant(config.material.specular);
    scene.globalMaterial().specular_tint = MaterialParam::constant(config.material.specular_tint);
    scene.globalMaterial().anisotropy = MaterialParam::constant(config.material.anisotropy);
    scene.globalMaterial().sheen = MaterialParam::constant(config.material.sheen);
    scene.globalMaterial().sheen_tint = MaterialParam::constant(config.material.sheen_tint);
    scene.globalMaterial().clearcoat = MaterialParam::constant(config.material.clearcoat);
    scene.globalMaterial().clearcoat_gloss = MaterialParam::constant(config.material.clearcoat_gloss);

    // Load environment.
    std::string envError;
    if (!scene.environment().loadFromFile(config.environment.hdri_path.c_str(), &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", config.environment.hdri_path.c_str(), envError.c_str());
        return 1;
    }
    scene.environment().setRotation(config.environment.rotation);
    scene.environment().setStrength(config.environment.strength);
    std::printf("Loaded environment: %s\n", config.environment.hdri_path.c_str());

    // Create renderer.
    RendererNeural renderer(scene, &config.neural_network);
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
        renderer.setClassicMeshIndex(0);  // Original mesh.
        renderer.resetSamples();  // Reset accumulation to ensure clean state.

        int remainingSamples = kTotalSamples;
        while (remainingSamples > 0) {
            int batchSamples = std::min(remainingSamples, kBatchSizeGT);
            renderer.setSamplesPerPixel(batchSamples);

            renderer.render(camera.position, groundTruthPixels);

            remainingSamples -= batchSamples;
            std::printf("  Progress: %d / %d samples\n", kTotalSamples - remainingSamples, kTotalSamples);
        }

        // The renderer has accumulated all samples internally.
        // groundTruthPixels now contains the final result.
        std::string gtPath = std::string(kOutputFolder) + "/" + kGroundTruthOutput;
        savePng(gtPath.c_str(), groundTruthPixels, kWidth, kHeight);
    }

    // Render neural.
    {
        std::printf("\n=== Rendering neural (%d samples) ===\n", kTotalSamples);

        renderer.setUseNeuralQuery(true);
        renderer.setClassicMeshIndex(0);  // Original mesh (same as ground truth).
        renderer.resetSamples();  // Reset accumulation to ensure clean state.

        int remainingSamples = kTotalSamples;
        while (remainingSamples > 0) {
            int batchSamples = std::min(remainingSamples, kBatchSizeNeural);
            renderer.setSamplesPerPixel(batchSamples);

            renderer.render(camera.position, neuralPixels);

            remainingSamples -= batchSamples;
            std::printf("  Progress: %d / %d samples\n", kTotalSamples - remainingSamples, kTotalSamples);
        }

        // The renderer has accumulated all samples internally.
        // neuralPixels now contains the final result.
        std::string neuralPath = std::string(kOutputFolder) + "/" + kNeuralOutput;
        savePng(neuralPath.c_str(), neuralPixels, kWidth, kHeight);
    }

    // Compute PSNR.
    float psnr = computePsnr(groundTruthPixels, neuralPixels, kWidth, kHeight);
    std::printf("\n=== Metrics ===\n");
    std::printf("PSNR: %.2f dB\n", psnr);

    // Compute FLIP.
    std::printf("Computing FLIP error...\n");
    std::string flipPath = std::string(kOutputFolder) + "/" + kFlipOutput;
    float flipError = computeFlip(groundTruthPixels, neuralPixels, kWidth, kHeight, flipPath.c_str());
    std::printf("FLIP: %.4f (mean)\n", flipError);

    std::printf("\nComparison complete.\n");
    return 0;
}
