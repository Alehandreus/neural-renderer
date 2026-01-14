#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "FLIP.h"

#include "cuda_renderer_neural.h"
#include "mesh_loader.h"
#include "renderer.h"
#include "scene.h"
#include "vec3.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;

struct CameraState {
    Vec3 position;
    float yaw;
    float pitch;
    float fovY;
};

CameraState initialCamera() {
    CameraState camera{};
    camera.position = Vec3(2.0f, 1.2f, 2.0f);
    camera.fovY = 60.0f * kDegToRad;

    Vec3 target(0.0f, 0.0f, 0.0f);
    Vec3 forward = normalize(target - camera.position);
    camera.yaw = atan2f(forward.z, forward.x) / kDegToRad;
    camera.pitch = asinf(forward.y) / kDegToRad;
    return camera;
}

RenderBasis buildBasis(const CameraState& camera) {
    float yawRad = camera.yaw * kDegToRad;
    float pitchRad = camera.pitch * kDegToRad;
    Vec3 forward(
            cosf(pitchRad) * cosf(yawRad),
            sinf(pitchRad),
            cosf(pitchRad) * sinf(yawRad));
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

    RenderBasis basis{};
    basis.forward = forward;
    basis.right = right;
    basis.up = up;
    basis.fovY = camera.fovY;
    return basis;
}

float calculatePpd(float dist, float resolutionX, float monitorWidth) {
    return dist * (resolutionX / monitorWidth) * (float(FLIP::PI) / 180.0f);
}

bool fillFlipImage(const std::vector<uchar4>& pixels, FLIP::image<FLIP::color3>* image) {
    if (!image || pixels.empty()) {
        return false;
    }
    FLIP::color3* host = image->getHostData();
    if (!host) {
        return false;
    }
    const float inv = 1.0f / 255.0f;
    for (size_t i = 0; i < pixels.size(); ++i) {
        host[i] = FLIP::color3(
                pixels[i].x * inv,
                pixels[i].y * inv,
                pixels[i].z * inv);
    }
    image->setState(FLIP::CudaTensorState::HOST_ONLY);
    return true;
}

float computeFlip(const std::vector<uchar4>& referencePixels,
                  const std::vector<uchar4>& testPixels,
                  int width,
                  int height,
                  const std::string& heatmapPath) {
    if (referencePixels.size() != testPixels.size() || referencePixels.empty()) {
        return -1.0f;
    }

    FLIP::image<FLIP::color3> reference(width, height);
    FLIP::image<FLIP::color3> test(width, height);
    if (!fillFlipImage(referencePixels, &reference) || !fillFlipImage(testPixels, &test)) {
        return -1.0f;
    }
    reference.synchronizeDevice();
    test.synchronizeDevice();

    FLIP::image<float> errorMap(width, height, 0.0f);
    float ppd = calculatePpd(0.7f, 3840.0f, 0.7f);
    errorMap.FLIP(reference, test, ppd);

    pooling<float> pooledValues;
    for (int y = 0; y < errorMap.getHeight(); ++y) {
        for (int x = 0; x < errorMap.getWidth(); ++x) {
            pooledValues.update(x, y, errorMap.get(x, y));
        }
    }

    if (!heatmapPath.empty()) {
        FLIP::image<FLIP::color3> magmaMap(FLIP::MapMagma, 256);
        FLIP::image<FLIP::color3> ldrFlip(width, height);
        ldrFlip.colorMap(errorMap, magmaMap);
        if (!ldrFlip.pngSave(heatmapPath)) {
            std::fprintf(stderr, "Failed to write %s\n", heatmapPath.c_str());
        }
    }
    return pooledValues.getMean();
}

bool savePng(const std::string& path,
             const std::vector<uchar4>& pixels,
             int width,
             int height) {
    if (width <= 0 || height <= 0 || pixels.size() != static_cast<size_t>(width * height)) {
        return false;
    }
    FLIP::image<FLIP::color3> image(width, height);
    if (!fillFlipImage(pixels, &image)) {
        return false;
    }
    return image.pngSave(path);
}

double computePsnr(const std::vector<uchar4>& a, const std::vector<uchar4>& b) {
    if (a.size() != b.size() || a.empty()) {
        return -1.0;
    }
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        int dr = static_cast<int>(a[i].x) - static_cast<int>(b[i].x);
        int dg = static_cast<int>(a[i].y) - static_cast<int>(b[i].y);
        int db = static_cast<int>(a[i].z) - static_cast<int>(b[i].z);
        mse += static_cast<double>(dr * dr + dg * dg + db * db);
    }
    mse /= static_cast<double>(a.size() * 3);
    if (mse <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    double maxVal = 255.0;
    return 10.0 * std::log10((maxVal * maxVal) / mse);
}

int parseSamples(int argc, char** argv, int index, int defaultSamples, int minValue, int maxValue) {
    if (argc <= index) {
        return defaultSamples;
    }
    char* end = nullptr;
    long value = std::strtol(argv[index], &end, 10);
    if (!end || *end != '\0' || value < minValue || value > maxValue) {
        return defaultSamples;
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    const char* kExactMeshPath = "/home/me/Downloads/chess_orig.fbx";
    // const char* kExactMeshPath = "/home/me/Downloads/chess_outer_10000.fbx";
    const char* kRoughMeshPath = "/home/me/Downloads/chess_outer_10000.fbx";
    // const char* kExactMeshPath = "/home/me/brain/mesh-mapping/models/dragon_outer_3000.fbx";
    // const char* kRoughMeshPath = "/home/me/brain/mesh-mapping/models/dragon_outer_3000.fbx";
    const char* kCheckpointPath = "/home/me/brain/mesh-mapping/checkpoints/outer_params.bin";
    const int kBounceCount = 3;
    const bool kNormalizeMeshes = true;
    const bool kNearestTextureSampling = true;
    const char* kDefaultHdriPath = "/home/me/Downloads/lilienstein_4k.hdr";
    const int kWidth = 1920;
    const int kHeight = 1080;

    const int samplesPerPixel = parseSamples(argc, argv, 1, 2048, 1, 1000000);
    int batchSamples = parseSamples(argc, argv, 2, 1, 1, samplesPerPixel);
    if (samplesPerPixel % batchSamples != 0) {
        std::printf("Batch size %d does not divide %d; using batch size 1.\n",
                    batchSamples,
                    samplesPerPixel);
        batchSamples = 1;
    }
    const int batchCount = samplesPerPixel / batchSamples;
    std::printf("Samples per pixel: %d (batch %d, %d passes)\n",
                samplesPerPixel,
                batchSamples,
                batchCount);

    Scene scene;
    Mesh& exactMesh = scene.exactMesh();
    Mesh& roughMesh = scene.roughMesh();
    if (kExactMeshPath && kExactMeshPath[0] != '\0') {
        std::string loadError;
        std::filesystem::path meshPath(kExactMeshPath);
        std::string ext = meshPath.extension().string();
        for (char& ch : ext) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        bool loaded = false;
        if (ext == ".gltf" || ext == ".glb") {
            loaded = LoadTexturedGltfFromFile(
                    kExactMeshPath,
                    &exactMesh,
                    &loadError,
                    kNormalizeMeshes,
                    kNearestTextureSampling);
        } else {
            loaded = LoadMeshFromFile(kExactMeshPath, &exactMesh, &loadError, kNormalizeMeshes);
        }
        if (!loaded) {
            std::fprintf(stderr, "Failed to load exact mesh '%s': %s\n", kExactMeshPath, loadError.c_str());
        }
    }
    if (exactMesh.triangleCount() == 0) {
        GenerateUvSphere(&exactMesh, 48, 96, 1.0f);
    }
    if (kRoughMeshPath && kRoughMeshPath[0] != '\0') {
        std::string loadError;
        if (!LoadMeshFromFile(kRoughMeshPath, &roughMesh, &loadError, kNormalizeMeshes)) {
            std::fprintf(stderr, "Failed to load rough mesh '%s': %s\n", kRoughMeshPath, loadError.c_str());
        }
    }
    if (roughMesh.triangleCount() == 0) {
        GenerateUvSphere(&roughMesh, 48, 96, 1.0f);
    }

    std::string envError;
    if (!scene.environment().loadFromFile(kDefaultHdriPath, &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", kDefaultHdriPath, envError.c_str());
    }

    RendererNeural renderer(scene);
    renderer.setUseNeuralQuery(false);
    renderer.setBounceCount(kBounceCount);
    renderer.setSamplesPerPixel(batchSamples);
    renderer.setLambertView(false);
    renderer.setLossView(false);
    renderer.setGdSteps(0);

    bool loaded = false;
    if (kCheckpointPath && kCheckpointPath[0] != '\0') {
        loaded = renderer.loadWeightsFromFile(kCheckpointPath);
    }
    if (loaded) {
        std::printf("Neural parameters loaded from file.\n");
    } else {
        std::printf("Neural parameters not loaded (using initialization).\n");
    }

    renderer.resize(kWidth, kHeight);
    std::vector<uchar4> classicPixels(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight));
    std::vector<uchar4> neuralPixels(classicPixels.size());

    CameraState camera = initialCamera();
    RenderBasis basis = buildBasis(camera);
    renderer.setCameraBasis(basis);

    renderer.setUseNeuralQuery(false);
    auto classicStart = std::chrono::steady_clock::now();
    for (int i = 0; i < batchCount; ++i) {
        renderer.render(camera.position, classicPixels);
    }
    auto classicEnd = std::chrono::steady_clock::now();
    double classicSeconds = std::chrono::duration<double>(classicEnd - classicStart).count();
    std::printf("Classic render time: %.3f s\n", classicSeconds);

    renderer.setUseNeuralQuery(true);
    auto neuralStart = std::chrono::steady_clock::now();
    for (int i = 0; i < batchCount; ++i) {
        renderer.render(camera.position, neuralPixels);
    }
    auto neuralEnd = std::chrono::steady_clock::now();
    double neuralSeconds = std::chrono::duration<double>(neuralEnd - neuralStart).count();
    std::printf("Neural render time: %.3f s\n", neuralSeconds);

    const std::string classicOut = "classic.png";
    const std::string neuralOut = "neural.png";
    if (!savePng(classicOut, classicPixels, kWidth, kHeight)) {
        std::fprintf(stderr, "Failed to write %s\n", classicOut.c_str());
    }
    if (!savePng(neuralOut, neuralPixels, kWidth, kHeight)) {
        std::fprintf(stderr, "Failed to write %s\n", neuralOut.c_str());
    }

    double psnr = computePsnr(neuralPixels, classicPixels);
    if (psnr >= 0.0 && std::isfinite(psnr)) {
        std::printf("PSNR (neural vs classic): %.2f dB\n", psnr);
    } else if (psnr >= 0.0) {
        std::printf("PSNR (neural vs classic): inf\n");
    } else {
        std::printf("PSNR (neural vs classic): --\n");
    }

    const std::string flipOut = "flip.png";
    float flip = computeFlip(classicPixels, neuralPixels, kWidth, kHeight, flipOut);
    if (flip >= 0.0f) {
        std::printf("FLIP (neural vs classic): %.6f\n", flip);
    } else {
        std::printf("FLIP (neural vs classic): --\n");
    }

    return 0;
}
