#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "renderer.h"

struct MeshDeviceView;
struct RenderParams;

namespace tcnn {
namespace cpp {
class Module;
}  // namespace cpp
}  // namespace tcnn

class Scene;
struct NeuralNetworkConfig;

class RendererNeural final {
 public:
    explicit RendererNeural(Scene& scene, const NeuralNetworkConfig* nnConfig = nullptr);
    ~RendererNeural();

    void resize(int width, int height);
    void setCameraBasis(const RenderBasis& basis);
    void render(const Vec3& camPos, std::vector<uchar4>& hostPixels);

    void setUseNeuralQuery(bool enabled) { useNeuralQuery_ = enabled; }
    bool useNeuralQuery() const { return useNeuralQuery_; }
    bool loadWeightsFromFile(const std::string& path);

    void setSamplesPerPixel(int samples) { samplesPerPixel_ = samples; }
    void setBounceCount(int count) { bounceCount_ = count; }
    void setLambertView(bool enabled) { lambertView_ = enabled; }
    void setClassicMeshIndex(int index) { classicMeshIndex_ = index; }
    int classicMeshIndex() const { return classicMeshIndex_; }
    void setEnvmapRotation(float degrees) { envmapRotation_ = degrees; }
    float envmapRotation() const { return envmapRotation_; }
    void resetSamples() { resetAccum(); }

    int samplesPerPixel() const { return samplesPerPixel_; }
    int bounceCount() const { return bounceCount_; }
    size_t paramsBytes() const { return pointEncParamsBytes_ + dirEncParamsBytes_ + mlpParamsBytes_; }
    int width() const { return width_; }
    int height() const { return height_; }

 private:
    void release();
    void releaseNetwork();
    bool ensureNetworkBuffers(size_t elementCount);
    bool ensureAccumBuffer(size_t pixelCount);
    void resetAccum();
    void traceNeuralSegmentsForRays(bool useCameraRays,
                                    const float* rayOrigins,
                                    const float* rayDirections,
                                    const int* rayActiveMask,
                                    const float* rayPdfs,
                                    size_t elementCount,
                                    const RenderParams& params,
                                    const MeshDeviceView& outerView,
                                    const MeshDeviceView& innerView,
                                    Vec3 outerMin,
                                    Vec3 outerInvExtent,
                                    float* outHitPositions,
                                    float* outHitNormals,
                                    float* outHitColors,
                                    int* outHitFlags);

    Scene* scene_ = nullptr;

    // Three separate tcnn modules matching Python RayModel.
    tcnn::cpp::Module* pointEncoding_ = nullptr;
    tcnn::cpp::Module* dirEncoding_ = nullptr;
    tcnn::cpp::Module* mlpNetwork_ = nullptr;

    // Per-module device parameters (FP16).
    void* pointEncParams_ = nullptr;
    void* dirEncParams_ = nullptr;
    void* mlpParams_ = nullptr;

    // Compacted encoding inputs (3 floats each, per compacted hit).
    float* compactedOuterPos_ = nullptr;
    float* compactedInnerPos_ = nullptr;
    float* compactedDirs_ = nullptr;

    // Encoding outputs (FP16, per compacted hit).
    void* pointEncOutput1_ = nullptr;
    void* pointEncOutput2_ = nullptr;
    void* dirEncOutput_ = nullptr;

    // Concatenated MLP input (FP32).
    float* mlpInput_ = nullptr;

    // MLP output (FP16).
    void* outputs_ = nullptr;

    // Compaction buffers.
    int* hitIndices_ = nullptr;
    int* hitCount_ = nullptr;

    // Shell tracing buffers.
    float* outerHitPositions_ = nullptr;
    float* innerHitPositions_ = nullptr;
    float* rayDirections_ = nullptr;
    int* outerHitFlags_ = nullptr;

    // Primary hit buffers.
    float* hitPositions_ = nullptr;
    float* hitNormals_ = nullptr;
    float* hitColors_ = nullptr;
    float* hitMaterialParams_ = nullptr;
    int* hitFlags_ = nullptr;

    // Additional mesh hit buffers (for hybrid rendering).
    float* additionalHitPositions_ = nullptr;
    float* additionalHitNormals_ = nullptr;
    float* additionalHitColors_ = nullptr;
    int* additionalHitFlags_ = nullptr;

    // Bounce buffers (ping-pong).
    float* bouncePositions_ = nullptr;
    float* bounceNormals_ = nullptr;
    float* bounceDirs_ = nullptr;
    float* bounceColors_ = nullptr;
    float* bounceMaterialParams_ = nullptr;
    int* bounceHitFlags_ = nullptr;

    float* bounce2Positions_ = nullptr;
    float* bounce2Normals_ = nullptr;
    float* bounce2Dirs_ = nullptr;
    float* bounce2Colors_ = nullptr;
    int* bounce2HitFlags_ = nullptr;

    // Path tracing state.
    Vec3* pathThroughput_ = nullptr;
    Vec3* pathRadiance_ = nullptr;
    int* pathActive_ = nullptr;

    // Wavefront path tracing buffers.
    float* bounceOrigins_ = nullptr;      // 3 floats per sample (ray origins)
    float* bounceDirections_ = nullptr;   // 3 floats per sample (ray directions)
    float* bouncePdfs_ = nullptr;         // 1 float per sample (BRDF PDF)
    float* bounceBRDFs_ = nullptr;        // 3 floats per sample (BRDF weight: f*cos/pdf)

    // Multi-segment iteration state buffers.
    int* rayActiveFlags_ = nullptr;      // 1 if ray still needs processing
    float* accumT_ = nullptr;            // Accumulated distance along ray
    float* currentEntryPos_ = nullptr;   // Current segment entry position (3 floats per ray)
    float* outerExitT_ = nullptr;        // Distance to outer shell exit
    float* innerEnterT_ = nullptr;       // Distance to inner shell enter
    int* innerHitFlags_ = nullptr;       // Whether inner shell was hit in segment
    float* segmentExitPos_ = nullptr;    // Segment exit position (3 floats per ray)

    size_t bufferElements_ = 0;
    size_t accumPixels_ = 0;

    // Per-module param sizes.
    size_t pointEncParamsBytes_ = 0;
    size_t dirEncParamsBytes_ = 0;
    size_t mlpParamsBytes_ = 0;

    // Module output dimensions.
    uint32_t pointEncOutDims_ = 0;
    uint32_t dirEncOutDims_ = 0;
    uint32_t mlpInputDims_ = 0;
    uint32_t mlpOutputDims_ = 0;
    size_t mlpOutputElemSize_ = 0;

    RenderBasis basis_{};
    Vec3 lightDir_{};
    int width_ = 0;
    int height_ = 0;
    uchar4* devicePixels_ = nullptr;
    Vec3* accum_ = nullptr;

    bool lambertView_ = false;
    bool useNeuralQuery_ = false;
    int samplesPerPixel_ = 1;
    int bounceCount_ = 0;
    int classicMeshIndex_ = 0;
    float envmapRotation_ = 0.0f;
    uint32_t accumSampleCount_ = 0;

    bool lastUseNeuralQuery_ = true;
    bool lastLambertView_ = false;
    int lastBounceCount_ = -1;
    int lastSamplesPerPixel_ = -1;
    int lastClassicMeshIndex_ = -1;
    float lastEnvmapRotation_ = 0.0f;
    bool hasLastCamera_ = false;
    Vec3 lastCamPos_{};
    RenderBasis lastBasis_{};
    float lastFovY_ = 0.0f;
    float sceneScale_ = 1.0f;
};
