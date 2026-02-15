#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "renderer.h"

struct MeshDeviceView;
struct RenderParams;

#ifdef USE_OPTIX
struct OptixState;
struct OptixLaunchParams;
#endif

namespace tcnn {
template <typename T> class NetworkWithInputEncoding;
}  // namespace tcnn

class Scene;
struct NeuralNetworkConfig;

class RendererNeural final {
 public:
    explicit RendererNeural(Scene& scene, const NeuralNetworkConfig* nnConfig = nullptr);
    ~RendererNeural();

    void resize(int width, int height);
    void setCameraBasis(const RenderBasis& basis);
    void render(const Vec3& camPos);
    uchar4* devicePixels() const { return devicePixels_; }

    void setUseNeuralQuery(bool enabled) { useNeuralQuery_ = enabled; }
    bool useNeuralQuery() const { return useNeuralQuery_; }

    void setUseHardwareRT(bool v) { if (v != useHardwareRT_) { useHardwareRT_ = v; resetAccum(); } }
    bool useHardwareRT() const { return useHardwareRT_; }
    bool loadWeightsFromFile(const std::string& path);
    // When true, the checkpoint has [hg_params | mlp_params] order instead of [mlp | hg].
    void setSwapParamOrder(bool v) { swapParamOrder_ = v; }
    bool swapParamOrder() const { return swapParamOrder_; }

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
    size_t paramsBytes() const { return networkParamsBytes_; }
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
                                    float* outHitMaterialParams,
                                    int* outHitFlags,
                                    float* outHitDistances);

    Scene* scene_ = nullptr;

    // Fused encoding + MLP (3 HashGrids + SH + FullyFusedMLP), all in FP16.
    std::shared_ptr<tcnn::NetworkWithInputEncoding<__half>> network_;

    // Device parameters for network_ (FP16, single contiguous block).
    // Layout as per NetworkWithInputEncoding::set_params_impl: [mlp | enc0 | enc1 | enc2].
    void* networkParams_ = nullptr;
    size_t networkParamsBytes_ = 0;

    // Flat raw-coordinate input buffer: [entry.xyz | exit.xyz | [mid.xyz] | dir.xyz] per element.
    float* networkInputs_ = nullptr;

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
    float* hitDistances_ = nullptr;  // Neural predicted distances

    // Additional mesh hit buffers (for hybrid rendering).
    float* additionalHitPositions_ = nullptr;
    float* additionalHitNormals_ = nullptr;
    float* additionalHitColors_ = nullptr;
    float* additionalHitMaterialParams_ = nullptr;
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
    float* bounceDistances_ = nullptr;    // 1 float per sample (neural predicted distance from previous hit)

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

    // Network dimensions.
    uint32_t inputDims_ = 0;       // raw coord input dims: pointCount*3 + 3
    uint32_t pointCount_ = 0;      // number of spatial points encoded (2 or 3)
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
    bool useMidpointEncoding_ = false;
    bool swapParamOrder_ = true;  // checkpoint has [hg | mlp] instead of [mlp | hg]
    int samplesPerPixel_ = 1;
    int bounceCount_ = 0;
    int classicMeshIndex_ = 0;
    float envmapRotation_ = 0.0f;
    uint32_t accumSampleCount_ = 0;

#ifdef USE_OPTIX
    OptixState*         optixState_       = nullptr;
    OptixLaunchParams*  dLaunchParams_    = nullptr;
#endif
    bool useHardwareRT_ = false;

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
