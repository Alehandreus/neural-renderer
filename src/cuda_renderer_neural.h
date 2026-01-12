#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "renderer.h"

namespace tcnn {
namespace cpp {
class Module;
}  // namespace cpp
}  // namespace tcnn

class Scene;

class RendererNeural final {
 public:
    enum class GradientMode {
        InputOnly,
        WeightsOnly,
        InputAndWeights
    };

    explicit RendererNeural(Scene& scene);
    ~RendererNeural();

    void resize(int width, int height);
    void setCameraBasis(const RenderBasis& basis);
    void render(const Vec3& camPos, std::vector<uchar4>& hostPixels);
    void setGradientMode(GradientMode mode) { gradientMode_ = mode; }
    GradientMode gradientMode() const { return gradientMode_; }
    void setUseNeuralQuery(bool enabled) { useNeuralQuery_ = enabled; }
    bool useNeuralQuery() const { return useNeuralQuery_; }
    bool loadWeightsFromFile(const std::string& path);
    void setLossView(bool enabled) { lossView_ = enabled; }
    void setGdSteps(int steps) { gdSteps_ = steps; }
    void setSamplesPerPixel(int samples) { samplesPerPixel_ = samples; }
    void setBounceCount(int count) { bounceCount_ = count; }
    void setLambertView(bool enabled) { lambertView_ = enabled; }
    float averageLoss() const { return lastAvgLoss_; }
    int gdSteps() const { return gdSteps_; }
    int samplesPerPixel() const { return samplesPerPixel_; }
    int bounceCount() const { return bounceCount_; }
    size_t paramsBytes() const { return paramsBytes_; }
    int width() const { return width_; }
    int height() const { return height_; }

 private:
    void release();
    void releaseNetwork();
    bool ensureNetworkBuffers(size_t elementCount);
    bool ensureAccumBuffer(size_t pixelCount);
    void resetAccum();

    Scene* scene_ = nullptr;
    tcnn::cpp::Module* network_ = nullptr;
    void* params_ = nullptr;
    void* outputs_ = nullptr;
    void* dL_doutput_ = nullptr;
    float* dL_dinput_ = nullptr;
    void* dL_dparams_ = nullptr;
    float* inputs_ = nullptr;
    float* hitPositions_ = nullptr;
    float* hitColors_ = nullptr;
    float* compactedInputs_ = nullptr;
    float* compactedDLDInput_ = nullptr;
    float* normals_ = nullptr;
    float* bounceInputs_ = nullptr;
    float* bouncePositions_ = nullptr;
    float* bounceNormals_ = nullptr;
    float* bounceDirs_ = nullptr;
    float* bounce2Inputs_ = nullptr;
    float* bounce2Positions_ = nullptr;
    float* bounce2Normals_ = nullptr;
    float* bounce2Dirs_ = nullptr;
    float* envDirs_ = nullptr;
    float* lossValues_ = nullptr;
    float* lossMax_ = nullptr;
    float* lossSum_ = nullptr;
    int* lossHitCount_ = nullptr;
    int* hitFlags_ = nullptr;
    int* bounceHitFlags_ = nullptr;
    int* bounce2HitFlags_ = nullptr;
    int* envHitFlags_ = nullptr;
    int* hitIndices_ = nullptr;
    int* hitCount_ = nullptr;
    size_t bufferElements_ = 0;
    size_t accumPixels_ = 0;
    size_t paramsBytes_ = 0;
    uint32_t outputDims_ = 0;
    size_t outputElemSize_ = 0;
    GradientMode gradientMode_ = GradientMode::InputOnly;
    RenderBasis basis_{};
    Vec3 lightDir_{};
    int width_ = 0;
    int height_ = 0;
    uchar4* devicePixels_ = nullptr;
    Vec3* accum_ = nullptr;
    bool lossView_ = false;
    bool lambertView_ = false;
    int gdSteps_ = 0;
    int samplesPerPixel_ = 1;
    int bounceCount_ = 0;
    uint32_t accumSampleCount_ = 0;
    float lastAvgLoss_ = 0.0f;
    int lastHitCount_ = 0;
    bool useNeuralQuery_ = true;
    bool lastUseNeuralQuery_ = true;
    bool lastLambertView_ = false;
    int lastBounceCount_ = -1;
    int lastSamplesPerPixel_ = -1;
    bool hasLastCamera_ = false;
    Vec3 lastCamPos_{};
    RenderBasis lastBasis_{};
    float lastFovY_ = 0.0f;
};
