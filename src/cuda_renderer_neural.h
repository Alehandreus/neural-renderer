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
    explicit RendererNeural(Scene& scene);
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
    float* inputs_ = nullptr;
    float* compactedInputs_ = nullptr;
    int* hitIndices_ = nullptr;
    int* hitCount_ = nullptr;

    float* outerHitPositions_ = nullptr;
    float* innerHitPositions_ = nullptr;
    float* rayDirections_ = nullptr;
    int* outerHitFlags_ = nullptr;

    float* hitPositions_ = nullptr;
    float* hitNormals_ = nullptr;
    float* hitColors_ = nullptr;
    int* hitFlags_ = nullptr;

    float* bouncePositions_ = nullptr;
    float* bounceNormals_ = nullptr;
    float* bounceDirs_ = nullptr;
    float* bounceColors_ = nullptr;
    int* bounceHitFlags_ = nullptr;

    float* bounce2Positions_ = nullptr;
    float* bounce2Normals_ = nullptr;
    float* bounce2Dirs_ = nullptr;
    float* bounce2Colors_ = nullptr;
    int* bounce2HitFlags_ = nullptr;

    float* envDirs_ = nullptr;
    int* envHitFlags_ = nullptr;

    Vec3* pathThroughput_ = nullptr;
    Vec3* pathRadiance_ = nullptr;
    int* pathActive_ = nullptr;

    size_t bufferElements_ = 0;
    size_t accumPixels_ = 0;
    size_t paramsBytes_ = 0;
    uint32_t outputDims_ = 0;
    size_t outputElemSize_ = 0;

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
    uint32_t accumSampleCount_ = 0;

    bool lastUseNeuralQuery_ = true;
    bool lastLambertView_ = false;
    int lastBounceCount_ = -1;
    int lastSamplesPerPixel_ = -1;
    bool hasLastCamera_ = false;
    Vec3 lastCamPos_{};
    RenderBasis lastBasis_{};
    float lastFovY_ = 0.0f;
};
