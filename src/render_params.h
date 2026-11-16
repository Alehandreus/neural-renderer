#pragma once

#include <cstdint>

#include "material.h"
#include "vec3.h"

// Lambert shading configuration.  The defaults reproduce this renderer's own
// headlight look; `litertMode` instead reproduces LiteRT's
// MULTI_RENDER_MODE_LAMBERT_NO_TEX exactly, so images are comparable against the
// other representations in the LiteRT benchmark (MESH, SCOM2, NEURAL_SDF, NBVH).
//
// LiteRT reference: MultiRenderer_resolve.cpp:617-664 shades with a white base
// colour, MultiRenderer_host.cpp:59 sets one direct light normalize(1,1,1) at
// 2/3 plus ambient 0.25, misses return the background colour unshaded, and
// nothing applies a gamma transfer on the way to 8-bit.
struct LambertSettings {
    bool litertMode = false;
    Vec3 lightDir{1.0f, 1.0f, 1.0f};   // normalized when applied
    float dirIntensity = 2.0f / 3.0f;
    float ambient = 0.25f;
    Vec3 background{1.0f, 1.0f, 1.0f};
    bool whiteBaseColor = true;        // ignore per-material colours, like LAMBERT_NO_TEX
    bool encodeSrgb = false;           // LiteRT writes linear values straight to 8-bit
};

struct RenderParams {
    Vec3 camPos;
    Vec3 camForward;
    Vec3 camRight;
    Vec3 camUp;
    Vec3 lightDir;
    Vec3 outerShellMin;
    Vec3 outerShellInvExtent;
    Material material;
    bool useConstantNeuralColor;
    Vec3 constantNeuralColor;
    bool useDirectEnvColor;
    Vec3 directEnvColor;
    LambertSettings lambert;
    bool centerRays;
    int strataPerAxis;   // centerRays: sqrt(total spp) grid, LiteRT RAY_GEN_MODE_REGULAR; 0 = pixel centre
    float fovY;
    float maxRadiance;
    float sceneScale;
    int maxBounces;
    int width;
    int height;
    int pixelCount;
    int samplesPerPixel;
    uint32_t sampleOffset;
};
