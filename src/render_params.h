#pragma once

#include <cstdint>

#include "material.h"
#include "vec3.h"

struct RenderParams {
    Vec3 camPos;
    Vec3 camForward;
    Vec3 camRight;
    Vec3 camUp;
    Vec3 lightDir;
    Vec3 outerShellMin;
    Vec3 outerShellInvExtent;
    Material material;
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
