#pragma once

#include "vec3.h"

struct RenderBasis {
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float fovY;
};
