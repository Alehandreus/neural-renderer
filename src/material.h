#pragma once

#include "vec3.h"

struct Material {
    Vec3 color{0.9f, 0.55f, 0.35f};
    // float reflectiveness = 1.0f;
    float reflectiveness = 0.9f;

    // Vec3 color{1.0f, 1.0f, 1.0f};
    // float reflectiveness = 1.0f;
};
