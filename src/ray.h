#pragma once

#include "vec3.h"

struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray() : origin(), direction() {}
    __host__ __device__ Ray(Vec3 origin_, Vec3 direction_) : origin(origin_), direction(direction_) {}

    __host__ __device__ Vec3 at(float t) const {
        return origin + direction * t;
    }
};
