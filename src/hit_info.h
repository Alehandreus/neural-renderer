#pragma once

#include "vec3.h"

// Minimal hit data from ray-triangle intersection
// Only stores what's needed during BVH traversal
struct PreliminaryHitInfo {
    float t = 1e30f;
    float u, v;           // Barycentric coordinates
    uint32_t primIdx;     // Triangle index

    __host__ __device__ bool valid() const { return t < 1e30f; }
};

// Full intersection data (computed after finding closest hit)
struct HitInfo {
    Vec3 position;
    float t;
    Vec3 shadingNormal;   // Interpolated + normal map applied
    Vec3 geometricNormal; // Flat triangle normal
    Vec2 uv;
    int materialId;       // -1 means use global material

    __host__ __device__ HitInfo()
        : position(), t(1e30f), shadingNormal(), geometricNormal(),
          uv(), materialId(-1) {}

    __host__ __device__ bool valid() const { return t < 1e30f; }
};
