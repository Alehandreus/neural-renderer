#pragma once

#include <cmath>

#include "ray.h"
#include "vec3.h"

struct HitInfo {
    bool hit;
    float distance;
    Vec3 normal;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
};

__host__ __device__ inline Triangle makeTriangle(Vec3 v0, Vec3 v1, Vec3 v2) {
    Vec3 normal = normalize(cross(v1 - v0, v2 - v0));
    return Triangle{v0, v1, v2, normal};
}

__host__ __device__ inline HitInfo intersectTriangle(const Ray& ray, const Triangle& tri) {
    const float kEpsilon = 1e-6f;
    Vec3 edge1 = tri.v1 - tri.v0;
    Vec3 edge2 = tri.v2 - tri.v0;
    Vec3 pvec = cross(ray.direction, edge2);
    float det = dot(edge1, pvec);
    if (fabsf(det) < kEpsilon) {
        return HitInfo{false, 0.0f, Vec3()};
    }

    float invDet = 1.0f / det;
    Vec3 tvec = ray.origin - tri.v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return HitInfo{false, 0.0f, Vec3()};
    }

    Vec3 qvec = cross(tvec, edge1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return HitInfo{false, 0.0f, Vec3()};
    }

    float t = dot(edge2, qvec) * invDet;
    if (t <= kEpsilon) {
        return HitInfo{false, 0.0f, Vec3()};
    }

    return HitInfo{true, t, tri.normal};
}
