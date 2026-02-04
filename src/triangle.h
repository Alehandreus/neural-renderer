#pragma once

#include <cmath>

#include "ray.h"
#include "vec3.h"

struct HitInfo {
    bool hit;
    float distance;
    Vec3 normal;
    Vec3 color;
    Vec2 uv;
    int texId;
    int normalTexId;
    int triIndex;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
    Vec3 c0;
    Vec3 c1;
    Vec3 c2;
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    int texId;
    int normalTexId;
    int _pad1;
    int _pad2;
};

__host__ __device__ inline Triangle makeTriangle(Vec3 v0, Vec3 v1, Vec3 v2) {
    Vec3 normal = normalize(cross(v1 - v0, v2 - v0));
    Vec3 white(1.0f, 1.0f, 1.0f);
    return Triangle{v0, v1, v2, normal, normal, normal, normal, white, white, white,
                    Vec2(), Vec2(), Vec2(), -1, -1, 0, 0};
}

__host__ __device__ inline HitInfo intersectTriangle(const Ray& ray, const Triangle& tri) {
    const float kEpsilon = 1e-8f;
    Vec3 edge1 = tri.v1 - tri.v0;
    Vec3 edge2 = tri.v2 - tri.v0;
    Vec3 pvec = cross(ray.direction, edge2);
    float det = dot(edge1, pvec);
    if (fabsf(det) < kEpsilon) {
        return HitInfo{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1, -1};
    }

    float invDet = 1.0f / det;
    Vec3 tvec = ray.origin - tri.v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return HitInfo{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1, -1};
    }

    Vec3 qvec = cross(tvec, edge1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return HitInfo{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1, -1};
    }

    float t = dot(edge2, qvec) * invDet;
    if (t <= kEpsilon) {
        return HitInfo{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1, -1};
    }

    float w = 1.0f - u - v;
    Vec3 normal = tri.n0 * w + tri.n1 * u + tri.n2 * v;
    float nlen = length(normal);
    if (nlen > 0.0f) {
        normal = normal / nlen;
    } else {
        normal = tri.normal;
    }

    Vec3 color = tri.c0 * w + tri.c1 * u + tri.c2 * v;
    Vec2 uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;

    return HitInfo{true, t, normal, color, uv, tri.texId, tri.normalTexId, -1};
}
