#pragma once

// SW BVH traversal helpers shared between cuda_renderer_neural.cu and
// (when USE_OPTIX is defined) src/rt/optix_programs.cu.
//
// This header is device-only (all functions are __device__ inline).
// Include order: mesh_intersection.cuh pulls in mesh.h, bvh_data.h, hit_info.h,
// material.h, ray.h, vec3.h.  render_params.h adds RenderParams.

#include "mesh_intersection.cuh"
#include "render_params.h"

// ===========================================================================
// RNG helpers
// ===========================================================================

__device__ inline uint32_t wangHash(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

__device__ inline uint32_t initRng(int pixelIdx, uint32_t sampleOffset, int sampleIdx) {
    uint32_t seed = static_cast<uint32_t>(pixelIdx) * 9781u + (sampleOffset + sampleIdx + 1u) * 6271u;
    return wangHash(seed);
}

__device__ inline float rand01(uint32_t& state) {
    state = wangHash(state);
    return (state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// ===========================================================================
// Primary ray generation
// ===========================================================================

__device__ inline Ray generatePrimaryRay(int x, int y, const RenderParams& params, uint32_t& rng) {
    float jitterX = rand01(rng);
    float jitterY = rand01(rng);
    float aspect = static_cast<float>(params.width) / static_cast<float>(params.height);
    float u = (static_cast<float>(x) + jitterX) / static_cast<float>(params.width);
    float v = 1.0f - (static_cast<float>(y) + jitterY) / static_cast<float>(params.height);
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    float tanHalfFov = tanf(params.fovY * 0.5f);
    Vec3 dir = params.camForward +
               params.camRight * (u * aspect * tanHalfFov) +
               params.camUp * (v * tanHalfFov);
    dir = normalize(dir);
    return Ray(params.camPos, dir);
}

// ===========================================================================
// AABB intersection
// ===========================================================================

__device__ inline bool intersectAabb(const Ray& ray,
                                     const Vec3& invDir,
                                     const Vec3& boundsMin,
                                     const Vec3& boundsMax,
                                     float tMax,
                                     float* outTNear) {
    const float kAabbEpsilon = 1e-10f;
    Vec3 minB(boundsMin.x - kAabbEpsilon, boundsMin.y - kAabbEpsilon, boundsMin.z - kAabbEpsilon);
    Vec3 maxB(boundsMax.x + kAabbEpsilon, boundsMax.y + kAabbEpsilon, boundsMax.z + kAabbEpsilon);

    float t1 = (minB.x - ray.origin.x) * invDir.x;
    float t2 = (maxB.x - ray.origin.x) * invDir.x;
    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);

    t1 = (minB.y - ray.origin.y) * invDir.y;
    t2 = (maxB.y - ray.origin.y) * invDir.y;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));

    t1 = (minB.z - ray.origin.z) * invDir.z;
    t2 = (maxB.z - ray.origin.z) * invDir.z;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));

    if (tmax < 0.0f || tmin > tMax || tmin > tmax) {
        return false;
    }
    if (outTNear) {
        *outTNear = tmin;
    }
    return true;
}

// ===========================================================================
// TraceMode — controls which triangle faces are considered for intersection.
// FORWARD_ONLY: front-facing (normal·dir < 0) — shell entry.
// BACKWARD_ONLY: back-facing (normal·dir > 0) — shell exit.
// ANY: all faces regardless of orientation.
//
// OptiX mapping:
//   FORWARD_ONLY  → OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
//   BACKWARD_ONLY → OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
//   ANY           → OPTIX_RAY_FLAG_NONE
// ===========================================================================

enum class TraceMode {
    ANY,
    FORWARD_ONLY,
    BACKWARD_ONLY,
};

// ===========================================================================
// Triangle geometric normal helper
// ===========================================================================

__device__ inline Vec3 getTriangleNormal(const MeshDeviceView& mesh, int triIdx) {
    uint3 idx = mesh.indices[triIdx];
    Vec3 v0 = mesh.vertices[idx.x];
    Vec3 v1 = mesh.vertices[idx.y];
    Vec3 v2 = mesh.vertices[idx.z];
    return normalize(cross(v1 - v0, v2 - v0));
}

// ===========================================================================
// BVH traversal — iterative DFS, software path
// ===========================================================================

__device__ inline bool traceMeshWithMode(const Ray& ray,
                                         MeshDeviceView mesh,
                                         HitInfo* outHit,
                                         TraceMode mode,
                                         bool /*allowNegative*/,
                                         const Material& globalMaterial) {
    if (mesh.numBvhNodes <= 0 || mesh.numTriangles <= 0) {
        return false;
    }

    PreliminaryHitInfo bestPi;
    bestPi.t = 1e30f;
    uint32_t bestTriIdx = 0;
    float minT = 1e-6f;
    Vec3 invDir(
        1.0f / ray.direction.x,
        1.0f / ray.direction.y,
        1.0f / ray.direction.z);

    constexpr int kMaxStack = 256;
    int stack[kMaxStack];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        if (nodeIndex < 0 || nodeIndex >= mesh.numBvhNodes) {
            continue;
        }

        const BvhNode node = mesh.bvhNodes[nodeIndex];
        float nodeTNear = 0.0f;
        if (!intersectAabb(ray, invDir, node.boundsMin, node.boundsMax, bestPi.t, &nodeTNear)) {
            continue;
        }

        if (node.isLeaf) {
            int start = node.first;
            int end = start + node.count;
            for (int i = start; i < end; ++i) {
                uint3 idx = mesh.indices[i];
                Vec3 v0 = mesh.vertices[idx.x];
                Vec3 v1 = mesh.vertices[idx.y];
                Vec3 v2 = mesh.vertices[idx.z];

                Vec3 triNormal = normalize(cross(v1 - v0, v2 - v0));
                float facingDot = dot(triNormal, ray.direction);

                if (mode == TraceMode::FORWARD_ONLY && facingDot >= 0.0f) {
                    continue;
                }
                if (mode == TraceMode::BACKWARD_ONLY && facingDot <= 0.0f) {
                    continue;
                }

                PreliminaryHitInfo pi = intersectTriangleIndexed(ray, v0, v1, v2);
                if (pi.valid() && pi.t > minT && pi.t < bestPi.t) {
                    bestPi = pi;
                    bestTriIdx = static_cast<uint32_t>(i);
                }
            }
        } else {
            int left = node.left;
            int right = node.right;
            float leftNear = 0.0f;
            float rightNear = 0.0f;
            bool hitLeft = false;
            bool hitRight = false;
            if (left >= 0 && left < mesh.numBvhNodes) {
                const BvhNode leftNode = mesh.bvhNodes[left];
                hitLeft = intersectAabb(ray, invDir, leftNode.boundsMin, leftNode.boundsMax, bestPi.t, &leftNear);
            }
            if (right >= 0 && right < mesh.numBvhNodes) {
                const BvhNode rightNode = mesh.bvhNodes[right];
                hitRight = intersectAabb(ray, invDir, rightNode.boundsMin, rightNode.boundsMax, bestPi.t, &rightNear);
            }

            if (hitLeft && hitRight) {
                int first = left;
                int second = right;
                if (rightNear < leftNear) {
                    first = right;
                    second = left;
                }
                if (stackSize < kMaxStack) stack[stackSize++] = second;
                if (stackSize < kMaxStack) stack[stackSize++] = first;
            } else if (hitLeft) {
                if (stackSize < kMaxStack) stack[stackSize++] = left;
            } else if (hitRight) {
                if (stackSize < kMaxStack) stack[stackSize++] = right;
            }
        }
    }

    if (bestPi.valid()) {
        bestPi.primIdx = bestTriIdx;
        HitInfo hitInfo = computeHitData(bestPi, bestTriIdx, ray, mesh);
        if (outHit) {
            *outHit = hitInfo;
        }
        return true;
    }

    return false;
}

__device__ inline bool traceMesh(const Ray& ray,
                                 MeshDeviceView mesh,
                                 HitInfo* outHit,
                                 bool cullBackfaces,
                                 const Material& globalMaterial) {
    TraceMode mode = cullBackfaces ? TraceMode::FORWARD_ONLY : TraceMode::ANY;
    return traceMeshWithMode(ray, mesh, outHit, mode, false, globalMaterial);
}
