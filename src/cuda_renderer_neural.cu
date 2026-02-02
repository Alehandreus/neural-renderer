#include "cuda_renderer_neural.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <utility>

#include <cuda_fp16.h>

#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/common_device.h>

#include "scene.h"

namespace {

struct RenderParams {
    Vec3 camPos;
    Vec3 camForward;
    Vec3 camRight;
    Vec3 camUp;
    Vec3 lightDir;
    Vec3 outerShellMin;
    Vec3 outerShellInvExtent;
    Vec3 materialColor;
    float fovY;
    float materialReflectiveness;
    float maxRadiance;
    int maxBounces;
    int width;
    int height;
    int pixelCount;
    int samplesPerPixel;
    uint32_t sampleOffset;
};

__device__ inline float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ inline Vec3 mul(Vec3 a, Vec3 b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline Vec3 srgbToLinear(Vec3 c) {
    auto convert = [](float v) {
        if (v <= 0.04045f) {
            return v / 12.92f;
        }
        return powf((v + 0.055f) / 1.055f, 2.4f);
    };
    return Vec3(convert(c.x), convert(c.y), convert(c.z));
}

__device__ inline float linearToSrgb(float v) {
    v = clampf(v, 0.0f, 1.0f);
    return powf(v, 1.0f / 2.2f);
}

__device__ inline Vec3 encodeSrgb(Vec3 c) {
    return Vec3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z));
}

__device__ inline Vec3 sampleTextureDevice(const TextureDeviceView* textures,
                                           int textureCount,
                                           int texId,
                                           float u,
                                           float v,
                                           bool nearestFilter) {
    if (!textures || texId < 0 || texId >= textureCount) {
        return Vec3(1.0f, 1.0f, 1.0f);
    }
    TextureDeviceView tex = textures[texId];
    if (!tex.pixels || tex.width <= 0 || tex.height <= 0 || tex.channels < 3) {
        return Vec3(1.0f, 1.0f, 1.0f);
    }

    u = u - floorf(u);
    v = v - floorf(v);
    v = 1.0f - v;

    auto fetch = [&](int xi, int yi) {
        int idx = (yi * tex.width + xi) * tex.channels;
        float r = tex.pixels[idx + 0] * (1.0f / 255.0f);
        float g = tex.pixels[idx + 1] * (1.0f / 255.0f);
        float b = tex.pixels[idx + 2] * (1.0f / 255.0f);
        return srgbToLinear(Vec3(r, g, b));
    };

    if (nearestFilter) {
        int x = static_cast<int>(u * static_cast<float>(tex.width));
        int y = static_cast<int>(v * static_cast<float>(tex.height));
        if (x < 0) {
            x = 0;
        } else if (x >= tex.width) {
            x = tex.width - 1;
        }
        if (y < 0) {
            y = 0;
        } else if (y >= tex.height) {
            y = tex.height - 1;
        }
        return fetch(x, y);
    }

    float x = u * static_cast<float>(tex.width - 1);
    float y = v * static_cast<float>(tex.height - 1);
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, tex.width - 1);
    int y1 = min(y0 + 1, tex.height - 1);
    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);

    Vec3 c00 = fetch(x0, y0);
    Vec3 c10 = fetch(x1, y0);
    Vec3 c01 = fetch(x0, y1);
    Vec3 c11 = fetch(x1, y1);
    Vec3 c0 = lerp(c00, c10, tx);
    Vec3 c1 = lerp(c01, c11, tx);
    return lerp(c0, c1, ty);
}

__device__ inline Vec3 sampleEnvironment(EnvironmentDeviceView env, Vec3 dir) {
    if (env.pixels && env.width > 0 && env.height > 0) {
        const float kInvTwoPi = 0.15915494309189535f;
        const float kInvPi = 0.3183098861837907f;
        float u = 0.5f + atan2f(dir.z, dir.x) * kInvTwoPi;
        float v = 0.5f - asinf(clampf(dir.y, -1.0f, 1.0f)) * kInvPi;
        u -= floorf(u);
        v = clampf(v, 0.0f, 1.0f);
        int x = static_cast<int>(u * static_cast<float>(env.width));
        int y = static_cast<int>(v * static_cast<float>(env.height));
        if (x >= env.width) {
            x = env.width - 1;
        }
        if (y >= env.height) {
            y = env.height - 1;
        }
        return env.pixels[y * env.width + x];
    }

    Vec3 skyTop(0.2f, 0.4f, 0.7f);
    Vec3 skyBottom(0.8f, 0.9f, 1.0f);
    float skyT = 0.5f * (dir.y + 1.0f);
    return lerp(skyBottom, skyTop, skyT);
}

__device__ inline Vec3 clampRadiance(Vec3 c, float maxLum) {
    if (maxLum <= 0.0f) {
        return c;
    }
    float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    if (lum > maxLum) {
        float scale = maxLum / lum;
        return c * scale;
    }
    return c;
}

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

__device__ inline Vec3 reflectDir(Vec3 dir, Vec3 normal) {
    return dir - normal * (2.0f * dot(dir, normal));
}

__device__ inline Vec3 sampleHemisphereCosine(Vec3 normal, uint32_t& rng) {
    float u1 = rand01(rng);
    float u2 = rand01(rng);
    float r = sqrtf(u1);
    float theta = 2.0f * 3.14159265358979323846f * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    Vec3 up = fabsf(normal.y) < 0.999f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangent = normalize(cross(up, normal));
    Vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

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

__device__ inline bool intersectAabb(const Ray& ray,
                                     const Vec3& invDir,
                                     const Vec3& boundsMin,
                                     const Vec3& boundsMax,
                                     float tMax,
                                     float* outTNear) {
    const float kAabbEpsilon = 1e-5f;
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

__device__ inline bool traceMesh(const Ray& ray,
                                 MeshDeviceView mesh,
                                 HitInfo* outHit,
                                 bool cullBackfaces = true) {
    if (mesh.nodeCount <= 0 || mesh.triangleCount <= 0) {
        return false;
    }

    HitInfo bestHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
    float closestT = 1e30f;
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
        if (nodeIndex < 0 || nodeIndex >= mesh.nodeCount) {
            continue;
        }

        const BvhNode node = mesh.nodes[nodeIndex];
        float nodeTNear = 0.0f;
        if (!intersectAabb(ray, invDir, node.boundsMin, node.boundsMax, closestT, &nodeTNear)) {
            continue;
        }

        if (node.isLeaf) {
            int start = node.first;
            int end = start + node.count;
            for (int i = start; i < end; ++i) {
                const Triangle& tri = mesh.triangles[i];
                if (cullBackfaces && dot(tri.normal, ray.direction) >= 0.0f) {
                    continue;
                }
                HitInfo hit = intersectTriangle(ray, tri);
                if (hit.hit && hit.distance < closestT) {
                    closestT = hit.distance;
                    bestHit = hit;
                    bestHit.triIndex = i;
                }
            }
        } else {
            int left = node.left;
            int right = node.right;
            float leftNear = 0.0f;
            float rightNear = 0.0f;
            bool hitLeft = false;
            bool hitRight = false;
            if (left >= 0 && left < mesh.nodeCount) {
                const BvhNode leftNode = mesh.nodes[left];
                hitLeft = intersectAabb(ray, invDir, leftNode.boundsMin, leftNode.boundsMax, closestT, &leftNear);
            }
            if (right >= 0 && right < mesh.nodeCount) {
                const BvhNode rightNode = mesh.nodes[right];
                hitRight = intersectAabb(ray, invDir, rightNode.boundsMin, rightNode.boundsMax, closestT, &rightNear);
            }

            if (hitLeft && hitRight) {
                int first = left;
                int second = right;
                if (rightNear < leftNear) {
                    first = right;
                    second = left;
                }
                if (stackSize < kMaxStack) {
                    stack[stackSize++] = second;
                }
                if (stackSize < kMaxStack) {
                    stack[stackSize++] = first;
                }
            } else if (hitLeft) {
                if (stackSize < kMaxStack) {
                    stack[stackSize++] = left;
                }
            } else if (hitRight) {
                if (stackSize < kMaxStack) {
                    stack[stackSize++] = right;
                }
            }
        }
    }

    if (bestHit.hit) {
        Vec3 texColor = sampleTextureDevice(
            mesh.textures,
            mesh.textureCount,
            bestHit.texId,
            bestHit.uv.x,
            bestHit.uv.y,
            mesh.textureNearest != 0);
        bestHit.color = mul(bestHit.color, texColor);
    }
    if (outHit) {
        *outHit = bestHit;
    }
    return bestHit.hit;
}

// ---------------------------------------------------------------------------
// Shell tracing: for each pixel, trace outer shell then inner shell.
// ---------------------------------------------------------------------------
__global__ void traceShellsKernel(float* outerHitPositions,
                                  float* innerHitPositions,
                                  float* rayDirections,
                                  int* outerHitFlags,
                                  RenderParams params,
                                  MeshDeviceView outerShell,
                                  MeshDeviceView innerShell) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray ray = generatePrimaryRay(x, y, params, rng);

        rayDirections[base + 0] = ray.direction.x;
        rayDirections[base + 1] = ray.direction.y;
        rayDirections[base + 2] = ray.direction.z;

        HitInfo outerHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
        bool hitOuter = traceMesh(ray, outerShell, &outerHit, false);
        if (hitOuter) {
            Vec3 outerPos = ray.at(outerHit.distance);
            outerHitPositions[base + 0] = outerPos.x;
            outerHitPositions[base + 1] = outerPos.y;
            outerHitPositions[base + 2] = outerPos.z;
            outerHitFlags[sampleIdx] = 1;

            HitInfo innerHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
            bool hitInner = traceMesh(ray, innerShell, &innerHit, false);
            if (hitInner) {
                Vec3 innerPos = ray.at(innerHit.distance);
                innerHitPositions[base + 0] = innerPos.x;
                innerHitPositions[base + 1] = innerPos.y;
                innerHitPositions[base + 2] = innerPos.z;
            } else {
                innerHitPositions[base + 0] = 0.0f;
                innerHitPositions[base + 1] = 0.0f;
                innerHitPositions[base + 2] = 0.0f;
            }
        } else {
            outerHitPositions[base + 0] = 0.0f;
            outerHitPositions[base + 1] = 0.0f;
            outerHitPositions[base + 2] = 0.0f;
            innerHitPositions[base + 0] = 0.0f;
            innerHitPositions[base + 1] = 0.0f;
            innerHitPositions[base + 2] = 0.0f;
            outerHitFlags[sampleIdx] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Compact outer shell hits.
// ---------------------------------------------------------------------------
__global__ void compactInputsKernel(const int* flags,
                                    int count,
                                    int* hitIndices,
                                    int* hitCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    if (flags[idx]) {
        int slot = atomicAdd(hitCount, 1);
        hitIndices[slot] = idx;
    }
}

// ---------------------------------------------------------------------------
// Build 3 separate encoding input buffers for compacted hits:
//   compactedOuterPos: normalized outer positions
//   compactedInnerPos: normalized inner positions (same outer shell bounds)
//   compactedDirs: directions mapped from [-1,1] to [0,1]
// ---------------------------------------------------------------------------
__global__ void buildNeuralInputsKernel(const float* outerHitPositions,
                                        const float* innerHitPositions,
                                        const float* rayDirections,
                                        const int* hitIndices,
                                        int hitCount,
                                        Vec3 outerMin,
                                        Vec3 outerInvExtent,
                                        float* compactedOuterPos,
                                        float* compactedInnerPos,
                                        float* compactedDirs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }
    int sampleIdx = hitIndices[idx];
    int srcBase = sampleIdx * 3;
    int dstBase = idx * 3;

    Vec3 outerPos(outerHitPositions[srcBase + 0],
                  outerHitPositions[srcBase + 1],
                  outerHitPositions[srcBase + 2]);
    Vec3 innerPos(innerHitPositions[srcBase + 0],
                  innerHitPositions[srcBase + 1],
                  innerHitPositions[srcBase + 2]);

    // Normalize positions w.r.t. outer shell bounds.
    Vec3 normOuter((outerPos.x - outerMin.x) * outerInvExtent.x,
                   (outerPos.y - outerMin.y) * outerInvExtent.y,
                   (outerPos.z - outerMin.z) * outerInvExtent.z);
    Vec3 normInner((innerPos.x - outerMin.x) * outerInvExtent.x,
                   (innerPos.y - outerMin.y) * outerInvExtent.y,
                   (innerPos.z - outerMin.z) * outerInvExtent.z);

    compactedOuterPos[dstBase + 0] = normOuter.x;
    compactedOuterPos[dstBase + 1] = normOuter.y;
    compactedOuterPos[dstBase + 2] = normOuter.z;
    compactedInnerPos[dstBase + 0] = normInner.x;
    compactedInnerPos[dstBase + 1] = normInner.y;
    compactedInnerPos[dstBase + 2] = normInner.z;

    // Directions: map from [-1,1] to [0,1] matching Python (directions + 1) / 2.
    Vec3 dir(rayDirections[srcBase + 0],
             rayDirections[srcBase + 1],
             rayDirections[srcBase + 2]);
    compactedDirs[dstBase + 0] = (dir.x + 1.0f) * 0.5f;
    compactedDirs[dstBase + 1] = (dir.y + 1.0f) * 0.5f;
    compactedDirs[dstBase + 2] = (dir.z + 1.0f) * 0.5f;
}

// ---------------------------------------------------------------------------
// Concatenate three FP16 encoding outputs into a single FP32 MLP input buffer.
// tcnn C++ API forward() takes float* and handles precision internally.
// Layout per element: [encA(dimA), encB(dimB), encC(dimC)]
// ---------------------------------------------------------------------------
__global__ void concatenateEncodingsKernel(const __half* encA,
                                           uint32_t dimA,
                                           const __half* encB,
                                           uint32_t dimB,
                                           const __half* encC,
                                           uint32_t dimC,
                                           float* output,
                                           uint32_t totalDim,
                                           int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int outBase = idx * totalDim;
    int aBase = idx * dimA;
    int bBase = idx * dimB;
    int cBase = idx * dimC;

    int offset = 0;
    for (uint32_t i = 0; i < dimA; ++i) {
        output[outBase + offset] = __half2float(encA[aBase + i]);
        ++offset;
    }
    for (uint32_t i = 0; i < dimB; ++i) {
        output[outBase + offset] = __half2float(encB[bBase + i]);
        ++offset;
    }
    for (uint32_t i = 0; i < dimC; ++i) {
        output[outBase + offset] = __half2float(encC[cBase + i]);
        ++offset;
    }
}

// ---------------------------------------------------------------------------
// Apply neural network outputs (5D FP16) back to full-size buffers.
// Output layout per hit: [has_intersection, distance, normal_x, normal_y, normal_z]
// ---------------------------------------------------------------------------
__global__ void applyNeuralOutputKernel(const __half* outputs,
                                        const int* hitIndices,
                                        int hitCount,
                                        int outputStride,
                                        const float* outerHitPositions,
                                        const float* rayDirections,
                                        float* hitPositions,
                                        float* hitNormals,
                                        float* hitColors,
                                        int* hitFlags,
                                        Vec3 materialColor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }
    int sampleIdx = hitIndices[idx];
    int srcBase = sampleIdx * 3;
    int outBase = idx * outputStride;

    float hasIntersection = __half2float(outputs[outBase + 0]);
    float distance = __half2float(outputs[outBase + 1]);
    float nx = __half2float(outputs[outBase + 2]);
    float ny = __half2float(outputs[outBase + 3]);
    float nz = __half2float(outputs[outBase + 4]);

    if (hasIntersection > 0.0f) {
        Vec3 outerPos(outerHitPositions[srcBase + 0],
                      outerHitPositions[srcBase + 1],
                      outerHitPositions[srcBase + 2]);
        Vec3 dir(rayDirections[srcBase + 0],
                 rayDirections[srcBase + 1],
                 rayDirections[srcBase + 2]);

        Vec3 hitPos = outerPos + dir * distance;
        hitPositions[srcBase + 0] = hitPos.x;
        hitPositions[srcBase + 1] = hitPos.y;
        hitPositions[srcBase + 2] = hitPos.z;

        Vec3 normal(nx, ny, nz);
        float nlen = length(normal);
        if (nlen > 1e-6f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        hitNormals[srcBase + 0] = normal.x;
        hitNormals[srcBase + 1] = normal.y;
        hitNormals[srcBase + 2] = normal.z;

        hitColors[srcBase + 0] = materialColor.x;
        hitColors[srcBase + 1] = materialColor.y;
        hitColors[srcBase + 2] = materialColor.z;
        hitFlags[sampleIdx] = 1;
    } else {
        hitPositions[srcBase + 0] = 0.0f;
        hitPositions[srcBase + 1] = 0.0f;
        hitPositions[srcBase + 2] = 0.0f;
        hitNormals[srcBase + 0] = 0.0f;
        hitNormals[srcBase + 1] = 0.0f;
        hitNormals[srcBase + 2] = 0.0f;
        hitColors[srcBase + 0] = 0.0f;
        hitColors[srcBase + 1] = 0.0f;
        hitColors[srcBase + 2] = 0.0f;
        hitFlags[sampleIdx] = 0;
    }
}

// ---------------------------------------------------------------------------
// Primary ray tracing for original mesh (non-neural path).
// ---------------------------------------------------------------------------
__global__ void renderOriginalKernel(float* hitPositions,
                                     float* hitNormals,
                                     float* hitColors,
                                     int* hitFlags,
                                     RenderParams params,
                                     MeshDeviceView mesh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray ray = generatePrimaryRay(x, y, params, rng);

        HitInfo bestHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
        bool hit = traceMesh(ray, mesh, &bestHit);
        if (hit) {
            Vec3 hitPos = ray.at(bestHit.distance);
            hitPositions[base + 0] = hitPos.x;
            hitPositions[base + 1] = hitPos.y;
            hitPositions[base + 2] = hitPos.z;
            hitNormals[base + 0] = bestHit.normal.x;
            hitNormals[base + 1] = bestHit.normal.y;
            hitNormals[base + 2] = bestHit.normal.z;
            hitColors[base + 0] = bestHit.color.x;
            hitColors[base + 1] = bestHit.color.y;
            hitColors[base + 2] = bestHit.color.z;
            hitFlags[sampleIdx] = 1;
        } else {
            hitPositions[base + 0] = 0.0f;
            hitPositions[base + 1] = 0.0f;
            hitPositions[base + 2] = 0.0f;
            hitNormals[base + 0] = 0.0f;
            hitNormals[base + 1] = 0.0f;
            hitNormals[base + 2] = 0.0f;
            hitColors[base + 0] = 0.0f;
            hitColors[base + 1] = 0.0f;
            hitColors[base + 2] = 0.0f;
            hitFlags[sampleIdx] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Path tracing kernel (bounces on original mesh).
// ---------------------------------------------------------------------------
__global__ void pathTraceKernel(uchar4* output,
                                Vec3* accum,
                                const float* hitPositions,
                                const float* hitNormals,
                                const float* hitColors,
                                const int* hitFlags,
                                RenderParams params,
                                MeshDeviceView mesh,
                                EnvironmentDeviceView env) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    Vec3 sum(0.0f, 0.0f, 0.0f);
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    float reflectiveness = clampf(params.materialReflectiveness, 0.0f, 1.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);
        Vec3 radiance(0.0f, 0.0f, 0.0f);
        Vec3 throughput(1.0f, 1.0f, 1.0f);

        bool hit = hitFlags[sampleIdx] != 0;
        Vec3 hitPos;
        Vec3 normal;
        if (hit) {
            throughput = Vec3(
                    hitColors[sampleIdx * 3 + 0],
                    hitColors[sampleIdx * 3 + 1],
                    hitColors[sampleIdx * 3 + 2]);
            hitPos = Vec3(
                    hitPositions[sampleIdx * 3 + 0],
                    hitPositions[sampleIdx * 3 + 1],
                    hitPositions[sampleIdx * 3 + 2]);
            normal = Vec3(
                    hitNormals[sampleIdx * 3 + 0],
                    hitNormals[sampleIdx * 3 + 1],
                    hitNormals[sampleIdx * 3 + 2]);
        }

        Ray ray = primaryRay;
        int maxBounces = params.maxBounces;
        if (maxBounces < 0) {
            maxBounces = 0;
        }
        for (int bounce = 0; bounce <= maxBounces; ++bounce) {
            if (!hit) {
                Vec3 envLight = sampleEnvironment(env, ray.direction);
                envLight = clampRadiance(envLight, params.maxRadiance);
                radiance += mul(throughput, envLight);
                break;
            }
            if (bounce == maxBounces) {
                break;
            }

            float nlen = length(normal);
            if (nlen > 0.0f) {
                normal = normal / nlen;
            } else {
                normal = Vec3(0.0f, 1.0f, 0.0f);
            }
            if (dot(normal, ray.direction) > 0.0f) {
                radiance = Vec3(0.0f, 0.0f, 0.0f);
                break;
            }

            Vec3 bounceDir;
            float choose = rand01(rng);
            if (choose < reflectiveness) {
                bounceDir = normalize(reflectDir(ray.direction, normal));
            } else {
                bounceDir = sampleHemisphereCosine(normal, rng);
            }

            ray = Ray(hitPos + normal * 1e-3f, bounceDir);
            HitInfo bounceHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
            hit = traceMesh(ray, mesh, &bounceHit);
            if (!hit) {
                Vec3 envLight = sampleEnvironment(env, ray.direction);
                envLight = clampRadiance(envLight, params.maxRadiance);
                radiance += mul(throughput, envLight);
                break;
            }

            hitPos = ray.at(bounceHit.distance);
            normal = bounceHit.normal;
            throughput = mul(throughput, bounceHit.color);
        }

        sum += radiance;
    }

    Vec3 prev = accum[pixelIdx];
    Vec3 newSum = prev + sum;
    accum[pixelIdx] = newSum;
    float invSamples = 1.0f / static_cast<float>(params.sampleOffset + params.samplesPerPixel);
    Vec3 color = newSum * invSamples;

    color = encodeSrgb(color);

    output[pixelIdx] = make_uchar4(
            static_cast<unsigned char>(color.x * 255.0f),
            static_cast<unsigned char>(color.y * 255.0f),
            static_cast<unsigned char>(color.z * 255.0f),
            255);
}

// ---------------------------------------------------------------------------
// Lambert shading (no bounces).
// ---------------------------------------------------------------------------
__global__ void lambertKernel(uchar4* output,
                              const float* hitNormals,
                              const float* hitColors,
                              const int* hitFlags,
                              RenderParams params,
                              EnvironmentDeviceView env) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    Vec3 sum(0.0f, 0.0f, 0.0f);
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);

        Vec3 color(0.0f, 0.0f, 0.0f);
        if (hitFlags[sampleIdx]) {
            Vec3 normal(
                    hitNormals[sampleIdx * 3 + 0],
                    hitNormals[sampleIdx * 3 + 1],
                    hitNormals[sampleIdx * 3 + 2]);
            Vec3 baseColor(
                    hitColors[sampleIdx * 3 + 0],
                    hitColors[sampleIdx * 3 + 1],
                    hitColors[sampleIdx * 3 + 2]);
            float nlen = length(normal);
            if (nlen > 0.0f) {
                normal = normal / nlen;
            } else {
                normal = Vec3(0.0f, 1.0f, 0.0f);
            }
            if (dot(normal, primaryRay.direction) > 0.0f) {
                color = Vec3(0.0f, 0.0f, 0.0f);
            } else {
                float ndotl = fmaxf(0.0f, dot(normal, -primaryRay.direction));
                color = baseColor * ndotl;
            }
        } else {
            color = sampleEnvironment(env, primaryRay.direction);
        }

        sum += color;
    }

    Vec3 color = sum * (1.0f / static_cast<float>(params.samplesPerPixel));
    color = encodeSrgb(color);

    output[pixelIdx] = make_uchar4(
            static_cast<unsigned char>(color.x * 255.0f),
            static_cast<unsigned char>(color.y * 255.0f),
            static_cast<unsigned char>(color.z * 255.0f),
            255);
}

// ---------------------------------------------------------------------------
// Neural path tracing state kernels.
// ---------------------------------------------------------------------------
__global__ void initNeuralPathKernel(Vec3* throughput,
                                     Vec3* radiance,
                                     int* active,
                                     const int* hitFlags,
                                     const float* hitNormals,
                                     const float* hitColors,
                                     RenderParams params,
                                     EnvironmentDeviceView env) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);

        Vec3 sampleRadiance(0.0f, 0.0f, 0.0f);
        Vec3 sampleThroughput(1.0f, 1.0f, 1.0f);
        int isActive = 0;

        if (hitFlags[sampleIdx]) {
            int base = sampleIdx * 3;
            Vec3 normal(hitNormals[base + 0], hitNormals[base + 1], hitNormals[base + 2]);
            float nlen = length(normal);
            if (nlen > 0.0f) {
                normal = normal / nlen;
            } else {
                normal = Vec3(0.0f, 1.0f, 0.0f);
            }
            if (dot(normal, primaryRay.direction) > 0.0f) {
                throughput[sampleIdx] = Vec3(0.0f, 0.0f, 0.0f);
                radiance[sampleIdx] = Vec3(0.0f, 0.0f, 0.0f);
                active[sampleIdx] = 0;
                continue;
            }
            sampleThroughput = Vec3(hitColors[base + 0], hitColors[base + 1], hitColors[base + 2]);
            isActive = 1;
        } else {
            Vec3 envLight = sampleEnvironment(env, primaryRay.direction);
            envLight = clampRadiance(envLight, params.maxRadiance);
            sampleRadiance = envLight;
        }

        throughput[sampleIdx] = sampleThroughput;
        radiance[sampleIdx] = sampleRadiance;
        active[sampleIdx] = isActive;
    }
}

__global__ void renderBounceKernel(const float* hitPositions,
                                   const float* hitNormals,
                                   const int* hitFlags,
                                   int* pathActive,
                                   RenderParams params,
                                   MeshDeviceView mesh,
                                   float* bouncePositions,
                                   float* bounceNormals,
                                   float* bounceColors,
                                   int* bounceFlags,
                                   float* bounceDirs) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    float reflectiveness = clampf(params.materialReflectiveness, 0.0f, 1.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (!hitFlags[sampleIdx]) {
            bouncePositions[base + 0] = 0.0f;
            bouncePositions[base + 1] = 0.0f;
            bouncePositions[base + 2] = 0.0f;
            bounceNormals[base + 0] = 0.0f;
            bounceNormals[base + 1] = 0.0f;
            bounceNormals[base + 2] = 0.0f;
            bounceColors[base + 0] = 0.0f;
            bounceColors[base + 1] = 0.0f;
            bounceColors[base + 2] = 0.0f;
            bounceDirs[base + 0] = 0.0f;
            bounceDirs[base + 1] = 0.0f;
            bounceDirs[base + 2] = 0.0f;
            bounceFlags[sampleIdx] = 0;
            continue;
        }

        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);

        Vec3 hitPos(hitPositions[base + 0], hitPositions[base + 1], hitPositions[base + 2]);
        Vec3 normal(hitNormals[base + 0], hitNormals[base + 1], hitNormals[base + 2]);

        float nlen = length(normal);
        if (nlen > 0.0f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        if (dot(normal, primaryRay.direction) > 0.0f) {
            if (pathActive) {
                pathActive[sampleIdx] = 0;
            }
            bouncePositions[base + 0] = 0.0f;
            bouncePositions[base + 1] = 0.0f;
            bouncePositions[base + 2] = 0.0f;
            bounceNormals[base + 0] = 0.0f;
            bounceNormals[base + 1] = 0.0f;
            bounceNormals[base + 2] = 0.0f;
            bounceColors[base + 0] = 0.0f;
            bounceColors[base + 1] = 0.0f;
            bounceColors[base + 2] = 0.0f;
            bounceDirs[base + 0] = 0.0f;
            bounceDirs[base + 1] = 0.0f;
            bounceDirs[base + 2] = 0.0f;
            bounceFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 bounceDir;
        float choose = rand01(rng);
        if (choose < reflectiveness) {
            bounceDir = normalize(reflectDir(primaryRay.direction, normal));
        } else {
            bounceDir = sampleHemisphereCosine(normal, rng);
        }

        bounceDirs[base + 0] = bounceDir.x;
        bounceDirs[base + 1] = bounceDir.y;
        bounceDirs[base + 2] = bounceDir.z;

        Ray bounceRay(hitPos + normal * 1e-3f, bounceDir);
        HitInfo bounceHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
        bool hit = traceMesh(bounceRay, mesh, &bounceHit);
        if (hit) {
            Vec3 bouncePos = bounceRay.at(bounceHit.distance);
            bouncePositions[base + 0] = bouncePos.x;
            bouncePositions[base + 1] = bouncePos.y;
            bouncePositions[base + 2] = bouncePos.z;
            bounceNormals[base + 0] = bounceHit.normal.x;
            bounceNormals[base + 1] = bounceHit.normal.y;
            bounceNormals[base + 2] = bounceHit.normal.z;
            bounceColors[base + 0] = bounceHit.color.x;
            bounceColors[base + 1] = bounceHit.color.y;
            bounceColors[base + 2] = bounceHit.color.z;
            bounceFlags[sampleIdx] = 1;
        } else {
            bouncePositions[base + 0] = 0.0f;
            bouncePositions[base + 1] = 0.0f;
            bouncePositions[base + 2] = 0.0f;
            bounceNormals[base + 0] = 0.0f;
            bounceNormals[base + 1] = 0.0f;
            bounceNormals[base + 2] = 0.0f;
            bounceColors[base + 0] = 0.0f;
            bounceColors[base + 1] = 0.0f;
            bounceColors[base + 2] = 0.0f;
            bounceFlags[sampleIdx] = 0;
        }
    }
}

__global__ void renderBounceFromStateKernel(const float* inPositions,
                                            const float* inNormals,
                                            const int* inFlags,
                                            const float* inDirs,
                                            int* pathActive,
                                            uint32_t bounceIndex,
                                            RenderParams params,
                                            MeshDeviceView mesh,
                                            float* outPositions,
                                            float* outNormals,
                                            float* outColors,
                                            int* outFlags,
                                            float* outDirs) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    float reflectiveness = clampf(params.materialReflectiveness, 0.0f, 1.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (!inFlags[sampleIdx]) {
            outPositions[base + 0] = 0.0f;
            outPositions[base + 1] = 0.0f;
            outPositions[base + 2] = 0.0f;
            outNormals[base + 0] = 0.0f;
            outNormals[base + 1] = 0.0f;
            outNormals[base + 2] = 0.0f;
            outColors[base + 0] = 0.0f;
            outColors[base + 1] = 0.0f;
            outColors[base + 2] = 0.0f;
            outDirs[base + 0] = 0.0f;
            outDirs[base + 1] = 0.0f;
            outDirs[base + 2] = 0.0f;
            outFlags[sampleIdx] = 0;
            continue;
        }

        uint32_t rng = initRng(pixelIdx, params.sampleOffset + bounceIndex, s);

        Vec3 hitPos(inPositions[base + 0], inPositions[base + 1], inPositions[base + 2]);
        Vec3 normal(inNormals[base + 0], inNormals[base + 1], inNormals[base + 2]);
        Vec3 incoming(inDirs[base + 0], inDirs[base + 1], inDirs[base + 2]);

        float nlen = length(normal);
        if (nlen > 0.0f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        if (dot(normal, incoming) > 0.0f) {
            if (pathActive) {
                pathActive[sampleIdx] = 0;
            }
            outPositions[base + 0] = 0.0f;
            outPositions[base + 1] = 0.0f;
            outPositions[base + 2] = 0.0f;
            outNormals[base + 0] = 0.0f;
            outNormals[base + 1] = 0.0f;
            outNormals[base + 2] = 0.0f;
            outColors[base + 0] = 0.0f;
            outColors[base + 1] = 0.0f;
            outColors[base + 2] = 0.0f;
            outDirs[base + 0] = 0.0f;
            outDirs[base + 1] = 0.0f;
            outDirs[base + 2] = 0.0f;
            outFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 bounceDir;
        float choose = rand01(rng);
        if (choose < reflectiveness) {
            bounceDir = normalize(reflectDir(incoming, normal));
        } else {
            bounceDir = sampleHemisphereCosine(normal, rng);
        }

        outDirs[base + 0] = bounceDir.x;
        outDirs[base + 1] = bounceDir.y;
        outDirs[base + 2] = bounceDir.z;

        Ray bounceRay(hitPos + normal * 1e-3f, bounceDir);
        HitInfo bounceHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1, -1};
        bool hit = traceMesh(bounceRay, mesh, &bounceHit);
        if (hit) {
            Vec3 bouncePos = bounceRay.at(bounceHit.distance);
            outPositions[base + 0] = bouncePos.x;
            outPositions[base + 1] = bouncePos.y;
            outPositions[base + 2] = bouncePos.z;
            outNormals[base + 0] = bounceHit.normal.x;
            outNormals[base + 1] = bounceHit.normal.y;
            outNormals[base + 2] = bounceHit.normal.z;
            outColors[base + 0] = bounceHit.color.x;
            outColors[base + 1] = bounceHit.color.y;
            outColors[base + 2] = bounceHit.color.z;
            outFlags[sampleIdx] = 1;
        } else {
            outPositions[base + 0] = 0.0f;
            outPositions[base + 1] = 0.0f;
            outPositions[base + 2] = 0.0f;
            outNormals[base + 0] = 0.0f;
            outNormals[base + 1] = 0.0f;
            outNormals[base + 2] = 0.0f;
            outColors[base + 0] = 0.0f;
            outColors[base + 1] = 0.0f;
            outColors[base + 2] = 0.0f;
            outFlags[sampleIdx] = 0;
        }
    }
}

__global__ void integrateNeuralBounceKernel(Vec3* throughput,
                                            Vec3* radiance,
                                            int* active,
                                            const int* bounceHitFlags,
                                            const float* bounceColors,
                                            const float* bounceDirs,
                                            int bounceIndex,
                                            RenderParams params,
                                            EnvironmentDeviceView env) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        if (!active[sampleIdx]) {
            continue;
        }

        if (!bounceHitFlags[sampleIdx]) {
            int base = sampleIdx * 3;
            Vec3 envDir(bounceDirs[base + 0], bounceDirs[base + 1], bounceDirs[base + 2]);
            Vec3 envLight = sampleEnvironment(env, envDir);
            envLight = clampRadiance(envLight, params.maxRadiance);
            radiance[sampleIdx] = radiance[sampleIdx] + mul(throughput[sampleIdx], envLight);
            active[sampleIdx] = 0;
            continue;
        }

        if (bounceIndex >= params.maxBounces) {
            active[sampleIdx] = 0;
            continue;
        }

        int base = sampleIdx * 3;
        Vec3 color(bounceColors[base + 0], bounceColors[base + 1], bounceColors[base + 2]);
        throughput[sampleIdx] = mul(throughput[sampleIdx], color);
    }
}

__global__ void finalizeNeuralPathKernel(uchar4* output,
                                         Vec3* accum,
                                         const Vec3* radiance,
                                         RenderParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    Vec3 sum(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        sum += radiance[sampleIdx];
    }

    Vec3 prev = accum[pixelIdx];
    Vec3 newSum = prev + sum;
    accum[pixelIdx] = newSum;
    float invSamples = 1.0f / static_cast<float>(params.sampleOffset + params.samplesPerPixel);
    Vec3 color = newSum * invSamples;

    color = encodeSrgb(color);

    output[pixelIdx] = make_uchar4(
            static_cast<unsigned char>(color.x * 255.0f),
            static_cast<unsigned char>(color.y * 255.0f),
            static_cast<unsigned char>(color.z * 255.0f),
            255);
}

// ---------------------------------------------------------------------------
// Utility functions.
// ---------------------------------------------------------------------------
size_t precisionBytes(tcnn::cpp::Precision precision) {
    (void)precision;
    return sizeof(uint16_t);
}

size_t roundUp(size_t value, size_t granularity) {
    if (granularity == 0) {
        return value;
    }
    return ((value + granularity - 1) / granularity) * granularity;
}

void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

size_t randomSeed() {
    std::random_device device;
    size_t seed = static_cast<size_t>(device());
    if (seed == 0) {
        seed = 1;
    }
    return seed;
}

// Helper: allocate and initialize parameters for a module.
void* allocAndInitParams(tcnn::cpp::Module* module, size_t* outBytes) {
    size_t nParams = module->n_params();
    size_t bytes = nParams * precisionBytes(module->param_precision());
    *outBytes = bytes;
    if (bytes == 0) {
        return nullptr;
    }

    void* deviceParams = nullptr;
    checkCuda(cudaMalloc(&deviceParams, bytes), "cudaMalloc module params");

    float* paramsFull = nullptr;
    size_t fullBytes = nParams * sizeof(float);
    checkCuda(cudaMalloc(&paramsFull, fullBytes), "cudaMalloc module params full");

    size_t seed = randomSeed();
    module->initialize_params(seed, paramsFull, 1.0f);

    const uint32_t count = static_cast<uint32_t>(nParams);
    const int blockSize = 256;
    int gridSize = static_cast<int>((count + blockSize - 1) / blockSize);
    tcnn::cast<__half><<<gridSize, blockSize>>>(count, paramsFull, static_cast<__half*>(deviceParams));
    checkCuda(cudaGetLastError(), "module params cast");
    checkCuda(cudaFree(paramsFull), "cudaFree module params full");

    return deviceParams;
}

}  // namespace

// ===========================================================================
// RendererNeural implementation.
// ===========================================================================

RendererNeural::RendererNeural(Scene& scene)
        : scene_(&scene),
          lightDir_(normalize(Vec3(1.0f, 1.5f, -1.0f))) {
    if (!tcnn::cpp::has_networks()) {
        std::fprintf(stderr, "tiny-cuda-nn was built without network support.\n");
        return;
    }

    // Point encoding: HashGrid matching Python config.
    tcnn::cpp::json pointEncConfig = {
            {"otype", "HashGrid"},
            {"n_levels", 8},
            {"n_features_per_level", 4},
            {"log2_hashmap_size", 16},
            {"base_resolution", 16},
            {"per_level_scale", 2},
            {"fixed_point_pos", false},
    };

    // Direction encoding: SphericalHarmonics matching Python config.
    tcnn::cpp::json dirEncConfig = {
            {"otype", "SphericalHarmonics"},
            {"degree", 4},
    };

    // MLP network matching Python config.
    tcnn::cpp::json mlpConfig = {
            {"otype", "FullyFusedMLP"},
            {"activation", "LeakyReLU"},
            {"output_activation", "None"},
            {"n_neurons", 128},
            {"n_hidden_layers", 4},
    };

    // Create point encoding (3D input).
    pointEncoding_ = tcnn::cpp::create_encoding(3, pointEncConfig, tcnn::cpp::Precision::Fp16);
    if (!pointEncoding_) {
        std::fprintf(stderr, "Failed to create point encoding.\n");
        return;
    }
    pointEncOutDims_ = pointEncoding_->n_output_dims();

    // Create direction encoding (3D input).
    dirEncoding_ = tcnn::cpp::create_encoding(3, dirEncConfig, tcnn::cpp::Precision::Fp16);
    if (!dirEncoding_) {
        std::fprintf(stderr, "Failed to create direction encoding.\n");
        delete pointEncoding_;
        pointEncoding_ = nullptr;
        return;
    }
    dirEncOutDims_ = dirEncoding_->n_output_dims();

    // MLP input = point_enc * 2 (outer + inner) + dir_enc.
    mlpInputDims_ = pointEncOutDims_ * 2 + dirEncOutDims_;
    mlpOutputDims_ = 5;

    // Create MLP network.
    mlpNetwork_ = tcnn::cpp::create_network(mlpInputDims_, mlpOutputDims_, mlpConfig);
    if (!mlpNetwork_) {
        std::fprintf(stderr, "Failed to create MLP network.\n");
        delete dirEncoding_;
        dirEncoding_ = nullptr;
        delete pointEncoding_;
        pointEncoding_ = nullptr;
        return;
    }

    // n_output_dims() returns padded_output_width (e.g. 16 for 5 requested outputs).
    // We must use this for buffer allocation and output stride.
    mlpOutputDims_ = mlpNetwork_->n_output_dims();
    mlpOutputElemSize_ = precisionBytes(mlpNetwork_->output_precision());

    std::printf("Neural architecture: point_enc=%u, dir_enc=%u, mlp_input=%u, mlp_output=%u (requested 5)\n",
                pointEncOutDims_, dirEncOutDims_, mlpInputDims_, mlpOutputDims_);

    // Allocate and initialize parameters for each module.
    pointEncParams_ = allocAndInitParams(pointEncoding_, &pointEncParamsBytes_);
    dirEncParams_ = allocAndInitParams(dirEncoding_, &dirEncParamsBytes_);
    mlpParams_ = allocAndInitParams(mlpNetwork_, &mlpParamsBytes_);

    std::printf("Params: point_enc=%zu bytes, dir_enc=%zu bytes, mlp=%zu bytes, total=%zu bytes\n",
                pointEncParamsBytes_, dirEncParamsBytes_, mlpParamsBytes_,
                pointEncParamsBytes_ + dirEncParamsBytes_ + mlpParamsBytes_);
}

RendererNeural::~RendererNeural() {
    release();
    releaseNetwork();
}

void RendererNeural::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    release();
    width_ = width;
    height_ = height;
    if (width_ > 0 && height_ > 0) {
        checkCuda(cudaMalloc(&devicePixels_, width_ * height_ * sizeof(uchar4)), "cudaMalloc");
    }
}

void RendererNeural::setCameraBasis(const RenderBasis& basis) {
    basis_ = basis;
}

bool RendererNeural::loadWeightsFromFile(const std::string& path) {
    size_t totalBytes = pointEncParamsBytes_ + dirEncParamsBytes_ + mlpParamsBytes_;
    if (!pointEncoding_ || !mlpNetwork_ || totalBytes == 0) {
        std::fprintf(stderr, "Neural network not initialized; cannot load weights.\n");
        return false;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "Failed to open weights file: %s\n", path.c_str());
        return false;
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "Weights file is empty: %s\n", path.c_str());
        return false;
    }
    if (static_cast<size_t>(size) != totalBytes) {
        std::fprintf(stderr,
                     "Weights size mismatch (got %lld bytes, expected %zu = %zu + %zu + %zu).\n",
                     static_cast<long long>(size),
                     totalBytes,
                     pointEncParamsBytes_,
                     dirEncParamsBytes_,
                     mlpParamsBytes_);
        return false;
    }

    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), size)) {
        std::fprintf(stderr, "Failed to read weights file: %s\n", path.c_str());
        return false;
    }

    // Layout: [point_enc_params | dir_enc_params | mlp_params]
    size_t offset = 0;
    if (pointEncParamsBytes_ > 0 && pointEncParams_) {
        checkCuda(cudaMemcpy(pointEncParams_, buffer.data() + offset,
                             pointEncParamsBytes_, cudaMemcpyHostToDevice),
                  "cudaMemcpy point enc params");
        offset += pointEncParamsBytes_;
    }
    if (dirEncParamsBytes_ > 0 && dirEncParams_) {
        checkCuda(cudaMemcpy(dirEncParams_, buffer.data() + offset,
                             dirEncParamsBytes_, cudaMemcpyHostToDevice),
                  "cudaMemcpy dir enc params");
        offset += dirEncParamsBytes_;
    }
    if (mlpParamsBytes_ > 0 && mlpParams_) {
        checkCuda(cudaMemcpy(mlpParams_, buffer.data() + offset,
                             mlpParamsBytes_, cudaMemcpyHostToDevice),
                  "cudaMemcpy mlp params");
    }

    return true;
}

void RendererNeural::render(const Vec3& camPos, std::vector<uchar4>& hostPixels) {
    if (width_ <= 0 || height_ <= 0) {
        return;
    }
    if (!scene_ || !devicePixels_) {
        return;
    }
    if (hostPixels.size() != static_cast<size_t>(width_ * height_)) {
        hostPixels.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_));
    }

    int maxBounces = bounceCount_;
    if (maxBounces < 0) {
        maxBounces = 0;
    }

    Mesh& originalMesh = scene_->originalMesh();
    Mesh& outerShell = scene_->outerShell();
    Mesh& innerShell = scene_->innerShell();

    // Upload all meshes that might be needed.
    originalMesh.uploadToDevice();
    outerShell.uploadToDevice();
    innerShell.uploadToDevice();

    // Select mesh for non-neural rendering (0=original, 1=inner, 2=outer).
    Mesh* classicMesh = &originalMesh;
    if (classicMeshIndex_ == 1 && innerShell.triangleCount() > 0) {
        classicMesh = &innerShell;
    } else if (classicMeshIndex_ == 2 && outerShell.triangleCount() > 0) {
        classicMesh = &outerShell;
    }

    EnvironmentMap& environment = scene_->environment();
    environment.uploadToDevice();
    MeshDeviceView classicView = classicMesh->deviceView();
    MeshDeviceView outerView = outerShell.deviceView();
    MeshDeviceView innerView = innerShell.deviceView();
    EnvironmentDeviceView envView = environment.deviceView();

    size_t pixelCount = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    int samplesPerPixel = samplesPerPixel_;
    if (samplesPerPixel <= 0) {
        samplesPerPixel = 1;
    }
    size_t elementCount = pixelCount * static_cast<size_t>(samplesPerPixel);
    size_t paddedCount = elementCount;
    if (mlpNetwork_) {
        size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
        paddedCount = roundUp(elementCount, granularity);
    }
    if (!ensureNetworkBuffers(paddedCount)) {
        return;
    }
    if (!ensureAccumBuffer(pixelCount)) {
        return;
    }

    // Outer shell bounds for normalization.
    Vec3 outerMin = outerShell.boundsMin();
    Vec3 outerMax = outerShell.boundsMax();
    Vec3 outerExtent = outerMax - outerMin;
    Vec3 outerInvExtent(
            outerExtent.x != 0.0f ? 1.0f / outerExtent.x : 0.0f,
            outerExtent.y != 0.0f ? 1.0f / outerExtent.y : 0.0f,
            outerExtent.z != 0.0f ? 1.0f / outerExtent.z : 0.0f);

    // Camera change detection.
    bool cameraMoved = !hasLastCamera_;
    const float kEps = 1e-4f;
    if (!cameraMoved) {
        if (fabsf(camPos.x - lastCamPos_.x) > kEps ||
            fabsf(camPos.y - lastCamPos_.y) > kEps ||
            fabsf(camPos.z - lastCamPos_.z) > kEps) {
            cameraMoved = true;
        }
        if (fabsf(basis_.forward.x - lastBasis_.forward.x) > kEps ||
            fabsf(basis_.forward.y - lastBasis_.forward.y) > kEps ||
            fabsf(basis_.forward.z - lastBasis_.forward.z) > kEps ||
            fabsf(basis_.right.x - lastBasis_.right.x) > kEps ||
            fabsf(basis_.right.y - lastBasis_.right.y) > kEps ||
            fabsf(basis_.right.z - lastBasis_.right.z) > kEps ||
            fabsf(basis_.up.x - lastBasis_.up.x) > kEps ||
            fabsf(basis_.up.y - lastBasis_.up.y) > kEps ||
            fabsf(basis_.up.z - lastBasis_.up.z) > kEps ||
            fabsf(basis_.fovY - lastFovY_) > kEps) {
            cameraMoved = true;
        }
    }
    if (cameraMoved || useNeuralQuery_ != lastUseNeuralQuery_ || lambertView_ != lastLambertView_ ||
        maxBounces != lastBounceCount_ || samplesPerPixel != lastSamplesPerPixel_ ||
        classicMeshIndex_ != lastClassicMeshIndex_) {
        resetAccum();
    }
    lastUseNeuralQuery_ = useNeuralQuery_;
    lastLambertView_ = lambertView_;
    lastBounceCount_ = maxBounces;
    lastSamplesPerPixel_ = samplesPerPixel;
    lastClassicMeshIndex_ = classicMeshIndex_;
    lastCamPos_ = camPos;
    lastBasis_ = basis_;
    lastFovY_ = basis_.fovY;
    hasLastCamera_ = true;

    const Material& material = scene_->material();

    RenderParams params;
    params.camPos = camPos;
    params.camForward = basis_.forward;
    params.camRight = basis_.right;
    params.camUp = basis_.up;
    params.lightDir = lightDir_;
    params.outerShellMin = outerMin;
    params.outerShellInvExtent = outerInvExtent;
    params.materialColor = material.color;
    params.fovY = basis_.fovY;
    params.materialReflectiveness = material.reflectiveness;
    params.maxRadiance = 20.0f;
    params.maxBounces = maxBounces;
    params.width = width_;
    params.height = height_;
    params.pixelCount = static_cast<int>(pixelCount);
    params.samplesPerPixel = samplesPerPixel;
    params.sampleOffset = accumSampleCount_;

    dim3 block(8, 8);
    dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

    bool neuralReady = useNeuralQuery_ && pointEncoding_ && mlpNetwork_ &&
                       outerShell.triangleCount() > 0;
    if (neuralReady) {
        // --- Neural shell-based rendering pipeline ---

        // 1. Trace outer and inner shells.
        traceShellsKernel<<<grid, block>>>(
                outerHitPositions_,
                innerHitPositions_,
                rayDirections_,
                outerHitFlags_,
                params,
                outerView,
                innerView);
        checkCuda(cudaGetLastError(), "traceShellsKernel launch");

        // Initialize hit buffers.
        int elementCountInt = static_cast<int>(elementCount);
        checkCuda(cudaMemcpy(hitFlags_, outerHitFlags_, elementCount * sizeof(int),
                             cudaMemcpyDeviceToDevice),
                  "cudaMemcpy hitFlags from outerHitFlags");
        checkCuda(cudaMemset(hitPositions_, 0, elementCount * 3 * sizeof(float)),
                  "cudaMemset hitPositions");
        checkCuda(cudaMemset(hitNormals_, 0, elementCount * 3 * sizeof(float)),
                  "cudaMemset hitNormals");
        checkCuda(cudaMemset(hitColors_, 0, elementCount * 3 * sizeof(float)),
                  "cudaMemset hitColors");

        // 2. Compact outer shell hits.
        const int compactBlock = 256;
        int compactGrid = static_cast<int>((elementCount + compactBlock - 1) / compactBlock);
        checkCuda(cudaMemset(hitCount_, 0, sizeof(int)), "cudaMemset hitCount");
        compactInputsKernel<<<compactGrid, compactBlock>>>(
                outerHitFlags_,
                elementCountInt,
                hitIndices_,
                hitCount_);
        checkCuda(cudaGetLastError(), "compactInputsKernel launch");

        int hitCount = 0;
        checkCuda(cudaMemcpy(&hitCount, hitCount_, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy hitCount");

        if (hitCount > 0) {
            size_t hitCountSize = static_cast<size_t>(hitCount);
            size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
            size_t paddedHitCount = roundUp(hitCountSize, granularity);

            // 3. Build 3 separate encoding input buffers.
            const int buildBlock = 256;
            int buildGrid = (hitCount + buildBlock - 1) / buildBlock;
            buildNeuralInputsKernel<<<buildGrid, buildBlock>>>(
                    outerHitPositions_,
                    innerHitPositions_,
                    rayDirections_,
                    hitIndices_,
                    hitCount,
                    outerMin,
                    outerInvExtent,
                    compactedOuterPos_,
                    compactedInnerPos_,
                    compactedDirs_);
            checkCuda(cudaGetLastError(), "buildNeuralInputsKernel launch");

            // Zero-pad tails for TCNN alignment.
            if (paddedHitCount > hitCountSize) {
                size_t tail = paddedHitCount - hitCountSize;
                checkCuda(cudaMemset(
                        compactedOuterPos_ + hitCountSize * 3,
                        0, tail * 3 * sizeof(float)),
                        "cudaMemset outer pos tail");
                checkCuda(cudaMemset(
                        compactedInnerPos_ + hitCountSize * 3,
                        0, tail * 3 * sizeof(float)),
                        "cudaMemset inner pos tail");
                checkCuda(cudaMemset(
                        compactedDirs_ + hitCountSize * 3,
                        0, tail * 3 * sizeof(float)),
                        "cudaMemset dirs tail");
            }

            uint32_t batchSize = static_cast<uint32_t>(paddedHitCount);

            // 4a. Point encoding: outer positions.
            {
                tcnn::cpp::Context ctx = pointEncoding_->forward(
                    0, batchSize,
                    compactedOuterPos_,
                    pointEncOutput1_,
                    pointEncParams_,
                    false);
                (void)ctx;
            }

            // 4b. Point encoding: inner positions (same encoder, same params).
            {
                tcnn::cpp::Context ctx = pointEncoding_->forward(
                    0, batchSize,
                    compactedInnerPos_,
                    pointEncOutput2_,
                    pointEncParams_,
                    false);
                (void)ctx;
            }

            // 4c. Direction encoding.
            {
                tcnn::cpp::Context ctx = dirEncoding_->forward(
                    0, batchSize,
                    compactedDirs_,
                    dirEncOutput_,
                    dirEncParams_,
                    false);
                (void)ctx;
            }

            // 5. Concatenate FP16 encoding outputs into FP32 MLP input.
            int concatGrid = (hitCount + buildBlock - 1) / buildBlock;
            concatenateEncodingsKernel<<<concatGrid, buildBlock>>>(
                    static_cast<const __half*>(pointEncOutput1_),
                    pointEncOutDims_,
                    static_cast<const __half*>(pointEncOutput2_),
                    pointEncOutDims_,
                    static_cast<const __half*>(dirEncOutput_),
                    dirEncOutDims_,
                    mlpInput_,
                    mlpInputDims_,
                    hitCount);
            checkCuda(cudaGetLastError(), "concatenateEncodingsKernel launch");

            // Zero-pad MLP input tail.
            if (paddedHitCount > hitCountSize) {
                size_t tail = paddedHitCount - hitCountSize;
                checkCuda(cudaMemset(
                        mlpInput_ + hitCountSize * mlpInputDims_,
                        0, tail * mlpInputDims_ * sizeof(float)),
                        "cudaMemset mlp input tail");
            }

            // 6. MLP forward pass.
            {
                tcnn::cpp::Context ctx = mlpNetwork_->forward(
                    0, batchSize,
                    mlpInput_,
                    outputs_,
                    mlpParams_,
                    false);
                (void)ctx;
            }

            // 7. Apply neural outputs.
            int applyGrid = (hitCount + buildBlock - 1) / buildBlock;
            applyNeuralOutputKernel<<<applyGrid, buildBlock>>>(
                    static_cast<const __half*>(outputs_),
                    hitIndices_,
                    hitCount,
                    static_cast<int>(mlpOutputDims_),
                    outerHitPositions_,
                    rayDirections_,
                    hitPositions_,
                    hitNormals_,
                    hitColors_,
                    hitFlags_,
                    params.materialColor);
            checkCuda(cudaGetLastError(), "applyNeuralOutputKernel launch");
        }

        // 8. Path trace from neural hits using original mesh for bounces.
        if (!lambertView_) {
            initNeuralPathKernel<<<grid, block>>>(
                    pathThroughput_,
                    pathRadiance_,
                    pathActive_,
                    hitFlags_,
                    hitNormals_,
                    hitColors_,
                    params,
                    envView);
            checkCuda(cudaGetLastError(), "initNeuralPathKernel launch");

            if (maxBounces > 0) {
                renderBounceKernel<<<grid, block>>>(
                        hitPositions_,
                        hitNormals_,
                        hitFlags_,
                        pathActive_,
                        params,
                        classicView,
                        bouncePositions_,
                        bounceNormals_,
                        bounceColors_,
                        bounceHitFlags_,
                        bounceDirs_);
                checkCuda(cudaGetLastError(), "renderBounceKernel launch");

                integrateNeuralBounceKernel<<<grid, block>>>(
                        pathThroughput_,
                        pathRadiance_,
                        pathActive_,
                        bounceHitFlags_,
                        bounceColors_,
                        bounceDirs_,
                        1,
                        params,
                        envView);
                checkCuda(cudaGetLastError(), "integrateNeuralBounceKernel launch");

                float* inPositions = bouncePositions_;
                float* inNormals = bounceNormals_;
                int* inFlags = bounceHitFlags_;
                float* inDirs = bounceDirs_;
                float* outPositions = bounce2Positions_;
                float* outNormals = bounce2Normals_;
                float* outColors = bounce2Colors_;
                int* outFlags = bounce2HitFlags_;
                float* outDirs = bounce2Dirs_;

                for (int bounce = 2; bounce <= maxBounces; ++bounce) {
                    renderBounceFromStateKernel<<<grid, block>>>(
                            inPositions,
                            inNormals,
                            inFlags,
                            inDirs,
                            pathActive_,
                            static_cast<uint32_t>(bounce),
                            params,
                            classicView,
                            outPositions,
                            outNormals,
                            outColors,
                            outFlags,
                            outDirs);
                    checkCuda(cudaGetLastError(), "renderBounceFromStateKernel launch");

                    integrateNeuralBounceKernel<<<grid, block>>>(
                            pathThroughput_,
                            pathRadiance_,
                            pathActive_,
                            outFlags,
                            outColors,
                            outDirs,
                            bounce,
                            params,
                            envView);
                    checkCuda(cudaGetLastError(), "integrateNeuralBounceKernel launch");

                    std::swap(inPositions, outPositions);
                    std::swap(inNormals, outNormals);
                    std::swap(inFlags, outFlags);
                    std::swap(inDirs, outDirs);
                }
            }

            finalizeNeuralPathKernel<<<grid, block>>>(
                    devicePixels_,
                    accum_,
                    pathRadiance_,
                    params);
            checkCuda(cudaGetLastError(), "finalizeNeuralPathKernel launch");
            accumSampleCount_ += static_cast<uint32_t>(samplesPerPixel);
        } else {
            lambertKernel<<<grid, block>>>(
                    devicePixels_,
                    hitNormals_,
                    hitColors_,
                    hitFlags_,
                    params,
                    envView);
            checkCuda(cudaGetLastError(), "lambertKernel launch");
            accumSampleCount_ = 0;
        }
    } else {
        // --- Standard original mesh path tracing ---
        renderOriginalKernel<<<grid, block>>>(
                hitPositions_,
                hitNormals_,
                hitColors_,
                hitFlags_,
                params,
                classicView);
        checkCuda(cudaGetLastError(), "renderOriginalKernel launch");

        if (lambertView_) {
            lambertKernel<<<grid, block>>>(
                    devicePixels_,
                    hitNormals_,
                    hitColors_,
                    hitFlags_,
                    params,
                    envView);
            checkCuda(cudaGetLastError(), "lambertKernel launch");
            accumSampleCount_ = 0;
        } else {
            pathTraceKernel<<<grid, block>>>(
                    devicePixels_,
                    accum_,
                    hitPositions_,
                    hitNormals_,
                    hitColors_,
                    hitFlags_,
                    params,
                    classicView,
                    envView);
            checkCuda(cudaGetLastError(), "pathTraceKernel launch");
            accumSampleCount_ += static_cast<uint32_t>(samplesPerPixel);
        }
    }

    checkCuda(cudaMemcpy(
            hostPixels.data(),
            devicePixels_,
            hostPixels.size() * sizeof(uchar4),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy");
}

// ===========================================================================
// Memory management.
// ===========================================================================

void RendererNeural::release() {
    auto freePtr = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    freePtr(devicePixels_);
    freePtr(accum_);
    freePtr(compactedOuterPos_);
    freePtr(compactedInnerPos_);
    freePtr(compactedDirs_);
    freePtr(pointEncOutput1_);
    freePtr(pointEncOutput2_);
    freePtr(dirEncOutput_);
    freePtr(mlpInput_);
    freePtr(hitIndices_);
    freePtr(hitCount_);
    freePtr(outerHitPositions_);
    freePtr(innerHitPositions_);
    freePtr(rayDirections_);
    freePtr(outerHitFlags_);
    freePtr(hitPositions_);
    freePtr(hitNormals_);
    freePtr(hitColors_);
    freePtr(hitFlags_);
    freePtr(outputs_);
    freePtr(bouncePositions_);
    freePtr(bounceNormals_);
    freePtr(bounceDirs_);
    freePtr(bounceColors_);
    freePtr(bounceHitFlags_);
    freePtr(bounce2Positions_);
    freePtr(bounce2Normals_);
    freePtr(bounce2Dirs_);
    freePtr(bounce2Colors_);
    freePtr(bounce2HitFlags_);
    freePtr(pathThroughput_);
    freePtr(pathRadiance_);
    freePtr(pathActive_);
    bufferElements_ = 0;
    accumPixels_ = 0;
    accumSampleCount_ = 0;
    hasLastCamera_ = false;
}

void RendererNeural::releaseNetwork() {
    auto freePtr = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    freePtr(pointEncParams_);
    freePtr(dirEncParams_);
    freePtr(mlpParams_);
    delete pointEncoding_;
    pointEncoding_ = nullptr;
    delete dirEncoding_;
    dirEncoding_ = nullptr;
    delete mlpNetwork_;
    mlpNetwork_ = nullptr;
    pointEncParamsBytes_ = 0;
    dirEncParamsBytes_ = 0;
    mlpParamsBytes_ = 0;
    pointEncOutDims_ = 0;
    dirEncOutDims_ = 0;
    mlpInputDims_ = 0;
    mlpOutputDims_ = 0;
    mlpOutputElemSize_ = 0;
}

bool RendererNeural::ensureNetworkBuffers(size_t elementCount) {
    if (elementCount == 0) {
        return false;
    }
    if (elementCount <= bufferElements_ &&
            hitPositions_ && hitNormals_ && hitColors_ && hitFlags_ &&
            outerHitPositions_ && innerHitPositions_ && rayDirections_ && outerHitFlags_ &&
            compactedOuterPos_ && compactedInnerPos_ && compactedDirs_ &&
            hitIndices_ && hitCount_ &&
            bouncePositions_ && bounceNormals_ && bounceDirs_ && bounceColors_ && bounceHitFlags_ &&
            bounce2Positions_ && bounce2Normals_ && bounce2Dirs_ && bounce2Colors_ && bounce2HitFlags_ &&
            pathThroughput_ && pathRadiance_ && pathActive_) {
        return true;
    }

    // Free old buffers.
    auto freePtr = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    freePtr(compactedOuterPos_);
    freePtr(compactedInnerPos_);
    freePtr(compactedDirs_);
    freePtr(pointEncOutput1_);
    freePtr(pointEncOutput2_);
    freePtr(dirEncOutput_);
    freePtr(mlpInput_);
    freePtr(hitIndices_);
    freePtr(hitCount_);
    freePtr(outerHitPositions_);
    freePtr(innerHitPositions_);
    freePtr(rayDirections_);
    freePtr(outerHitFlags_);
    freePtr(hitPositions_);
    freePtr(hitNormals_);
    freePtr(hitColors_);
    freePtr(hitFlags_);
    freePtr(outputs_);
    freePtr(bouncePositions_);
    freePtr(bounceNormals_);
    freePtr(bounceDirs_);
    freePtr(bounceColors_);
    freePtr(bounceHitFlags_);
    freePtr(bounce2Positions_);
    freePtr(bounce2Normals_);
    freePtr(bounce2Dirs_);
    freePtr(bounce2Colors_);
    freePtr(bounce2HitFlags_);
    freePtr(pathThroughput_);
    freePtr(pathRadiance_);
    freePtr(pathActive_);

    size_t vec3Bytes = elementCount * 3 * sizeof(float);
    size_t intBytes = elementCount * sizeof(int);

    // Shell tracing buffers.
    checkCuda(cudaMalloc(&outerHitPositions_, vec3Bytes), "cudaMalloc outerHitPositions");
    checkCuda(cudaMalloc(&innerHitPositions_, vec3Bytes), "cudaMalloc innerHitPositions");
    checkCuda(cudaMalloc(&rayDirections_, vec3Bytes), "cudaMalloc rayDirections");
    checkCuda(cudaMalloc(&outerHitFlags_, intBytes), "cudaMalloc outerHitFlags");

    // Encoding input buffers (3 floats each per element).
    checkCuda(cudaMalloc(&compactedOuterPos_, vec3Bytes), "cudaMalloc compactedOuterPos");
    checkCuda(cudaMalloc(&compactedInnerPos_, vec3Bytes), "cudaMalloc compactedInnerPos");
    checkCuda(cudaMalloc(&compactedDirs_, vec3Bytes), "cudaMalloc compactedDirs");

    // Encoding output buffers (FP16).
    if (pointEncOutDims_ > 0) {
        size_t encOutBytes = elementCount * pointEncOutDims_ * sizeof(__half);
        checkCuda(cudaMalloc(&pointEncOutput1_, encOutBytes), "cudaMalloc pointEncOutput1");
        checkCuda(cudaMalloc(&pointEncOutput2_, encOutBytes), "cudaMalloc pointEncOutput2");
    }
    if (dirEncOutDims_ > 0) {
        size_t dirEncOutBytes = elementCount * dirEncOutDims_ * sizeof(__half);
        checkCuda(cudaMalloc(&dirEncOutput_, dirEncOutBytes), "cudaMalloc dirEncOutput");
    }

    // MLP input (FP32 concatenated encodings).
    if (mlpInputDims_ > 0) {
        checkCuda(cudaMalloc(&mlpInput_, elementCount * mlpInputDims_ * sizeof(float)),
                  "cudaMalloc mlpInput");
    }

    // Compaction buffers.
    checkCuda(cudaMalloc(&hitIndices_, intBytes), "cudaMalloc hitIndices");
    checkCuda(cudaMalloc(&hitCount_, sizeof(int)), "cudaMalloc hitCount");

    // MLP output (FP16).
    if (mlpOutputDims_ > 0 && mlpOutputElemSize_ > 0) {
        size_t outputBytes = elementCount * mlpOutputDims_ * mlpOutputElemSize_;
        checkCuda(cudaMalloc(&outputs_, outputBytes), "cudaMalloc outputs");
    }

    // Primary hit buffers.
    checkCuda(cudaMalloc(&hitPositions_, vec3Bytes), "cudaMalloc hitPositions");
    checkCuda(cudaMalloc(&hitNormals_, vec3Bytes), "cudaMalloc hitNormals");
    checkCuda(cudaMalloc(&hitColors_, vec3Bytes), "cudaMalloc hitColors");
    checkCuda(cudaMalloc(&hitFlags_, intBytes), "cudaMalloc hitFlags");

    // Bounce buffers.
    checkCuda(cudaMalloc(&bouncePositions_, vec3Bytes), "cudaMalloc bouncePositions");
    checkCuda(cudaMalloc(&bounceNormals_, vec3Bytes), "cudaMalloc bounceNormals");
    checkCuda(cudaMalloc(&bounceColors_, vec3Bytes), "cudaMalloc bounceColors");
    checkCuda(cudaMalloc(&bounceDirs_, vec3Bytes), "cudaMalloc bounceDirs");
    checkCuda(cudaMalloc(&bounceHitFlags_, intBytes), "cudaMalloc bounceHitFlags");

    checkCuda(cudaMalloc(&bounce2Positions_, vec3Bytes), "cudaMalloc bounce2Positions");
    checkCuda(cudaMalloc(&bounce2Normals_, vec3Bytes), "cudaMalloc bounce2Normals");
    checkCuda(cudaMalloc(&bounce2Colors_, vec3Bytes), "cudaMalloc bounce2Colors");
    checkCuda(cudaMalloc(&bounce2Dirs_, vec3Bytes), "cudaMalloc bounce2Dirs");
    checkCuda(cudaMalloc(&bounce2HitFlags_, intBytes), "cudaMalloc bounce2HitFlags");

    // Path state.
    checkCuda(cudaMalloc(&pathThroughput_, elementCount * sizeof(Vec3)), "cudaMalloc pathThroughput");
    checkCuda(cudaMalloc(&pathRadiance_, elementCount * sizeof(Vec3)), "cudaMalloc pathRadiance");
    checkCuda(cudaMalloc(&pathActive_, intBytes), "cudaMalloc pathActive");

    bufferElements_ = elementCount;
    return true;
}

bool RendererNeural::ensureAccumBuffer(size_t pixelCount) {
    if (pixelCount == 0) {
        return false;
    }
    if (accum_ && accumPixels_ == pixelCount) {
        return true;
    }
    if (accum_) {
        cudaFree(accum_);
        accum_ = nullptr;
    }
    checkCuda(cudaMalloc(&accum_, pixelCount * sizeof(Vec3)), "cudaMalloc accum");
    checkCuda(cudaMemset(accum_, 0, pixelCount * sizeof(Vec3)), "cudaMemset accum");
    accumPixels_ = pixelCount;
    accumSampleCount_ = 0;
    return true;
}

void RendererNeural::resetAccum() {
    if (accum_ && accumPixels_ > 0) {
        checkCuda(cudaMemset(accum_, 0, accumPixels_ * sizeof(Vec3)), "cudaMemset accum");
    }
    accumSampleCount_ = 0;
}
