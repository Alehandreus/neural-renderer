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
#include "point_query.cuh"

namespace {

struct RenderParams {
    Vec3 camPos;
    Vec3 camForward;
    Vec3 camRight;
    Vec3 camUp;
    Vec3 lightDir;
    Vec3 meshMin;
    Vec3 meshInvExtent;
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
                                     float* outTNear);

__device__ inline bool traceMesh(const Ray& ray, MeshDeviceView mesh, HitInfo* outHit) {
    if (mesh.nodeCount <= 0 || mesh.triangleCount <= 0) {
        return false;
    }

    HitInfo bestHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
    float closestT = 1e30f;
    Vec3 invDir(
        1.0f / ray.direction.x,
        1.0f / ray.direction.y,
        1.0f / ray.direction.z);

    // Use a larger stack to avoid missing nodes in deep BVHs.
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
                if (dot(tri.normal, ray.direction) >= 0.0f) {
                    continue;
                }
                HitInfo hit = intersectTriangle(ray, tri);
                if (hit.hit && hit.distance < closestT) {
                    closestT = hit.distance;
                    bestHit = hit;
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

__device__ inline float atomicMaxFloat(float* address, float value) {
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;
    int assumed = old;
    if (__int_as_float(old) >= value) {
        return __int_as_float(old);
    }
    do {
        assumed = old;
        old = atomicCAS(addressAsInt, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void renderNeuralKernel(float* neuralInputs,
                                   float* hitPositions,
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

    int idx = y * params.width + x;
    int sampleIdx = idx;
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    float reflectiveness = clampf(params.materialReflectiveness, 0.0f, 1.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = idx + s * params.pixelCount;
        uint32_t rng = initRng(idx, params.sampleOffset, s);
        Ray ray = generatePrimaryRay(x, y, params, rng);

        HitInfo bestHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
        bool hit = traceMesh(ray, mesh, &bestHit);
        if (hit) {
            Vec3 hitPos = ray.at(bestHit.distance);
            Vec3 local = hitPos - params.meshMin;
            Vec3 normalized(
                    local.x * params.meshInvExtent.x,
                    local.y * params.meshInvExtent.y,
                    local.z * params.meshInvExtent.z);
            int base = sampleIdx * 3;
            neuralInputs[base + 0] = normalized.x;
            neuralInputs[base + 1] = normalized.y;
            neuralInputs[base + 2] = normalized.z;
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
            int base = sampleIdx * 3;
            neuralInputs[base + 0] = 0.0f;
            neuralInputs[base + 1] = 0.0f;
            neuralInputs[base + 2] = 0.0f;
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

__global__ void renderBounceKernel(const float* hitPositions,
                                   const float* hitNormals,
                                   const int* hitFlags,
                                   RenderParams params,
                                   MeshDeviceView mesh,
                                   float* bounceInputs,
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
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    float reflectiveness = clampf(params.materialReflectiveness, 0.0f, 1.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (!hitFlags[sampleIdx]) {
            bounceInputs[base + 0] = 0.0f;
            bounceInputs[base + 1] = 0.0f;
            bounceInputs[base + 2] = 0.0f;
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

        Vec3 hitPos(
                hitPositions[base + 0],
                hitPositions[base + 1],
                hitPositions[base + 2]);
        Vec3 normal(
                hitNormals[base + 0],
                hitNormals[base + 1],
                hitNormals[base + 2]);

        float nlen = length(normal);
        if (nlen > 0.0f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        if (dot(normal, primaryRay.direction) > 0.0f) {
            normal = -normal;
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
        HitInfo bounceHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
        bool hit = traceMesh(bounceRay, mesh, &bounceHit);
        if (hit) {
            Vec3 bouncePos = bounceRay.at(bounceHit.distance);
            Vec3 local = bouncePos - params.meshMin;
            Vec3 normalized(
                    local.x * params.meshInvExtent.x,
                    local.y * params.meshInvExtent.y,
                    local.z * params.meshInvExtent.z);
            bounceInputs[base + 0] = normalized.x;
            bounceInputs[base + 1] = normalized.y;
            bounceInputs[base + 2] = normalized.z;
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
            bounceInputs[base + 0] = 0.0f;
            bounceInputs[base + 1] = 0.0f;
            bounceInputs[base + 2] = 0.0f;
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
                                            uint32_t bounceIndex,
                                            RenderParams params,
                                            MeshDeviceView mesh,
                                            float* outInputs,
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
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    float reflectiveness = clampf(params.materialReflectiveness, 0.0f, 1.0f);
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (!inFlags[sampleIdx]) {
            outInputs[base + 0] = 0.0f;
            outInputs[base + 1] = 0.0f;
            outInputs[base + 2] = 0.0f;
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

        Vec3 hitPos(
                inPositions[base + 0],
                inPositions[base + 1],
                inPositions[base + 2]);
        Vec3 normal(
                inNormals[base + 0],
                inNormals[base + 1],
                inNormals[base + 2]);
        Vec3 incoming(
                inDirs[base + 0],
                inDirs[base + 1],
                inDirs[base + 2]);

        float nlen = length(normal);
        if (nlen > 0.0f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        if (dot(normal, incoming) > 0.0f) {
            normal = -normal;
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
        HitInfo bounceHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
        bool hit = traceMesh(bounceRay, mesh, &bounceHit);
        if (hit) {
            Vec3 bouncePos = bounceRay.at(bounceHit.distance);
            Vec3 local = bouncePos - params.meshMin;
            Vec3 normalized(
                    local.x * params.meshInvExtent.x,
                    local.y * params.meshInvExtent.y,
                    local.z * params.meshInvExtent.z);
            outInputs[base + 0] = normalized.x;
            outInputs[base + 1] = normalized.y;
            outInputs[base + 2] = normalized.z;
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
            outInputs[base + 0] = 0.0f;
            outInputs[base + 1] = 0.0f;
            outInputs[base + 2] = 0.0f;
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

__global__ void fillOnesKernel(__half* data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    data[idx] = __float2half(1.0f);
}

__global__ void compactInputsKernel(const float* inputs,
                                    const int* hitFlags,
                                    int count,
                                    float* compactedInputs,
                                    int* hitIndices,
                                    int* hitCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    if (!hitFlags[idx]) {
        return;
    }
    int writeIdx = atomicAdd(hitCount, 1);
    int inputBase = idx * 3;
    int compactBase = writeIdx * 3;
    compactedInputs[compactBase + 0] = inputs[inputBase + 0];
    compactedInputs[compactBase + 1] = inputs[inputBase + 1];
    compactedInputs[compactBase + 2] = inputs[inputBase + 2];
    if (hitIndices) {
        hitIndices[writeIdx] = idx;
    }
}

__global__ void applyNetworkDeltaKernel(const float* compactedInputs,
                                        const __half* outputs,
                                        const int* hitIndices,
                                        int hitCount,
                                        Vec3 meshMin,
                                        Vec3 meshExtent,
                                        int outputStride,
                                        float* hitPositions,
                                        float* normals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int inputBase = idx * 3;
    Vec3 xNorm(
            compactedInputs[inputBase + 0],
            compactedInputs[inputBase + 1],
            compactedInputs[inputBase + 2]);

    int outputBase = idx * outputStride;
    Vec3 deltaNorm(
            __half2float(outputs[outputBase + 0]),
            __half2float(outputs[outputBase + 1]),
            __half2float(outputs[outputBase + 2]));
    float deltaLen = length(deltaNorm);
    Vec3 deltaUnit = deltaLen > 1e-6f ? (deltaNorm / deltaLen) : Vec3(0.0f, 1.0f, 0.0f);

    Vec3 xUpdated = xNorm + deltaNorm;
    Vec3 xWorld(
            xUpdated.x * meshExtent.x + meshMin.x,
            xUpdated.y * meshExtent.y + meshMin.y,
            xUpdated.z * meshExtent.z + meshMin.z);

    int pixelIdx = hitIndices[idx];
    int fullBase = pixelIdx * 3;
    hitPositions[fullBase + 0] = xWorld.x;
    hitPositions[fullBase + 1] = xWorld.y;
    hitPositions[fullBase + 2] = xWorld.z;
    normals[fullBase + 0] = -deltaUnit.x;
    normals[fullBase + 1] = -deltaUnit.y;
    normals[fullBase + 2] = -deltaUnit.z;
}

__global__ void projectInputsToMeshKernel(float* compactedInputs,
                                          int hitCount,
                                          Vec3 meshMin,
                                          Vec3 meshExtent,
                                          Vec3 meshInvExtent,
                                          MeshDeviceView mesh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int base = idx * 3;
    Vec3 xNorm(
            compactedInputs[base + 0],
            compactedInputs[base + 1],
            compactedInputs[base + 2]);
    Vec3 xWorld(
            xNorm.x * meshExtent.x + meshMin.x,
            xNorm.y * meshExtent.y + meshMin.y,
            xNorm.z * meshExtent.z + meshMin.z);
    Vec3 closest = closestPointOnMesh(xWorld, mesh);
    Vec3 local = closest - meshMin;
    Vec3 projected(
            local.x * meshInvExtent.x,
            local.y * meshInvExtent.y,
            local.z * meshInvExtent.z);
    compactedInputs[base + 0] = projected.x;
    compactedInputs[base + 1] = projected.y;
    compactedInputs[base + 2] = projected.z;
}

__global__ void computeLossGradKernel(const float* compactedInputs,
                                      const __half* outputs,
                                      const int* hitIndices,
                                      int hitCount,
                                      RenderParams params,
                                      Vec3 meshMin,
                                      Vec3 meshExtent,
                                      int outputStride,
                                      __half* dL_doutput) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int inputBase = idx * 3;
    Vec3 xNorm(
            compactedInputs[inputBase + 0],
            compactedInputs[inputBase + 1],
            compactedInputs[inputBase + 2]);

    int outputBase = idx * outputStride;
    Vec3 deltaNorm(
            __half2float(outputs[outputBase + 0]),
            __half2float(outputs[outputBase + 1]),
            __half2float(outputs[outputBase + 2]));

    Vec3 xUpdated = xNorm + deltaNorm;
    Vec3 hitPos(
            xUpdated.x * meshExtent.x + meshMin.x,
            xUpdated.y * meshExtent.y + meshMin.y,
            xUpdated.z * meshExtent.z + meshMin.z);

    int sampleIdx = hitIndices[idx];
    int pixelIdx = params.pixelCount > 0 ? (sampleIdx % params.pixelCount) : sampleIdx;
    int sampleInPixel = params.pixelCount > 0 ? (sampleIdx / params.pixelCount) : 0;
    int px = pixelIdx - (pixelIdx / params.width) * params.width;
    int py = pixelIdx / params.width;

    uint32_t rng = initRng(pixelIdx, params.sampleOffset, sampleInPixel);
    Ray primaryRay = generatePrimaryRay(px, py, params, rng);
    Vec3 dir = primaryRay.direction;

    Vec3 toPoint = hitPos - params.camPos;
    float t = dot(toPoint, dir);
    Vec3 closest = params.camPos + dir * t;
    Vec3 residual = hitPos - closest;
    float dist = length(residual);
    Vec3 grad(0.0f, 0.0f, 0.0f);
    if (dist > 1e-6f) {
        grad = residual / dist;
    }

    Vec3 gradDelta(
            grad.x * meshExtent.x,
            grad.y * meshExtent.y,
            grad.z * meshExtent.z);

    dL_doutput[outputBase + 0] = __float2half(gradDelta.x);
    dL_doutput[outputBase + 1] = __float2half(gradDelta.y);
    dL_doutput[outputBase + 2] = __float2half(gradDelta.z);
}

__global__ void addDirectGradKernel(const float* compactedInputs,
                                    const __half* outputs,
                                    const int* hitIndices,
                                    int hitCount,
                                    RenderParams params,
                                    Vec3 meshMin,
                                    Vec3 meshExtent,
                                    int outputStride,
                                    float* dL_dinput) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int inputBase = idx * 3;
    Vec3 xNorm(
            compactedInputs[inputBase + 0],
            compactedInputs[inputBase + 1],
            compactedInputs[inputBase + 2]);

    int outputBase = idx * outputStride;
    Vec3 deltaNorm(
            __half2float(outputs[outputBase + 0]),
            __half2float(outputs[outputBase + 1]),
            __half2float(outputs[outputBase + 2]));

    Vec3 xUpdated = xNorm + deltaNorm;
    Vec3 hitPos(
            xUpdated.x * meshExtent.x + meshMin.x,
            xUpdated.y * meshExtent.y + meshMin.y,
            xUpdated.z * meshExtent.z + meshMin.z);

    int sampleIdx = hitIndices[idx];
    int pixelIdx = params.pixelCount > 0 ? (sampleIdx % params.pixelCount) : sampleIdx;
    int sampleInPixel = params.pixelCount > 0 ? (sampleIdx / params.pixelCount) : 0;
    int px = pixelIdx - (pixelIdx / params.width) * params.width;
    int py = pixelIdx / params.width;

    uint32_t rng = initRng(pixelIdx, params.sampleOffset, sampleInPixel);
    Ray primaryRay = generatePrimaryRay(px, py, params, rng);
    Vec3 dir = primaryRay.direction;

    Vec3 toPoint = hitPos - params.camPos;
    float t = dot(toPoint, dir);
    Vec3 closest = params.camPos + dir * t;
    Vec3 residual = hitPos - closest;
    float dist = length(residual);
    Vec3 grad(0.0f, 0.0f, 0.0f);
    if (dist > 1e-6f) {
        grad = residual / dist;
    }

    Vec3 gradDelta(
            grad.x * meshExtent.x,
            grad.y * meshExtent.y,
            grad.z * meshExtent.z);

    dL_dinput[inputBase + 0] += gradDelta.x;
    dL_dinput[inputBase + 1] += gradDelta.y;
    dL_dinput[inputBase + 2] += gradDelta.z;
}

__global__ void sgdInputsKernel(float* inputs,
                                const float* grads,
                                int count,
                                float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * 3;
    if (idx >= total) {
        return;
    }
    inputs[idx] -= lr * grads[idx];
}

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
                normal = -normal;
            }

            Vec3 bounceDir;
            float choose = rand01(rng);
            if (choose < reflectiveness) {
                bounceDir = normalize(reflectDir(ray.direction, normal));
            } else {
                bounceDir = sampleHemisphereCosine(normal, rng);
            }

            ray = Ray(hitPos + normal * 1e-3f, bounceDir);
            HitInfo bounceHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
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
                normal = -normal;
            }
            float ndotl = fmaxf(0.0f, dot(normal, -primaryRay.direction));
            color = baseColor * ndotl;
            // color = {
            //     // ndotl < 0, ndotl < 0, ndotl < 0
            //     ndotl > 0, ndotl > 0, ndotl > 0
            // };
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

__global__ void initEnvMissesKernel(const int* hitFlags,
                                    RenderParams params,
                                    float* envDirs,
                                    int* envHitFlags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (!hitFlags[sampleIdx]) {
            uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
            Ray primaryRay = generatePrimaryRay(x, y, params, rng);
            envDirs[base + 0] = primaryRay.direction.x;
            envDirs[base + 1] = primaryRay.direction.y;
            envDirs[base + 2] = primaryRay.direction.z;
            envHitFlags[sampleIdx] = 1;
        } else {
            envDirs[base + 0] = 0.0f;
            envDirs[base + 1] = 0.0f;
            envDirs[base + 2] = 0.0f;
            envHitFlags[sampleIdx] = 0;
        }
    }
}

__global__ void recordEnvMissesKernel(const int* missFlags,
                                      const float* missDirs,
                                      int* envHitFlags,
                                      float* envDirs,
                                      RenderParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        if (missFlags[sampleIdx] == 0 && envHitFlags[sampleIdx] == 0) {
            int base = sampleIdx * 3;
            envDirs[base + 0] = missDirs[base + 0];
            envDirs[base + 1] = missDirs[base + 1];
            envDirs[base + 2] = missDirs[base + 2];
            envHitFlags[sampleIdx] = 1;
        }
    }
}

__global__ void pathTraceNeuralEnvKernel(uchar4* output,
                                         Vec3* accum,
                                         const int* hitFlags,
                                         const float* hitColors,
                                         const int* envHitFlags,
                                         const float* envDirs,
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
        Vec3 radiance(0.0f, 0.0f, 0.0f);
        Vec3 throughput(1.0f, 1.0f, 1.0f);

        if (envHitFlags[sampleIdx]) {
            if (hitFlags[sampleIdx]) {
                int base = sampleIdx * 3;
                throughput = Vec3(
                        hitColors[base + 0],
                        hitColors[base + 1],
                        hitColors[base + 2]);
            }
            int base = sampleIdx * 3;
            Vec3 envDir(
                    envDirs[base + 0],
                    envDirs[base + 1],
                    envDirs[base + 2]);
            Vec3 envLight = sampleEnvironment(env, envDir);
            envLight = clampRadiance(envLight, params.maxRadiance);
            radiance += mul(throughput, envLight);
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

__global__ void initNeuralPathKernel(Vec3* throughput,
                                     Vec3* radiance,
                                     int* active,
                                     const int* hitFlags,
                                     const float* hitColors,
                                     RenderParams params,
                                     EnvironmentDeviceView env) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);

        Vec3 sampleRadiance(0.0f, 0.0f, 0.0f);
        Vec3 sampleThroughput(1.0f, 1.0f, 1.0f);
        int isActive = 0;

        if (hitFlags[sampleIdx]) {
            int base = sampleIdx * 3;
            sampleThroughput = Vec3(
                    hitColors[base + 0],
                    hitColors[base + 1],
                    hitColors[base + 2]);
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
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        if (!active[sampleIdx]) {
            continue;
        }

        if (!bounceHitFlags[sampleIdx]) {
            int base = sampleIdx * 3;
            Vec3 envDir(
                    bounceDirs[base + 0],
                    bounceDirs[base + 1],
                    bounceDirs[base + 2]);
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
        Vec3 color(
                bounceColors[base + 0],
                bounceColors[base + 1],
                bounceColors[base + 2]);
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
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

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

__global__ void lossNeuralKernel(float* lossValues,
                                 const float* hitPositions,
                                 const int* hitFlags,
                                 RenderParams params,
                                 float* lossMax,
                                 float* lossSum,
                                 int* lossHitCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int idx = y * params.width + x;
    int sampleIdx = idx;

    uint32_t rng = initRng(idx, params.sampleOffset, 0);
    Ray primaryRay = generatePrimaryRay(x, y, params, rng);
    Vec3 dir = primaryRay.direction;

    float loss = 0.0f;
    if (hitFlags[sampleIdx]) {
        Vec3 hitPos(
                hitPositions[sampleIdx * 3 + 0],
                hitPositions[sampleIdx * 3 + 1],
                hitPositions[sampleIdx * 3 + 2]);
        Vec3 toPoint = hitPos - params.camPos;
        float t = dot(toPoint, dir);
        Vec3 closest = params.camPos + dir * t;
        Vec3 delta = hitPos - closest;
        loss = length(delta);
    }

    lossValues[idx] = loss;
    if (hitFlags[sampleIdx] && loss > 0.0f) {
        atomicMaxFloat(lossMax, loss);
    }
    if (lossSum && hitFlags[sampleIdx]) {
        atomicAdd(lossSum, loss);
    }
    if (lossHitCount && hitFlags[sampleIdx]) {
        atomicAdd(lossHitCount, 1);
    }
}

__global__ void lossToneMapKernel(uchar4* output,
                                  const float* lossValues,
                                  RenderParams params,
                                  const float* lossMax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int idx = y * params.width + x;
    float maxLoss = *lossMax;
    float normalized = 0.0f;
    if (maxLoss > 1e-6f) {
        normalized = fminf(lossValues[idx] / maxLoss, 1.0f);
    }
    unsigned char value = static_cast<unsigned char>(normalized * 255.0f);
    output[idx] = make_uchar4(value, value, value, 255);
}

__global__ void scatterInputGradsKernel(const float* compactedGradients,
                                        const int* hitIndices,
                                        int hitCount,
                                        float* fullGradients) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }
    int pixelIdx = hitIndices[idx];
    int compactBase = idx * 3;
    int fullBase = pixelIdx * 3;
    fullGradients[fullBase + 0] = compactedGradients[compactBase + 0];
    fullGradients[fullBase + 1] = compactedGradients[compactBase + 1];
    fullGradients[fullBase + 2] = compactedGradients[compactBase + 2];
}

}  // namespace

RendererNeural::RendererNeural(Scene& scene)
        : scene_(&scene),
          lightDir_(normalize(Vec3(1.0f, 1.5f, -1.0f))) {
    if (!tcnn::cpp::has_networks()) {
        std::fprintf(stderr, "tiny-cuda-nn was built without network support.\n");
        return;
    }

    tcnn::cpp::json encoding = {
            {"otype", "HashGrid"},
            {"n_levels", 8},
            {"n_features_per_level", 8},
            {"log2_hashmap_size", 13},
            {"base_resolution", 2},
            {"per_level_scale", 2.0},
            {"fixed_point_pos", false},
    };
    tcnn::cpp::json network = {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 4},
    };

    network_ = tcnn::cpp::create_network_with_input_encoding(3, 3, encoding, network);
    if (!network_) {
        std::fprintf(stderr, "Failed to create tiny-cuda-nn network.\n");
        return;
    }

    if (network_->param_precision() != tcnn::cpp::Precision::Fp16 ||
            network_->output_precision() != tcnn::cpp::Precision::Fp16) {
        std::fprintf(stderr, "RendererNeural only supports FP16 networks.\n");
        delete network_;
        network_ = nullptr;
        return;
    }

    outputDims_ = network_->n_output_dims();
    outputElemSize_ = precisionBytes(network_->output_precision());
    paramsBytes_ = network_->n_params() * precisionBytes(network_->param_precision());
    if (paramsBytes_ > 0) {
        checkCuda(cudaMalloc(&params_, paramsBytes_), "cudaMalloc tcnn params");
        checkCuda(cudaMalloc(&dL_dparams_, paramsBytes_), "cudaMalloc tcnn dL_dparams");
        checkCuda(cudaMemset(dL_dparams_, 0, paramsBytes_), "cudaMemset tcnn dL_dparams");
        size_t seed = randomSeed();
        float* paramsFull = nullptr;
        size_t fullBytes = network_->n_params() * sizeof(float);
        checkCuda(cudaMalloc(&paramsFull, fullBytes), "cudaMalloc tcnn params full");
        network_->initialize_params(seed, paramsFull, 1.0f);
        const uint32_t count = static_cast<uint32_t>(network_->n_params());
        const int blockSize = 256;
        int gridSize = static_cast<int>((count + blockSize - 1) / blockSize);
        tcnn::cast<__half><<<gridSize, blockSize>>>(count, paramsFull, static_cast<__half*>(params_));
        checkCuda(cudaGetLastError(), "tcnn params cast");
        checkCuda(cudaFree(paramsFull), "cudaFree tcnn params full");
    }
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
    if (!network_ || !params_ || paramsBytes_ == 0) {
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
    if (static_cast<size_t>(size) != paramsBytes_) {
        std::fprintf(stderr,
                     "Weights size mismatch (got %lld bytes, expected %zu).\n",
                     static_cast<long long>(size),
                     paramsBytes_);
        return false;
    }

    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), size)) {
        std::fprintf(stderr, "Failed to read weights file: %s\n", path.c_str());
        return false;
    }

    checkCuda(cudaMemcpy(params_, buffer.data(), paramsBytes_, cudaMemcpyHostToDevice),
              "cudaMemcpy tcnn params");
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

    Mesh& mesh = useNeuralQuery_ ? scene_->roughMesh() : scene_->exactMesh();
    if (!mesh.uploadToDevice()) {
        return;
    }
    EnvironmentMap& environment = scene_->environment();
    environment.uploadToDevice();
    MeshDeviceView meshView = mesh.deviceView();
    EnvironmentDeviceView envView = environment.deviceView();

    size_t pixelCount = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    int samplesPerPixel = samplesPerPixel_;
    if (samplesPerPixel <= 0) {
        samplesPerPixel = 1;
    }
    size_t elementCount = pixelCount * static_cast<size_t>(samplesPerPixel);
    size_t paddedCount = elementCount;
    if (network_) {
        size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
        paddedCount = roundUp(elementCount, granularity);
    }
    if (!ensureNetworkBuffers(paddedCount)) {
        return;
    }
    if (!ensureAccumBuffer(pixelCount)) {
        return;
    }

    Vec3 meshMin = mesh.boundsMin();
    Vec3 meshMax = mesh.boundsMax();
    Vec3 meshExtent = meshMax - meshMin;
    Vec3 meshInvExtent(
            meshExtent.x != 0.0f ? 1.0f / meshExtent.x : 0.0f,
            meshExtent.y != 0.0f ? 1.0f / meshExtent.y : 0.0f,
            meshExtent.z != 0.0f ? 1.0f / meshExtent.z : 0.0f);

    bool cameraMoved = !hasLastCamera_;
    if (!cameraMoved) {
        const float kEps = 1e-4f;
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
        maxBounces != lastBounceCount_ || samplesPerPixel != lastSamplesPerPixel_) {
        resetAccum();
    }
    lastUseNeuralQuery_ = useNeuralQuery_;
    lastLambertView_ = lambertView_;
    lastBounceCount_ = maxBounces;
    lastSamplesPerPixel_ = samplesPerPixel;
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
    params.meshMin = meshMin;
    params.meshInvExtent = meshInvExtent;
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

    dim3 block(8, 8);  // reduce block size to allow more resident blocks when register pressure is high
    dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    renderNeuralKernel<<<grid, block>>>(
            inputs_,
            hitPositions_,
            normals_,
            hitColors_,
            hitFlags_,
            params,
            meshView);
    checkCuda(cudaGetLastError(), "renderNeuralKernel launch");

    const int compactBlock = 256;
    int compactGrid = static_cast<int>((elementCount + compactBlock - 1) / compactBlock);
    const int blockSize = 256;
    int elementCountInt = static_cast<int>(elementCount);

    bool neuralActive = false;
    if (useNeuralQuery_ && network_) {
        bool needInput = true;
        bool needWeights = false;
        if (inputs_ && hitPositions_ && normals_ && compactedInputs_ &&
                outputs_ && dL_doutput_ && params_ &&
                hitFlags_ && hitIndices_ && hitCount_ &&
                (!needInput || (dL_dinput_ && compactedDLDInput_)) &&
                (!needWeights || dL_dparams_)) {
            neuralActive = true;
            checkCuda(cudaMemset(hitCount_, 0, sizeof(int)), "cudaMemset hitCount");
            compactInputsKernel<<<compactGrid, compactBlock>>>(
                    inputs_,
                    hitFlags_,
                    elementCountInt,
                    compactedInputs_,
                    hitIndices_,
                    hitCount_);
            checkCuda(cudaGetLastError(), "compactInputsKernel launch");

            int hitCount = 0;
            checkCuda(cudaMemcpy(&hitCount, hitCount_, sizeof(int), cudaMemcpyDeviceToHost),
                      "cudaMemcpy hitCount");
            if (hitCount > 0) {
                if (outputDims_ < 3 || !normals_) {
                    std::fprintf(stderr, "Network output dims or normal buffer invalid.\n");
                    hitCount = 0;
                }
            }
            if (hitCount > 0) {
                size_t hitCountSize = static_cast<size_t>(hitCount);
                size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
                paddedCount = roundUp(hitCountSize, granularity);
                if (paddedCount > hitCountSize) {
                    size_t tail = paddedCount - hitCountSize;
                    checkCuda(cudaMemset(
                            compactedInputs_ + hitCountSize * 3,
                            0,
                            tail * 3 * sizeof(float)),
                            "cudaMemset tcnn compacted inputs tail");
                }

                size_t outputCount = paddedCount * static_cast<size_t>(outputDims_);
                size_t outputBytes = outputCount * outputElemSize_;
                const int blockSize = 256;
                int gradGrid = static_cast<int>((hitCount + blockSize - 1) / blockSize);
                int inputGrid = static_cast<int>(((hitCount * 3) + blockSize - 1) / blockSize);
                int steps = gdSteps_;
                if (steps < 0) {
                    steps = 0;
                }
                // if (steps > 10) {
                //     steps = 10;
                // }
                constexpr float kLearningRate = 0.0003f;

                for (int step = 0; step < steps; ++step) {
                    tcnn::cpp::Context ctx = network_->forward(
                        0,
                        static_cast<uint32_t>(paddedCount),
                        compactedInputs_,
                        outputs_,
                        params_,
                        true
                    );
                    (void)ctx;

                    checkCuda(cudaMemset(dL_doutput_, 0, outputBytes), "cudaMemset tcnn dL_doutput");
                    computeLossGradKernel<<<gradGrid, blockSize>>>(
                            compactedInputs_,
                            static_cast<const __half*>(outputs_),
                            hitIndices_,
                            hitCount,
                            params,
                            meshMin,
                            meshExtent,
                            static_cast<int>(outputDims_),
                            static_cast<__half*>(dL_doutput_));
                    checkCuda(cudaGetLastError(), "computeLossGradKernel launch");

                    network_->backward(
                            0,
                            ctx,
                            static_cast<uint32_t>(paddedCount),
                            compactedDLDInput_,
                            dL_doutput_,
                            nullptr,
                            compactedInputs_,
                            outputs_,
                            params_);

                    addDirectGradKernel<<<gradGrid, blockSize>>>(
                            compactedInputs_,
                            static_cast<const __half*>(outputs_),
                            hitIndices_,
                            hitCount,
                            params,
                            meshMin,
                            meshExtent,
                            static_cast<int>(outputDims_),
                            compactedDLDInput_);
                    checkCuda(cudaGetLastError(), "addDirectGradKernel launch");

                    sgdInputsKernel<<<inputGrid, blockSize>>>(
                            compactedInputs_,
                            compactedDLDInput_,
                            hitCount,
                            kLearningRate);
                    checkCuda(cudaGetLastError(), "sgdInputsKernel launch");

                    projectInputsToMeshKernel<<<gradGrid, blockSize>>>(
                            compactedInputs_,
                            hitCount,
                            meshMin,
                            meshExtent,
                            meshInvExtent,
                            meshView);
                    checkCuda(cudaGetLastError(), "projectInputsToMeshKernel launch");
                }

                {
                    tcnn::cpp::Context ctx = network_->forward(
                        0,
                        static_cast<uint32_t>(paddedCount),
                        compactedInputs_,
                        outputs_,
                        params_,
                        false
                    );
                    (void)ctx;
                }

                int applyGrid = (hitCount + blockSize - 1) / blockSize;
                applyNetworkDeltaKernel<<<applyGrid, blockSize>>>(
                        compactedInputs_,
                        static_cast<const __half*>(outputs_),
                        hitIndices_,
                        hitCount,
                        meshMin,
                        meshExtent,
                        static_cast<int>(outputDims_),
                        hitPositions_,
                        normals_);
                checkCuda(cudaGetLastError(), "applyNetworkDeltaKernel launch");

                // if (needInput) {
                //     checkCuda(cudaMemset(dL_dinput_, 0, elementCount * 3 * sizeof(float)),
                //               "cudaMemset dL_dinput");
                //     int scatterGrid = (hitCount + blockSize - 1) / blockSize;
                //     scatterInputGradsKernel<<<scatterGrid, blockSize>>>(
                //             compactedDLDInput_,
                //             hitIndices_,
                //             hitCount,
                //             dL_dinput_);
                //     checkCuda(cudaGetLastError(), "scatterInputGradsKernel launch");
                // }
            }
        }
    }

    auto forwardOnHits = [&](float* inputs,
                             int* flags,
                             float* positions,
                             float* normalsOut,
                             const char* tag) {
        if (!neuralActive) {
            return 0;
        }
        checkCuda(cudaMemset(hitCount_, 0, sizeof(int)), tag);
        compactInputsKernel<<<compactGrid, compactBlock>>>(
                inputs,
                flags,
                elementCountInt,
                compactedInputs_,
                hitIndices_,
                hitCount_);
        checkCuda(cudaGetLastError(), "compactInputsKernel forward launch");

        int hitCount = 0;
        checkCuda(cudaMemcpy(&hitCount, hitCount_, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy forward hitCount");
        if (hitCount > 0) {
            if (outputDims_ < 3 || !normalsOut) {
                std::fprintf(stderr, "Network output dims or normal buffer invalid.\n");
                return 0;
            }
            size_t hitCountSize = static_cast<size_t>(hitCount);
            size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
            size_t paddedCount = roundUp(hitCountSize, granularity);
            if (paddedCount > hitCountSize) {
                size_t tail = paddedCount - hitCountSize;
                checkCuda(cudaMemset(
                        compactedInputs_ + hitCountSize * 3,
                        0,
                        tail * 3 * sizeof(float)),
                        "cudaMemset tcnn forward compacted inputs tail");
            }

            {
                tcnn::cpp::Context ctx = network_->forward(
                    0,
                    static_cast<uint32_t>(paddedCount),
                    compactedInputs_,
                    outputs_,
                    params_,
                    false
                );
                (void)ctx;
            }

            int applyGrid = (hitCount + blockSize - 1) / blockSize;
            applyNetworkDeltaKernel<<<applyGrid, blockSize>>>(
                    compactedInputs_,
                    static_cast<const __half*>(outputs_),
                    hitIndices_,
                    hitCount,
                    meshMin,
                    meshExtent,
                    static_cast<int>(outputDims_),
                    positions,
                    normalsOut);
            checkCuda(cudaGetLastError(), "applyNetworkDeltaKernel forward launch");
        }
        return hitCount;
    };

    if (neuralActive && !lossView_ && !lambertView_) {
        initNeuralPathKernel<<<grid, block>>>(
                pathThroughput_,
                pathRadiance_,
                pathActive_,
                hitFlags_,
                hitColors_,
                params,
                envView);
        checkCuda(cudaGetLastError(), "initNeuralPathKernel launch");

        if (maxBounces > 0) {
            renderBounceKernel<<<grid, block>>>(
                    hitPositions_,
                    normals_,
                    hitFlags_,
                    params,
                    meshView,
                    bounceInputs_,
                    bouncePositions_,
                    bounceNormals_,
                    bounceColors_,
                    bounceHitFlags_,
                    bounceDirs_);
            checkCuda(cudaGetLastError(), "renderBounceKernel launch");

            forwardOnHits(bounceInputs_, bounceHitFlags_, bouncePositions_, bounceNormals_,
                          "cudaMemset bounce hitCount");
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

            float* inInputs = bounceInputs_;
            float* inPositions = bouncePositions_;
            float* inNormals = bounceNormals_;
            float* inColors = bounceColors_;
            int* inFlags = bounceHitFlags_;
            float* inDirs = bounceDirs_;
            float* outInputs = bounce2Inputs_;
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
                        static_cast<uint32_t>(bounce),
                        params,
                        meshView,
                        outInputs,
                        outPositions,
                        outNormals,
                        outColors,
                        outFlags,
                        outDirs);
                checkCuda(cudaGetLastError(), "renderBounceFromStateKernel launch");

                forwardOnHits(outInputs, outFlags, outPositions, outNormals,
                              "cudaMemset bounce hitCount");
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

                std::swap(inInputs, outInputs);
                std::swap(inPositions, outPositions);
                std::swap(inNormals, outNormals);
                std::swap(inColors, outColors);
                std::swap(inFlags, outFlags);
                std::swap(inDirs, outDirs);
            }
        }
    }

    if (lossValues_ && lossMax_ && lossSum_ && lossHitCount_) {
        checkCuda(cudaMemset(lossMax_, 0, sizeof(float)), "cudaMemset lossMax");
        checkCuda(cudaMemset(lossSum_, 0, sizeof(float)), "cudaMemset lossSum");
        checkCuda(cudaMemset(lossHitCount_, 0, sizeof(int)), "cudaMemset lossHitCount");
        lossNeuralKernel<<<grid, block>>>(lossValues_, hitPositions_, hitFlags_, params, lossMax_, lossSum_, lossHitCount_);
        checkCuda(cudaGetLastError(), "lossNeuralKernel launch");

        float lossSumHost = 0.0f;
        int hitCountHost = 0;
        checkCuda(cudaMemcpy(&lossSumHost, lossSum_, sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy lossSum");
        checkCuda(cudaMemcpy(&hitCountHost, lossHitCount_, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy lossHitCount");
        lastHitCount_ = hitCountHost;
        if (hitCountHost > 0) {
            lastAvgLoss_ = lossSumHost / static_cast<float>(hitCountHost);
        } else {
            lastAvgLoss_ = 0.0f;
        }
    }

    bool useNeuralBounce = neuralActive && !lossView_ && !lambertView_;
    if (lambertView_) {
        lambertKernel<<<grid, block>>>(
                devicePixels_,
                normals_,
                hitColors_,
                hitFlags_,
                params,
                envView);
        checkCuda(cudaGetLastError(), "lambertKernel launch");
        accumSampleCount_ = 0;
    } else if (lossView_) {
        if (lossValues_ && lossMax_) {
            lossToneMapKernel<<<grid, block>>>(devicePixels_, lossValues_, params, lossMax_);
            checkCuda(cudaGetLastError(), "lossToneMapKernel launch");
        }
    } else if (useNeuralBounce) {
        finalizeNeuralPathKernel<<<grid, block>>>(
                devicePixels_,
                accum_,
                pathRadiance_,
                params);
        checkCuda(cudaGetLastError(), "finalizeNeuralPathKernel launch");
        accumSampleCount_ += static_cast<uint32_t>(samplesPerPixel);
    } else {
        pathTraceKernel<<<grid, block>>>(
                devicePixels_,
                accum_,
                hitPositions_,
                normals_,
                hitColors_,
                hitFlags_,
                params,
                meshView,
                envView);
        checkCuda(cudaGetLastError(), "pathTraceKernel launch");
        accumSampleCount_ += static_cast<uint32_t>(samplesPerPixel);
    }

    checkCuda(cudaMemcpy(
            hostPixels.data(),
            devicePixels_,
            hostPixels.size() * sizeof(uchar4),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy");
}

void RendererNeural::release() {
    if (devicePixels_) {
        cudaFree(devicePixels_);
        devicePixels_ = nullptr;
    }
    if (accum_) {
        cudaFree(accum_);
        accum_ = nullptr;
    }
    if (inputs_) {
        cudaFree(inputs_);
        inputs_ = nullptr;
    }
    if (hitPositions_) {
        cudaFree(hitPositions_);
        hitPositions_ = nullptr;
    }
    if (hitColors_) {
        cudaFree(hitColors_);
        hitColors_ = nullptr;
    }
    if (compactedInputs_) {
        cudaFree(compactedInputs_);
        compactedInputs_ = nullptr;
    }
    if (normals_) {
        cudaFree(normals_);
        normals_ = nullptr;
    }
    if (bounceInputs_) {
        cudaFree(bounceInputs_);
        bounceInputs_ = nullptr;
    }
    if (bouncePositions_) {
        cudaFree(bouncePositions_);
        bouncePositions_ = nullptr;
    }
    if (bounceNormals_) {
        cudaFree(bounceNormals_);
        bounceNormals_ = nullptr;
    }
    if (bounceColors_) {
        cudaFree(bounceColors_);
        bounceColors_ = nullptr;
    }
    if (bounceDirs_) {
        cudaFree(bounceDirs_);
        bounceDirs_ = nullptr;
    }
    if (bounce2Inputs_) {
        cudaFree(bounce2Inputs_);
        bounce2Inputs_ = nullptr;
    }
    if (bounce2Positions_) {
        cudaFree(bounce2Positions_);
        bounce2Positions_ = nullptr;
    }
    if (bounce2Normals_) {
        cudaFree(bounce2Normals_);
        bounce2Normals_ = nullptr;
    }
    if (bounce2Colors_) {
        cudaFree(bounce2Colors_);
        bounce2Colors_ = nullptr;
    }
    if (bounce2Dirs_) {
        cudaFree(bounce2Dirs_);
        bounce2Dirs_ = nullptr;
    }
    if (envDirs_) {
        cudaFree(envDirs_);
        envDirs_ = nullptr;
    }
    if (pathThroughput_) {
        cudaFree(pathThroughput_);
        pathThroughput_ = nullptr;
    }
    if (pathRadiance_) {
        cudaFree(pathRadiance_);
        pathRadiance_ = nullptr;
    }
    if (lossValues_) {
        cudaFree(lossValues_);
        lossValues_ = nullptr;
    }
    if (lossMax_) {
        cudaFree(lossMax_);
        lossMax_ = nullptr;
    }
    if (lossSum_) {
        cudaFree(lossSum_);
        lossSum_ = nullptr;
    }
    if (lossHitCount_) {
        cudaFree(lossHitCount_);
        lossHitCount_ = nullptr;
    }
    if (outputs_) {
        cudaFree(outputs_);
        outputs_ = nullptr;
    }
    if (dL_doutput_) {
        cudaFree(dL_doutput_);
        dL_doutput_ = nullptr;
    }
    if (dL_dinput_) {
        cudaFree(dL_dinput_);
        dL_dinput_ = nullptr;
    }
    if (compactedDLDInput_) {
        cudaFree(compactedDLDInput_);
        compactedDLDInput_ = nullptr;
    }
    if (hitFlags_) {
        cudaFree(hitFlags_);
        hitFlags_ = nullptr;
    }
    if (bounceHitFlags_) {
        cudaFree(bounceHitFlags_);
        bounceHitFlags_ = nullptr;
    }
    if (bounce2HitFlags_) {
        cudaFree(bounce2HitFlags_);
        bounce2HitFlags_ = nullptr;
    }
    if (envHitFlags_) {
        cudaFree(envHitFlags_);
        envHitFlags_ = nullptr;
    }
    if (pathActive_) {
        cudaFree(pathActive_);
        pathActive_ = nullptr;
    }
    if (hitIndices_) {
        cudaFree(hitIndices_);
        hitIndices_ = nullptr;
    }
    if (hitCount_) {
        cudaFree(hitCount_);
        hitCount_ = nullptr;
    }
    bufferElements_ = 0;
    accumPixels_ = 0;
    accumSampleCount_ = 0;
    hasLastCamera_ = false;
}

void RendererNeural::releaseNetwork() {
    if (params_) {
        cudaFree(params_);
        params_ = nullptr;
    }
    if (dL_dparams_) {
        cudaFree(dL_dparams_);
        dL_dparams_ = nullptr;
    }
    delete network_;
    network_ = nullptr;
    paramsBytes_ = 0;
    outputDims_ = 0;
    outputElemSize_ = 0;
}

bool RendererNeural::ensureNetworkBuffers(size_t elementCount) {
    if (elementCount == 0) {
        return false;
    }
    if (elementCount <= bufferElements_ &&
            hitColors_ &&
            bounceInputs_ && bouncePositions_ && bounceNormals_ && bounceDirs_ && bounceColors_ && bounceHitFlags_ &&
            bounce2Inputs_ && bounce2Positions_ && bounce2Normals_ && bounce2Dirs_ && bounce2Colors_ && bounce2HitFlags_ &&
            envDirs_ && envHitFlags_ &&
            pathThroughput_ && pathRadiance_ && pathActive_) {
        return true;
    }
    if (inputs_) {
        cudaFree(inputs_);
        inputs_ = nullptr;
    }
    if (hitPositions_) {
        cudaFree(hitPositions_);
        hitPositions_ = nullptr;
    }
    if (hitColors_) {
        cudaFree(hitColors_);
        hitColors_ = nullptr;
    }
    if (compactedInputs_) {
        cudaFree(compactedInputs_);
        compactedInputs_ = nullptr;
    }
    if (outputs_) {
        cudaFree(outputs_);
        outputs_ = nullptr;
    }
    if (dL_doutput_) {
        cudaFree(dL_doutput_);
        dL_doutput_ = nullptr;
    }
    if (dL_dinput_) {
        cudaFree(dL_dinput_);
        dL_dinput_ = nullptr;
    }
    if (compactedDLDInput_) {
        cudaFree(compactedDLDInput_);
        compactedDLDInput_ = nullptr;
    }
    if (normals_) {
        cudaFree(normals_);
        normals_ = nullptr;
    }
    if (bounceInputs_) {
        cudaFree(bounceInputs_);
        bounceInputs_ = nullptr;
    }
    if (bouncePositions_) {
        cudaFree(bouncePositions_);
        bouncePositions_ = nullptr;
    }
    if (bounceNormals_) {
        cudaFree(bounceNormals_);
        bounceNormals_ = nullptr;
    }
    if (bounceColors_) {
        cudaFree(bounceColors_);
        bounceColors_ = nullptr;
    }
    if (bounceDirs_) {
        cudaFree(bounceDirs_);
        bounceDirs_ = nullptr;
    }
    if (bounce2Inputs_) {
        cudaFree(bounce2Inputs_);
        bounce2Inputs_ = nullptr;
    }
    if (bounce2Positions_) {
        cudaFree(bounce2Positions_);
        bounce2Positions_ = nullptr;
    }
    if (bounce2Normals_) {
        cudaFree(bounce2Normals_);
        bounce2Normals_ = nullptr;
    }
    if (bounce2Colors_) {
        cudaFree(bounce2Colors_);
        bounce2Colors_ = nullptr;
    }
    if (bounce2Dirs_) {
        cudaFree(bounce2Dirs_);
        bounce2Dirs_ = nullptr;
    }
    if (envDirs_) {
        cudaFree(envDirs_);
        envDirs_ = nullptr;
    }
    if (pathThroughput_) {
        cudaFree(pathThroughput_);
        pathThroughput_ = nullptr;
    }
    if (pathRadiance_) {
        cudaFree(pathRadiance_);
        pathRadiance_ = nullptr;
    }
    if (lossValues_) {
        cudaFree(lossValues_);
        lossValues_ = nullptr;
    }
    if (lossMax_) {
        cudaFree(lossMax_);
        lossMax_ = nullptr;
    }
    if (lossSum_) {
        cudaFree(lossSum_);
        lossSum_ = nullptr;
    }
    if (lossHitCount_) {
        cudaFree(lossHitCount_);
        lossHitCount_ = nullptr;
    }
    if (hitFlags_) {
        cudaFree(hitFlags_);
        hitFlags_ = nullptr;
    }
    if (bounceHitFlags_) {
        cudaFree(bounceHitFlags_);
        bounceHitFlags_ = nullptr;
    }
    if (bounce2HitFlags_) {
        cudaFree(bounce2HitFlags_);
        bounce2HitFlags_ = nullptr;
    }
    if (envHitFlags_) {
        cudaFree(envHitFlags_);
        envHitFlags_ = nullptr;
    }
    if (pathActive_) {
        cudaFree(pathActive_);
        pathActive_ = nullptr;
    }
    if (hitIndices_) {
        cudaFree(hitIndices_);
        hitIndices_ = nullptr;
    }
    if (hitCount_) {
        cudaFree(hitCount_);
        hitCount_ = nullptr;
    }

    size_t inputBytes = elementCount * 3 * sizeof(float);
    checkCuda(cudaMalloc(&inputs_, inputBytes), "cudaMalloc tcnn inputs");
    checkCuda(cudaMalloc(&hitPositions_, inputBytes), "cudaMalloc hit positions");
    checkCuda(cudaMalloc(&hitColors_, inputBytes), "cudaMalloc hit colors");
    checkCuda(cudaMalloc(&compactedInputs_, inputBytes), "cudaMalloc tcnn compacted inputs");
    checkCuda(cudaMalloc(&normals_, inputBytes), "cudaMalloc tcnn normals");
    checkCuda(cudaMalloc(&bounceInputs_, inputBytes), "cudaMalloc bounce inputs");
    checkCuda(cudaMalloc(&bouncePositions_, inputBytes), "cudaMalloc bounce positions");
    checkCuda(cudaMalloc(&bounceNormals_, inputBytes), "cudaMalloc bounce normals");
    checkCuda(cudaMalloc(&bounceColors_, inputBytes), "cudaMalloc bounce colors");
    checkCuda(cudaMalloc(&bounceDirs_, inputBytes), "cudaMalloc bounce dirs");
    checkCuda(cudaMalloc(&bounce2Inputs_, inputBytes), "cudaMalloc bounce2 inputs");
    checkCuda(cudaMalloc(&bounce2Positions_, inputBytes), "cudaMalloc bounce2 positions");
    checkCuda(cudaMalloc(&bounce2Normals_, inputBytes), "cudaMalloc bounce2 normals");
    checkCuda(cudaMalloc(&bounce2Colors_, inputBytes), "cudaMalloc bounce2 colors");
    checkCuda(cudaMalloc(&bounce2Dirs_, inputBytes), "cudaMalloc bounce2 dirs");
    checkCuda(cudaMalloc(&envDirs_, inputBytes), "cudaMalloc env dirs");
    checkCuda(cudaMalloc(&pathThroughput_, elementCount * sizeof(Vec3)), "cudaMalloc path throughput");
    checkCuda(cudaMalloc(&pathRadiance_, elementCount * sizeof(Vec3)), "cudaMalloc path radiance");
    checkCuda(cudaMalloc(&lossValues_, elementCount * sizeof(float)), "cudaMalloc loss values");
    checkCuda(cudaMalloc(&lossMax_, sizeof(float)), "cudaMalloc loss max");
    checkCuda(cudaMalloc(&lossSum_, sizeof(float)), "cudaMalloc loss sum");
    checkCuda(cudaMalloc(&lossHitCount_, sizeof(int)), "cudaMalloc loss hit count");

    size_t outputBytes = elementCount * static_cast<size_t>(outputDims_) * outputElemSize_;
    if (outputBytes > 0) {
        checkCuda(cudaMalloc(&outputs_, outputBytes), "cudaMalloc tcnn outputs");
        checkCuda(cudaMalloc(&dL_doutput_, outputBytes), "cudaMalloc tcnn dL_doutput");
    }
    size_t inputGradBytes = elementCount * 3 * sizeof(float);
    checkCuda(cudaMalloc(&dL_dinput_, inputGradBytes), "cudaMalloc tcnn dL_dinput");
    checkCuda(cudaMalloc(&compactedDLDInput_, inputGradBytes), "cudaMalloc tcnn compacted dL_dinput");
    checkCuda(cudaMalloc(&hitFlags_, elementCount * sizeof(int)), "cudaMalloc tcnn hit flags");
    checkCuda(cudaMalloc(&bounceHitFlags_, elementCount * sizeof(int)), "cudaMalloc bounce hit flags");
    checkCuda(cudaMalloc(&bounce2HitFlags_, elementCount * sizeof(int)), "cudaMalloc bounce2 hit flags");
    checkCuda(cudaMalloc(&envHitFlags_, elementCount * sizeof(int)), "cudaMalloc env hit flags");
    checkCuda(cudaMalloc(&pathActive_, elementCount * sizeof(int)), "cudaMalloc path active");
    checkCuda(cudaMalloc(&hitIndices_, elementCount * sizeof(int)), "cudaMalloc tcnn hit indices");
    checkCuda(cudaMalloc(&hitCount_, sizeof(int)), "cudaMalloc tcnn hit count");

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
