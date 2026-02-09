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

#include "config_loader.h"
#include "scene.h"
#include "disney_brdf.cuh"
#include "mesh_intersection.cuh"
#include "hit_info.h"
#include "ray.h"

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

namespace {

struct BoundingBox {
    Vec3 min;
    Vec3 max;
};

// Intersection mode for path tracing
enum IntersectionMode {
    GT_MESH,   // Ground truth mesh intersection
    NEURAL     // Neural network intersection
};

// Hit data structure returned by intersection kernels
struct HitData {
    bool hit;           // Whether ray hit anything
    Vec3 position;      // Hit position in world space
    Vec3 normal;        // Surface normal at hit point
    Vec3 albedo;        // Surface color/albedo (texture-modulated)
    Vec3 materialParams; // Material parameter triplet
    float distance;     // Distance to hit (for ray offsetting)
};

__device__ inline float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ inline Vec3 mul(Vec3 a, Vec3 b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float linearToSrgb(float v) {
    // Clamp to prevent negative values
    v = fmaxf(0.0f, v);

    // Apply correct sRGB formula
    float result;
    if (v <= 0.0031308f) {
        result = 12.92f * v;
    } else {
        result = 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
    }

    // Clamp output to [0, 1] range for display
    return fminf(1.0f, result);
}


__device__ inline Vec3 encodeSrgb(Vec3 c) {
    return Vec3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z));
}

__device__ inline Vec3 sampleEnvironment(EnvironmentDeviceView env, Vec3 dir) {
    if (env.pixels && env.width > 0 && env.height > 0) {
        // Apply rotation around Y axis
        if (env.rotation != 0.0f) {
            const float kDegToRad = 3.14159265358979323846f / 180.0f;
            float angle = env.rotation * kDegToRad;
            float cosA = cosf(angle);
            float sinA = sinf(angle);
            float newX = dir.x * cosA + dir.z * sinA;
            float newZ = -dir.x * sinA + dir.z * cosA;
            dir.x = newX;
            dir.z = newZ;
        }

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

        // Sample environment and apply strength
        Vec3 envColor = env.pixels[y * env.width + x] * env.strength;

        // Clamp to 100.0 to avoid fireflies from bright light sources (matching nbvh)
        envColor.x = fminf(envColor.x, 100.0f);
        envColor.y = fminf(envColor.y, 100.0f);
        envColor.z = fminf(envColor.z, 100.0f);

        return envColor;
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

__device__ inline void buildTangentSpace(Vec3 normal, Vec3* tangent, Vec3* bitangent) {
    // Match NBVH's ortho_basis: pick most perpendicular cardinal axis
    Vec3 bitangent_init = Vec3(0.0f, 0.0f, 0.0f);

    if (normal.x < 0.6f && normal.x > -0.6f) {
        bitangent_init.x = 1.0f;
    } else if (normal.y < 0.6f && normal.y > -0.6f) {
        bitangent_init.y = 1.0f;
    } else if (normal.z < 0.6f && normal.z > -0.6f) {
        bitangent_init.z = 1.0f;
    } else {
        bitangent_init.x = 1.0f;
    }

    *tangent = normalize(cross(bitangent_init, normal));
    *bitangent = normalize(cross(normal, *tangent));
}

// Helper function: Sample environment map with radiance clamping
__forceinline__ __device__ Vec3 sampleEnvironmentWithClamp(EnvironmentDeviceView env,
                                                            Vec3 direction,
                                                            float maxRadiance) {
    Vec3 envLight = sampleEnvironment(env, direction);
    return Vec3(
        fminf(envLight.x, maxRadiance),
        fminf(envLight.y, maxRadiance),
        fminf(envLight.z, maxRadiance)
    );
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

// TraceMode controls which triangle faces are considered for intersection.
// FORWARD_ONLY: Only faces facing the camera (normal dot direction < 0) - for shell entry
// BACKWARD_ONLY: Only faces facing away (normal dot direction > 0) - for shell exit
// ANY: All faces regardless of orientation
enum class TraceMode {
    ANY,
    FORWARD_ONLY,
    BACKWARD_ONLY,
};

// Helper to compute geometric normal for a triangle (for face culling)
__device__ inline Vec3 getTriangleNormal(const MeshDeviceView& mesh, int triIdx) {
    uint3 idx = mesh.indices[triIdx];
    Vec3 v0 = mesh.vertices[idx.x];
    Vec3 v1 = mesh.vertices[idx.y];
    Vec3 v2 = mesh.vertices[idx.z];
    return normalize(cross(v1 - v0, v2 - v0));
}

// Trace mesh with indexed geometry - returns HitInfo with full intersection data
__device__ inline bool traceMeshWithMode(const Ray& ray,
                                         MeshDeviceView mesh,
                                         HitInfo* outHit,
                                         TraceMode mode,
                                         bool allowNegative,
                                         const Material& globalMaterial) {
    if (mesh.numBvhNodes <= 0 || mesh.numTriangles <= 0) {
        return false;
    }

    PreliminaryHitInfo bestPi;
    bestPi.t = 1e30f;
    uint32_t bestTriIdx = 0;
    float minT = allowNegative ? -1e30f : 1e-6f;
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
                // Get triangle vertices
                uint3 idx = mesh.indices[i];
                Vec3 v0 = mesh.vertices[idx.x];
                Vec3 v1 = mesh.vertices[idx.y];
                Vec3 v2 = mesh.vertices[idx.z];

                // Compute geometric normal for face culling
                Vec3 triNormal = normalize(cross(v1 - v0, v2 - v0));
                float facingDot = dot(triNormal, ray.direction);

                // Apply trace mode filter
                if (mode == TraceMode::FORWARD_ONLY && facingDot >= 0.0f) {
                    continue;  // Skip back-facing triangles
                }
                if (mode == TraceMode::BACKWARD_ONLY && facingDot <= 0.0f) {
                    continue;  // Skip front-facing triangles
                }

                // Intersect with indexed geometry
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

    if (bestPi.valid()) {
        // Compute full hit data from preliminary info
        bestPi.primIdx = bestTriIdx;
        HitInfo hitInfo = computeHitData(bestPi, bestTriIdx, ray, mesh);
        if (outHit) {
            *outHit = hitInfo;
        }
        return true;
    }

    return false;
}

// Main trace function with indexed geometry
__device__ inline bool traceMesh(const Ray& ray,
                                 MeshDeviceView mesh,
                                 HitInfo* outHit,
                                 bool cullBackfaces,
                                 const Material& globalMaterial) {
    TraceMode mode = cullBackfaces ? TraceMode::FORWARD_ONLY : TraceMode::ANY;
    return traceMeshWithMode(ray, mesh, outHit, mode, false, globalMaterial);
}

// ---------------------------------------------------------------------------
// Unified Intersection Kernels
// ---------------------------------------------------------------------------

// Device function: Trace a single ray against GT mesh and return HitData
__forceinline__ __device__ HitData traceRayGT(const Ray& ray,
                                               MeshDeviceView mesh,
                                               const Material& globalMaterial) {
    HitData result;
    HitInfo hitInfo;
    bool hit = traceMesh(ray, mesh, &hitInfo, true, globalMaterial);

    result.hit = hit;
    if (hit) {
        result.position = hitInfo.position;
        result.normal = hitInfo.shadingNormal;

        // Resolve material at hit point
        const Material* mat = &globalMaterial;
        if (hitInfo.materialId >= 0 && hitInfo.materialId < mesh.numMaterials && mesh.materials) {
            mat = &mesh.materials[hitInfo.materialId];
        }
        ResolvedMaterial resolved = resolveMaterial(*mat, hitInfo.uv, mesh);

        result.albedo = resolved.base_color;
        // result.albedo = Vec3(1.0f, 1.0f, 1.0f);  // Use white albedo to isolate material param effects
        result.materialParams = Vec3(resolved.metallic, resolved.roughness, resolved.specular);
        // result.materialParams = Vec3(0.0, 0.0, 0.0);
        result.distance = hitInfo.t;
    } else {
        result.position = Vec3(0.0f, 0.0f, 0.0f);
        result.normal = Vec3(0.0f, 0.0f, 0.0f);
        result.albedo = Vec3(0.0f, 0.0f, 0.0f);
        result.distance = 0.0f;
    }

    return result;
}

// Device function: Unified ray intersection wrapper
// NOTE: Currently uses GT mesh for both modes. Neural inference for arbitrary rays
// would require significant architectural changes (batched host-side inference).
// For bounce rays, using GT mesh provides accurate lighting even in neural mode.
// Kernel: Intersect primary rays with ground truth mesh
__global__ void intersectGroundTruthKernel(float* hitPositions,
                                           float* hitNormals,
                                           float* hitColors,
                                           float* hitMaterialParams,
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

        HitData hit = traceRayGT(ray, mesh, params.material);

        if (hit.hit) {
            hitPositions[base + 0] = hit.position.x;
            hitPositions[base + 1] = hit.position.y;
            hitPositions[base + 2] = hit.position.z;
            hitNormals[base + 0] = hit.normal.x;
            hitNormals[base + 1] = hit.normal.y;
            hitNormals[base + 2] = hit.normal.z;
            hitColors[base + 0] = hit.albedo.x;
            hitColors[base + 1] = hit.albedo.y;
            hitColors[base + 2] = hit.albedo.z;
            hitMaterialParams[base + 0] = hit.materialParams.x;
            hitMaterialParams[base + 1] = hit.materialParams.y;
            hitMaterialParams[base + 2] = hit.materialParams.z;
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
            hitMaterialParams[base + 0] = params.material.metallic.value;
            hitMaterialParams[base + 1] = params.material.roughness.value;
            hitMaterialParams[base + 2] = params.material.specular.value;
            hitFlags[sampleIdx] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Wavefront Path Tracing Kernels (Shared between GT and Neural)
// ---------------------------------------------------------------------------

// Initialize path state from primary hits
__global__ void initializePathStateKernel(Vec3* throughput,
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
            // Initialize throughput to 1.0 (albedo is in BRDF)
            sampleThroughput = Vec3(1.0f, 1.0f, 1.0f);
            isActive = 1;
        } else {
            Vec3 envLight = sampleEnvironmentWithClamp(env, primaryRay.direction, params.maxRadiance);
            sampleRadiance = envLight;
        }

        throughput[sampleIdx] = sampleThroughput;
        radiance[sampleIdx] = sampleRadiance;
        active[sampleIdx] = isActive;
    }
}

// Sample bounce directions using Disney BRDF (shared kernel)
__global__ void sampleBounceDirectionsKernel(const float* hitPositions,
                                             const float* hitNormals,
                                             const float* hitColors,
                                             const float* hitMaterialParams,
                                             const int* hitFlags,
                                             int* pathActive,
                                             RenderParams params,
                                             float* bounceOrigins,      // Output
                                             float* bounceDirections,   // Output
                                             float* bouncePdfs,         // Output
                                             float* bounceBRDFs) {      // Output (f * cos / pdf)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        if (!hitFlags[sampleIdx] || (pathActive && !pathActive[sampleIdx])) {
            bounceOrigins[base + 0] = 0.0f;
            bounceOrigins[base + 1] = 0.0f;
            bounceOrigins[base + 2] = 0.0f;
            bounceDirections[base + 0] = 0.0f;
            bounceDirections[base + 1] = 0.0f;
            bounceDirections[base + 2] = 0.0f;
            bouncePdfs[sampleIdx] = 0.0f;
            bounceBRDFs[base + 0] = 0.0f;
            bounceBRDFs[base + 1] = 0.0f;
            bounceBRDFs[base + 2] = 0.0f;
            continue;
        }

        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);

        Vec3 hitPos(hitPositions[base + 0], hitPositions[base + 1], hitPositions[base + 2]);
        Vec3 normal(hitNormals[base + 0], hitNormals[base + 1], hitNormals[base + 2]);
        Vec3 albedo(hitColors[base + 0], hitColors[base + 1], hitColors[base + 2]);

        // Normalize normal
        float nlen = length(normal);
        if (nlen > 0.0f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }

        // Check for back-facing hit
        if (dot(normal, primaryRay.direction) > 0.0f) {
            if (pathActive) pathActive[sampleIdx] = 0;
            bouncePdfs[sampleIdx] = 0.0f;
            continue;
        }

        // Sample Disney BRDF for bounce direction
        Vec3 wo = primaryRay.direction * -1.0f;

        // Build tangent space
        Vec3 tangent, bitangent;
        buildTangentSpace(normal, &tangent, &bitangent);

        Vec3 materialParams(params.material.metallic.value, params.material.roughness.value, params.material.specular.value);
        if (hitMaterialParams) {
            materialParams.x = hitMaterialParams[base + 0];
            materialParams.y = hitMaterialParams[base + 1];
            materialParams.z = hitMaterialParams[base + 2];
        }

        // Create material with texture-modulated base color
        Material surfaceMat = params.material;
        surfaceMat.base_color = MaterialParamVec3::constant(albedo);
        surfaceMat.metallic = MaterialParam::constant(materialParams.x);
        surfaceMat.roughness = MaterialParam::constant(materialParams.y);
        surfaceMat.specular = MaterialParam::constant(materialParams.z);

        // Generate random numbers for BRDF sampling
        float u1 = rand01(rng);
        float u2 = rand01(rng);
        float u3 = rand01(rng);
        float pdf;
        // Use new 3-component sampler (diffuse, specular, clearcoat)
        Vec3 wi = disney_sample_3component(surfaceMat, normal, wo, tangent, bitangent,
                                           u1, u2, u3, &pdf);

        if (pdf <= 0.0f) {
            if (pathActive) pathActive[sampleIdx] = 0;
            bouncePdfs[sampleIdx] = 0.0f;
            continue;
        }

        // Evaluate Disney BRDF
        Vec3 f = disney_eval(surfaceMat, normal, wo, wi, tangent, bitangent);

        // Compute BRDF weight (f * cos / pdf)
        // Use fabs to match NBVH - handles numerical errors at grazing angles
        float cos_theta = fabsf(dot(normal, wi));
        Vec3 brdfWeight = f * (cos_theta / pdf);

        // Safety check for numerical issues
        if (isnan(brdfWeight.x) || isnan(brdfWeight.y) || isnan(brdfWeight.z) ||
            isinf(brdfWeight.x) || isinf(brdfWeight.y) || isinf(brdfWeight.z)) {
            if (pathActive) pathActive[sampleIdx] = 0;
            bouncePdfs[sampleIdx] = 0.0f;
            continue;
        }

        // Output bounce ray
        float rayOffset = params.sceneScale * 1e-6f;
        Vec3 origin = hitPos + normal * rayOffset;

        bounceOrigins[base + 0] = origin.x;
        bounceOrigins[base + 1] = origin.y;
        bounceOrigins[base + 2] = origin.z;
        bounceDirections[base + 0] = wi.x;
        bounceDirections[base + 1] = wi.y;
        bounceDirections[base + 2] = wi.z;
        bouncePdfs[sampleIdx] = pdf;
        bounceBRDFs[base + 0] = brdfWeight.x;
        bounceBRDFs[base + 1] = brdfWeight.y;
        bounceBRDFs[base + 2] = brdfWeight.z;
    }
}

// Trace bounce rays against GT mesh (GT mode only)
__global__ void traceGroundTruthBouncesKernel(const float* bounceOrigins,
                                              const float* bounceDirections,
                                              const float* bouncePdfs,
                                              MeshDeviceView mesh,
                                              RenderParams params,
                                              float* bounceHitPositions,    // Output
                                              float* bounceHitNormals,      // Output
                                              float* bounceHitColors,       // Output
                                              float* bounceMaterialParams,  // Output
                                              int* bounceHitFlags) {        // Output
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        if (bouncePdfs[sampleIdx] <= 0.0f) {
            bounceHitFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 origin(bounceOrigins[base + 0], bounceOrigins[base + 1], bounceOrigins[base + 2]);
        Vec3 direction(bounceDirections[base + 0], bounceDirections[base + 1], bounceDirections[base + 2]);
        Ray bounceRay(origin, direction);

        HitInfo hitInfo;
        bool hit = traceMesh(bounceRay, mesh, &hitInfo, true, params.material);

        if (hit) {
            bounceHitPositions[base + 0] = hitInfo.position.x;
            bounceHitPositions[base + 1] = hitInfo.position.y;
            bounceHitPositions[base + 2] = hitInfo.position.z;
            bounceHitNormals[base + 0] = hitInfo.shadingNormal.x;
            bounceHitNormals[base + 1] = hitInfo.shadingNormal.y;
            bounceHitNormals[base + 2] = hitInfo.shadingNormal.z;

            // Resolve material at hit point
            const Material* mat = &params.material;
            if (hitInfo.materialId >= 0 && hitInfo.materialId < mesh.numMaterials && mesh.materials) {
                mat = &mesh.materials[hitInfo.materialId];
            }
            ResolvedMaterial resolved = resolveMaterial(*mat, hitInfo.uv, mesh);

            bounceHitColors[base + 0] = resolved.base_color.x;
            bounceHitColors[base + 1] = resolved.base_color.y;
            bounceHitColors[base + 2] = resolved.base_color.z;
            bounceMaterialParams[base + 0] = resolved.metallic;
            bounceMaterialParams[base + 1] = resolved.roughness;
            bounceMaterialParams[base + 2] = resolved.specular;
            bounceHitFlags[sampleIdx] = 1;
        } else {
            bounceHitFlags[sampleIdx] = 0;
            bounceMaterialParams[base + 0] = params.material.metallic.value;
            bounceMaterialParams[base + 1] = params.material.roughness.value;
            bounceMaterialParams[base + 2] = params.material.specular.value;
        }
    }
}

// Integrate bounce results and update path state (shared kernel)
__global__ void integrateBounceKernel(Vec3* throughput,
                                      Vec3* radiance,
                                      int* active,
                                      const int* bounceHitFlags,
                                      const float* bounceDirections,
                                      const float* bounceBRDFs,
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

        int base = sampleIdx * 3;
        Vec3 brdfWeight(bounceBRDFs[base + 0], bounceBRDFs[base + 1], bounceBRDFs[base + 2]);
        Vec3 newThroughput = mul(throughput[sampleIdx], brdfWeight);
        throughput[sampleIdx] = newThroughput;

        if (!bounceHitFlags[sampleIdx]) {
            // Ray missed - sample environment and terminate
            Vec3 envDir(bounceDirections[base + 0], bounceDirections[base + 1], bounceDirections[base + 2]);
            Vec3 envLight = sampleEnvironmentWithClamp(env, envDir, params.maxRadiance);
            radiance[sampleIdx] = radiance[sampleIdx] + mul(newThroughput, envLight);
            active[sampleIdx] = 0;
            continue;
        }

        if (bounceIndex >= params.maxBounces) {
            active[sampleIdx] = 0;
            continue;
        }

        // Russian roulette path termination (after 3 bounces)
        if (bounceIndex > 3) {
            Vec3 tp = throughput[sampleIdx];
            float q = fmaxf(0.05f, 1.0f - fmaxf(tp.x, fmaxf(tp.y, tp.z)));
            uint32_t rng = initRng(pixelIdx, params.sampleOffset + bounceIndex, s);
            if (rand01(rng) < q) {
                active[sampleIdx] = 0;
                continue;
            }
            throughput[sampleIdx] = tp * (1.0f / (1.0f - q));
        }
    }
}

// ---------------------------------------------------------------------------
// Trace hybrid bounces: check both shells and additional mesh (two-box early culling)
// ---------------------------------------------------------------------------
__global__ void traceHybridBouncesKernel(
        const float* bounceOrigins,
        const float* bounceDirections,
        const float* bouncePdfs,
        MeshDeviceView outerShell,
        MeshDeviceView additionalMesh,
        BoundingBox shellsBounds,
        BoundingBox additionalBounds,
        RenderParams params,
        float* bouncePositions,
        float* bounceNormals,
        float* bounceColors,
        int* bounceHitFlags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        // Reconstruct bounce ray
        Vec3 origin(bounceOrigins[base+0], bounceOrigins[base+1], bounceOrigins[base+2]);
        Vec3 dir(bounceDirections[base+0], bounceDirections[base+1], bounceDirections[base+2]);
        Ray ray(origin, dir);

        HitInfo closestHit;
        MeshDeviceView hitMesh = outerShell;  // Track which mesh we hit for material resolution
        float closestT = 1e30f;

        // If additional mesh is empty, only trace shells
        if (additionalMesh.numTriangles == 0) {
            traceMesh(ray, outerShell, &closestHit, true, params.material);
        } else {
            // Two-box early culling
            Vec3 invDir(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);
            float tNearShells = 0.0f, tNearAdditional = 0.0f;
            bool hitShellsBox = intersectAabb(ray, invDir, shellsBounds.min, shellsBounds.max, 1e30f, &tNearShells);
            bool hitAdditionalBox = intersectAabb(ray, invDir, additionalBounds.min, additionalBounds.max, 1e30f, &tNearAdditional);

            // Trace in near-to-far order with early exit
            if (hitShellsBox && hitAdditionalBox) {
                if (tNearShells < tNearAdditional) {
                    // Check shells first
                    HitInfo shellHit;
                    if (traceMesh(ray, outerShell, &shellHit, true, params.material) && shellHit.t < closestT) {
                        closestHit = shellHit;
                        closestT = shellHit.t;
                        hitMesh = outerShell;
                    }
                    // Only check additional if it could be closer
                    if (tNearAdditional < closestT) {
                        HitInfo addHit;
                        if (traceMesh(ray, additionalMesh, &addHit, true, params.material) && addHit.t < closestT) {
                            closestHit = addHit;
                            hitMesh = additionalMesh;
                        }
                    }
                } else {
                    // Check additional first, then shells
                    HitInfo addHit;
                    if (traceMesh(ray, additionalMesh, &addHit, true, params.material) && addHit.t < closestT) {
                        closestHit = addHit;
                        closestT = addHit.t;
                        hitMesh = additionalMesh;
                    }
                    if (tNearShells < closestT) {
                        HitInfo shellHit;
                        if (traceMesh(ray, outerShell, &shellHit, true, params.material) && shellHit.t < closestT) {
                            closestHit = shellHit;
                            hitMesh = outerShell;
                        }
                    }
                }
            } else if (hitShellsBox) {
                traceMesh(ray, outerShell, &closestHit, true, params.material);
            } else if (hitAdditionalBox) {
                if (traceMesh(ray, additionalMesh, &closestHit, true, params.material)) {
                    hitMesh = additionalMesh;
                }
            }
        }

        // Populate output
        if (closestHit.valid()) {
            bouncePositions[base+0] = closestHit.position.x;
            bouncePositions[base+1] = closestHit.position.y;
            bouncePositions[base+2] = closestHit.position.z;
            bounceNormals[base+0] = closestHit.shadingNormal.x;
            bounceNormals[base+1] = closestHit.shadingNormal.y;
            bounceNormals[base+2] = closestHit.shadingNormal.z;

            // Resolve material
            const Material* mat = &params.material;
            if (closestHit.materialId >= 0 && closestHit.materialId < hitMesh.numMaterials && hitMesh.materials) {
                mat = &hitMesh.materials[closestHit.materialId];
            }
            ResolvedMaterial resolved = resolveMaterial(*mat, closestHit.uv, hitMesh);
            bounceColors[base+0] = resolved.base_color.x;
            bounceColors[base+1] = resolved.base_color.y;
            bounceColors[base+2] = resolved.base_color.z;
            bounceHitFlags[sampleIdx] = 1;
        } else {
            bounceHitFlags[sampleIdx] = 0;
        }
    }
}

// Finalize and output (shared kernel)
__global__ void finalizePathTracingKernel(uchar4* output,
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
// Multi-segment constants.
// ---------------------------------------------------------------------------
constexpr int kMaxSegmentIterations = 10;
constexpr float kSegmentEpsilon = 1e-10f;

// ---------------------------------------------------------------------------
// Initial outer shell entry tracing for multi-segment method.
// ---------------------------------------------------------------------------
__global__ void traceOuterShellEntryKernel(
        float* entryPositions,
        float* entryT,
        float* rayDirections,
        int* activeFlags,
        float* accumT,
        RenderParams params,
        MeshDeviceView outerShell) {
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

        // Store ray direction
        rayDirections[base + 0] = ray.direction.x;
        rayDirections[base + 1] = ray.direction.y;
        rayDirections[base + 2] = ray.direction.z;

        // Trace outer shell entry (FORWARD_ONLY: allow_backward=false, allow_forward=true)
        HitInfo outerHit;
        bool hitOuter = traceMeshWithMode(ray, outerShell, &outerHit, TraceMode::FORWARD_ONLY, false, params.material);

        if (hitOuter) {
            Vec3 entryPos = ray.at(outerHit.t);
            entryPositions[base + 0] = entryPos.x;
            entryPositions[base + 1] = entryPos.y;
            entryPositions[base + 2] = entryPos.z;
            entryT[sampleIdx] = outerHit.t;
            activeFlags[sampleIdx] = 1;
            accumT[sampleIdx] = outerHit.t;
        } else {
            entryPositions[base + 0] = 0.0f;
            entryPositions[base + 1] = 0.0f;
            entryPositions[base + 2] = 0.0f;
            entryT[sampleIdx] = 0.0f;
            activeFlags[sampleIdx] = 0;
            accumT[sampleIdx] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Initial outer shell entry tracing for arbitrary rays.
// Handles rays that start inside the outer shell by exiting first.
// ---------------------------------------------------------------------------
__global__ void traceOuterShellEntryFromRaysKernel(
        const float* rayOrigins,
        const float* rayDirections,
        const int* rayActiveMask,
        const float* rayPdfs,
        float* entryPositions,
        float* entryT,
        int* activeFlags,
        float* accumT,
        RenderParams params,
        MeshDeviceView outerShell) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        if (rayActiveMask && !rayActiveMask[sampleIdx]) {
            entryPositions[base + 0] = 0.0f;
            entryPositions[base + 1] = 0.0f;
            entryPositions[base + 2] = 0.0f;
            entryT[sampleIdx] = 0.0f;
            activeFlags[sampleIdx] = 0;
            accumT[sampleIdx] = 0.0f;
            continue;
        }
        if (rayPdfs && rayPdfs[sampleIdx] <= 0.0f) {
            entryPositions[base + 0] = 0.0f;
            entryPositions[base + 1] = 0.0f;
            entryPositions[base + 2] = 0.0f;
            entryT[sampleIdx] = 0.0f;
            activeFlags[sampleIdx] = 0;
            accumT[sampleIdx] = 0.0f;
            continue;
        }

        Vec3 origin(rayOrigins[base + 0], rayOrigins[base + 1], rayOrigins[base + 2]);
        Vec3 dir(rayDirections[base + 0], rayDirections[base + 1], rayDirections[base + 2]);
        Ray ray(origin, dir);

        HitInfo outerHit;
        bool hitOuter = traceMeshWithMode(ray, outerShell, &outerHit, TraceMode::FORWARD_ONLY, false, params.material);

        float baseOffset = 0.0f;
        Vec3 entryOrigin = origin;
        if (!hitOuter) {
            HitInfo exitHit;
            bool hitExit = traceMeshWithMode(ray, outerShell, &exitHit, TraceMode::BACKWARD_ONLY, false, params.material);
            if (hitExit) {
                baseOffset = exitHit.t + kSegmentEpsilon;
                entryOrigin = origin + dir * baseOffset;
                Ray shiftedRay(entryOrigin, dir);
                hitOuter = traceMeshWithMode(shiftedRay, outerShell, &outerHit, TraceMode::FORWARD_ONLY, false, params.material);
            }
        }

        if (hitOuter) {
            Vec3 entryPos = entryOrigin + dir * outerHit.t;
            float totalEntryT = baseOffset + outerHit.t;
            entryPositions[base + 0] = entryPos.x;
            entryPositions[base + 1] = entryPos.y;
            entryPositions[base + 2] = entryPos.z;
            entryT[sampleIdx] = totalEntryT;
            activeFlags[sampleIdx] = 1;
            accumT[sampleIdx] = totalEntryT;
        } else {
            entryPositions[base + 0] = 0.0f;
            entryPositions[base + 1] = 0.0f;
            entryPositions[base + 2] = 0.0f;
            entryT[sampleIdx] = 0.0f;
            activeFlags[sampleIdx] = 0;
            accumT[sampleIdx] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Trace segment exits: outer shell backward and inner shell forward.
// Computes exit position as min(outer_exit, inner_enter).
// ---------------------------------------------------------------------------
__global__ void traceSegmentExitsKernel(
        const float* entryPositions,
        const float* rayDirections,
        const int* hitIndices,
        int hitCount,
        MeshDeviceView outerShell,
        MeshDeviceView innerShell,
        Material material,
        float* exitPositions,
        float* outerExitT,
        float* innerEnterT,
        int* innerHitFlags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int sampleIdx = hitIndices[idx];
    int base = sampleIdx * 3;

    Vec3 entryPos(entryPositions[base + 0],
                  entryPositions[base + 1],
                  entryPositions[base + 2]);
    Vec3 dir(rayDirections[base + 0],
             rayDirections[base + 1],
             rayDirections[base + 2]);

    // Shift entry point into shell by epsilon (matching Python: x_outer_enter + ds_left * eps)
    Vec3 shiftedEntry = entryPos + dir * kSegmentEpsilon;
    Ray ray(shiftedEntry, dir);

    // Trace outer shell EXIT (BACKWARD_ONLY: allow_backward=true, allow_forward=false)
    HitInfo outerExit;
    bool hitOuterExit = traceMeshWithMode(ray, outerShell, &outerExit, TraceMode::BACKWARD_ONLY, false, material);

    float exitT;
    if (hitOuterExit) {
        exitT = outerExit.t;
    } else {
        // Fallback: minimal segment (matching Python: x_outer_exit_t[~mask] = 1e-8)
        exitT = kSegmentEpsilon;
    }
    outerExitT[sampleIdx] = exitT;

    // Trace inner shell (ANY mode, allow_negative=True as in Python)
    HitInfo innerHit;
    bool hitInner = traceMeshWithMode(ray, innerShell, &innerHit, TraceMode::ANY, true, material);

    float innerT;
    if (hitInner) {
        innerT = innerHit.t;
        innerHitFlags[sampleIdx] = 1;
    } else {
        innerT = 1e30f;  // No inner hit
        innerHitFlags[sampleIdx] = 0;
    }
    innerEnterT[sampleIdx] = innerT;

    // Exit position is the nearer of outer_exit or inner_enter
    // (matching Python: inner_apply = x_inner_mask & (x_inner_t < x_outer_exit_t))
    // Note: Python allows negative inner_t (allow_negative=True) so we don't check > 0
    bool innerBeforeOuter = hitInner && (innerT < exitT);
    Vec3 exitPos;
    if (innerBeforeOuter) {
        exitPos = shiftedEntry + dir * innerT;
    } else {
        exitPos = shiftedEntry + dir * exitT;
    }

    exitPositions[base + 0] = exitPos.x;
    exitPositions[base + 1] = exitPos.y;
    exitPositions[base + 2] = exitPos.z;
}

// ---------------------------------------------------------------------------
// Build normalized neural network inputs for current segment.
// ---------------------------------------------------------------------------
__global__ void buildSegmentNeuralInputsKernel(
        const float* entryPositions,
        const float* exitPositions,
        const float* rayDirections,
        const int* hitIndices,
        int hitCount,
        Vec3 outerMin,
        Vec3 outerInvExtent,
        float* compactedPointInputs,
        float* compactedDirs,
        int pointCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int sampleIdx = hitIndices[idx];
    int srcBase = sampleIdx * 3;
    int dstBase = idx * pointCount * 3;

    // Shift entry by epsilon before normalization (matching Python)
    Vec3 entryPos(entryPositions[srcBase + 0],
                  entryPositions[srcBase + 1],
                  entryPositions[srcBase + 2]);
    Vec3 dir(rayDirections[srcBase + 0],
             rayDirections[srcBase + 1],
             rayDirections[srcBase + 2]);
    Vec3 shiftedEntry = entryPos + dir * kSegmentEpsilon;

    Vec3 exitPos(exitPositions[srcBase + 0],
                 exitPositions[srcBase + 1],
                 exitPositions[srcBase + 2]);

    // Normalize positions w.r.t. outer shell bounds
    Vec3 normEntry((shiftedEntry.x - outerMin.x) * outerInvExtent.x,
                   (shiftedEntry.y - outerMin.y) * outerInvExtent.y,
                   (shiftedEntry.z - outerMin.z) * outerInvExtent.z);
    Vec3 normExit((exitPos.x - outerMin.x) * outerInvExtent.x,
                  (exitPos.y - outerMin.y) * outerInvExtent.y,
                  (exitPos.z - outerMin.z) * outerInvExtent.z);

    int offset = 0;
    compactedPointInputs[dstBase + offset++] = normEntry.x;
    compactedPointInputs[dstBase + offset++] = normEntry.y;
    compactedPointInputs[dstBase + offset++] = normEntry.z;
    compactedPointInputs[dstBase + offset++] = normExit.x;
    compactedPointInputs[dstBase + offset++] = normExit.y;
    compactedPointInputs[dstBase + offset++] = normExit.z;
    if (pointCount == 3) {
        Vec3 midPos = (shiftedEntry + exitPos) * 0.5f;
        Vec3 normMid((midPos.x - outerMin.x) * outerInvExtent.x,
                     (midPos.y - outerMin.y) * outerInvExtent.y,
                     (midPos.z - outerMin.z) * outerInvExtent.z);
        compactedPointInputs[dstBase + offset++] = normMid.x;
        compactedPointInputs[dstBase + offset++] = normMid.y;
        compactedPointInputs[dstBase + offset++] = normMid.z;
    }

    // Directions: map from [-1,1] to [0,1] (matching Python: (directions + 1) / 2)
    compactedDirs[dstBase + 0] = (dir.x + 1.0f) * 0.5f;
    compactedDirs[dstBase + 1] = (dir.y + 1.0f) * 0.5f;
    compactedDirs[dstBase + 2] = (dir.z + 1.0f) * 0.5f;
}

// ---------------------------------------------------------------------------
// Apply neural network outputs and update ray state.
// Handles: intersection found, inner shell forces intersection.
// ---------------------------------------------------------------------------
__global__ void applySegmentNeuralOutputKernel(
        const __half* outputs,
        int outputStride,
        const int* hitIndices,
        int hitCount,
        const float* entryPositions,
        const float* rayDirections,
        const float* accumT,
        const int* innerHitFlags,
        const float* innerEnterT,
        const float* outerExitT,
        float* hitPositions,
        float* hitNormals,
        float* hitColors,
        float* hitMaterialParams,
        int* hitFlags,
        int* activeFlags,
        Material material) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int sampleIdx = hitIndices[idx];
    int base = sampleIdx * 3;
    int outBase = idx * outputStride;

    float hasIntersection = __half2float(outputs[outBase + 0]);
    float distance = __half2float(outputs[outBase + 1]);
    float nx = __half2float(outputs[outBase + 2]);
    float ny = __half2float(outputs[outBase + 3]);
    float nz = __half2float(outputs[outBase + 4]);
    float cr = __half2float(outputs[outBase + 5]);
    float cg = __half2float(outputs[outBase + 6]);
    float cb = __half2float(outputs[outBase + 7]);

    Vec3 entryPos(entryPositions[base + 0],
                  entryPositions[base + 1],
                  entryPositions[base + 2]);
    Vec3 dir(rayDirections[base + 0],
             rayDirections[base + 1],
             rayDirections[base + 2]);

    // Check if neural network predicts intersection
    // (matching Python: pred_intersection_mask = (pred_intersection >= 0))
    bool neuralHit = (hasIntersection >= 0.0f);

    // Check if inner shell hit before outer exit - forces intersection
    // (matching Python: pred_intersection_mask[x_inner_mask & (x_inner_t < x_outer_exit_t)] = True)
    // Note: Python allows negative inner_t (allow_negative=True) so we don't check > 0
    bool innerHitBeforeExit = (innerHitFlags[sampleIdx] != 0) &&
                               (innerEnterT[sampleIdx] < outerExitT[sampleIdx]);

    bool foundIntersection = neuralHit || innerHitBeforeExit;

    if (foundIntersection) {
        // Compute final hit position
        // The entry position is at accumT from camera, plus predicted distance from entry
        // (matching Python: pred_t_global = pred_t + accum_t)
        // But note: the neural network predicts distance from shifted entry point
        Vec3 shiftedEntry = entryPos + dir * kSegmentEpsilon;
        Vec3 hitPos = shiftedEntry + dir * distance * 0;

        hitPositions[base + 0] = hitPos.x;
        hitPositions[base + 1] = hitPos.y;
        hitPositions[base + 2] = hitPos.z;

        // Normalize and store normal
        Vec3 normal(nx, ny, nz);
        float nlen = length(normal);
        if (nlen > 1e-6f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        hitNormals[base + 0] = normal.x;
        hitNormals[base + 1] = normal.y;
        hitNormals[base + 2] = normal.z;

        Vec3 fallbackColor(material.base_color.value.x,
                           material.base_color.value.y,
                           material.base_color.value.z);
        Vec3 neuralColor = fallbackColor;
        if (outputStride >= 8) {
            neuralColor.x = saturate(cr);
            neuralColor.y = saturate(cg);
            neuralColor.z = saturate(cb);
        }
        hitColors[base + 0] = neuralColor.x;
        hitColors[base + 1] = neuralColor.y;
        hitColors[base + 2] = neuralColor.z;
        if (hitMaterialParams) {
            hitMaterialParams[base + 0] = material.metallic.value;
            hitMaterialParams[base + 1] = material.roughness.value;
            hitMaterialParams[base + 2] = material.specular.value;
        }

        hitFlags[sampleIdx] = 1;
        activeFlags[sampleIdx] = 0;  // Ray done
    }
    // If no intersection found, activeFlags remains 1 (will be updated by prepareNextIterationKernel)
}

// ---------------------------------------------------------------------------
// Trace additional mesh for all primary rays (hybrid rendering)
// ---------------------------------------------------------------------------
__global__ void traceAdditionalMeshPrimaryRaysKernel(
        float* hitPositions,
        float* hitNormals,
        float* hitColors,
        float* hitMaterialParams,
        int* hitFlags,
        RenderParams params,
        MeshDeviceView additionalMesh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (additionalMesh.numTriangles == 0) {
            hitFlags[sampleIdx] = 0;
            continue;
        }

        // Reconstruct primary ray (same as shell tracing)
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray ray = generatePrimaryRay(x, y, params, rng);

        // Trace against additional mesh
        HitInfo hit;
        if (traceMesh(ray, additionalMesh, &hit, true, params.material)) {
            int base = sampleIdx * 3;
            hitPositions[base+0] = hit.position.x;
            hitPositions[base+1] = hit.position.y;
            hitPositions[base+2] = hit.position.z;
            hitNormals[base+0] = hit.shadingNormal.x;
            hitNormals[base+1] = hit.shadingNormal.y;
            hitNormals[base+2] = hit.shadingNormal.z;

            // Resolve material
            const Material* mat = &params.material;
            if (hit.materialId >= 0 && hit.materialId < additionalMesh.numMaterials && additionalMesh.materials) {
                mat = &additionalMesh.materials[hit.materialId];
            }
            ResolvedMaterial resolved = resolveMaterial(*mat, hit.uv, additionalMesh);
            hitColors[base+0] = resolved.base_color.x;
            hitColors[base+1] = resolved.base_color.y;
            hitColors[base+2] = resolved.base_color.z;
            hitFlags[sampleIdx] = 1;
            if (hitMaterialParams) {
                hitMaterialParams[base+0] = resolved.metallic;
                hitMaterialParams[base+1] = resolved.roughness;
                hitMaterialParams[base+2] = resolved.specular;
            }
        } else {
            hitFlags[sampleIdx] = 0;
            if (hitMaterialParams) {
                hitMaterialParams[base+0] = params.material.metallic.value;
                hitMaterialParams[base+1] = params.material.roughness.value;
                hitMaterialParams[base+2] = params.material.specular.value;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trace additional mesh for arbitrary rays (hybrid rendering).
// ---------------------------------------------------------------------------
__global__ void traceAdditionalMeshRaysKernel(
        const float* rayOrigins,
        const float* rayDirections,
        const float* rayPdfs,
        float* hitPositions,
        float* hitNormals,
        float* hitColors,
        float* hitMaterialParams,
        int* hitFlags,
        RenderParams params,
        MeshDeviceView additionalMesh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        if (additionalMesh.numTriangles == 0) {
            hitFlags[sampleIdx] = 0;
            continue;
        }

        if (rayPdfs && rayPdfs[sampleIdx] <= 0.0f) {
            hitFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 origin(rayOrigins[base + 0], rayOrigins[base + 1], rayOrigins[base + 2]);
        Vec3 dir(rayDirections[base + 0], rayDirections[base + 1], rayDirections[base + 2]);
        Ray ray(origin, dir);

        HitInfo hit;
        if (traceMesh(ray, additionalMesh, &hit, true, params.material)) {
            hitPositions[base + 0] = hit.position.x;
            hitPositions[base + 1] = hit.position.y;
            hitPositions[base + 2] = hit.position.z;
            hitNormals[base + 0] = hit.shadingNormal.x;
            hitNormals[base + 1] = hit.shadingNormal.y;
            hitNormals[base + 2] = hit.shadingNormal.z;

            // Resolve material
            const Material* mat = &params.material;
            if (hit.materialId >= 0 && hit.materialId < additionalMesh.numMaterials && additionalMesh.materials) {
                mat = &additionalMesh.materials[hit.materialId];
            }
            ResolvedMaterial resolved = resolveMaterial(*mat, hit.uv, additionalMesh);
            hitColors[base + 0] = resolved.base_color.x;
            hitColors[base + 1] = resolved.base_color.y;
            hitColors[base + 2] = resolved.base_color.z;
            hitFlags[sampleIdx] = 1;
            if (hitMaterialParams) {
                hitMaterialParams[base + 0] = resolved.metallic;
                hitMaterialParams[base + 1] = resolved.roughness;
                hitMaterialParams[base + 2] = resolved.specular;
            }
        } else {
            hitFlags[sampleIdx] = 0;
            if (hitMaterialParams) {
                hitMaterialParams[base + 0] = params.material.metallic.value;
                hitMaterialParams[base + 1] = params.material.roughness.value;
                hitMaterialParams[base + 2] = params.material.specular.value;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Select closest hit between shells and additional mesh (hybrid rendering)
// ---------------------------------------------------------------------------
__global__ void selectClosestPrimaryHitKernel(
        float* shellHitPositions,
        float* shellHitNormals,
        float* shellHitColors,
        float* shellHitMaterialParams,
        int* shellHitFlags,
        const float* additionalHitPositions,
        const float* additionalHitNormals,
        const float* additionalHitColors,
        const float* additionalHitMaterialParams,
        const int* additionalHitFlags,
        RenderParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        bool shellHit = shellHitFlags[sampleIdx] != 0;
        bool additionalHit = additionalHitFlags[sampleIdx] != 0;

        if (shellHit && additionalHit) {
            // Both hit - compute distances and use closer
            Vec3 rayOrigin = params.camPos;
            Vec3 shellPos(shellHitPositions[base+0], shellHitPositions[base+1], shellHitPositions[base+2]);
            Vec3 additionalPos(additionalHitPositions[base+0], additionalHitPositions[base+1], additionalHitPositions[base+2]);

            float shellDist = length(shellPos - rayOrigin);
            float additionalDist = length(additionalPos - rayOrigin);

            if (additionalDist < shellDist) {
                // Use additional mesh hit
                shellHitPositions[base+0] = additionalHitPositions[base+0];
                shellHitPositions[base+1] = additionalHitPositions[base+1];
                shellHitPositions[base+2] = additionalHitPositions[base+2];
                shellHitNormals[base+0] = additionalHitNormals[base+0];
                shellHitNormals[base+1] = additionalHitNormals[base+1];
                shellHitNormals[base+2] = additionalHitNormals[base+2];
                shellHitColors[base+0] = additionalHitColors[base+0];
                shellHitColors[base+1] = additionalHitColors[base+1];
                shellHitColors[base+2] = additionalHitColors[base+2];
                if (shellHitMaterialParams && additionalHitMaterialParams) {
                    shellHitMaterialParams[base+0] = additionalHitMaterialParams[base+0];
                    shellHitMaterialParams[base+1] = additionalHitMaterialParams[base+1];
                    shellHitMaterialParams[base+2] = additionalHitMaterialParams[base+2];
                }
            }
            // Else keep shell hit
        } else if (additionalHit) {
            // Only additional mesh hit
            shellHitPositions[base+0] = additionalHitPositions[base+0];
            shellHitPositions[base+1] = additionalHitPositions[base+1];
            shellHitPositions[base+2] = additionalHitPositions[base+2];
            shellHitNormals[base+0] = additionalHitNormals[base+0];
            shellHitNormals[base+1] = additionalHitNormals[base+1];
            shellHitNormals[base+2] = additionalHitNormals[base+2];
            shellHitColors[base+0] = additionalHitColors[base+0];
            shellHitColors[base+1] = additionalHitColors[base+1];
            shellHitColors[base+2] = additionalHitColors[base+2];
            if (shellHitMaterialParams && additionalHitMaterialParams) {
                shellHitMaterialParams[base+0] = additionalHitMaterialParams[base+0];
                shellHitMaterialParams[base+1] = additionalHitMaterialParams[base+1];
                shellHitMaterialParams[base+2] = additionalHitMaterialParams[base+2];
            }
            shellHitFlags[sampleIdx] = 1;
        }
        // Else keep shell hit or miss
    }
}

// ---------------------------------------------------------------------------
// Select closest hit between shells and additional mesh (per-ray origin).
// ---------------------------------------------------------------------------
__global__ void selectClosestHitKernel(
        float* shellHitPositions,
        float* shellHitNormals,
        float* shellHitColors,
        float* shellHitMaterialParams,
        int* shellHitFlags,
        const float* additionalHitPositions,
        const float* additionalHitNormals,
        const float* additionalHitColors,
        const float* additionalHitMaterialParams,
        const int* additionalHitFlags,
        const float* rayOrigins,
        RenderParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;

    int pixelIdx = y * params.width + x;
    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;

        bool shellHit = shellHitFlags[sampleIdx] != 0;
        bool additionalHit = additionalHitFlags[sampleIdx] != 0;

        if (shellHit && additionalHit) {
            Vec3 rayOrigin(rayOrigins[base + 0], rayOrigins[base + 1], rayOrigins[base + 2]);
            Vec3 shellPos(shellHitPositions[base + 0], shellHitPositions[base + 1], shellHitPositions[base + 2]);
            Vec3 additionalPos(additionalHitPositions[base + 0], additionalHitPositions[base + 1], additionalHitPositions[base + 2]);

            float shellDist = length(shellPos - rayOrigin);
            float additionalDist = length(additionalPos - rayOrigin);

            if (additionalDist < shellDist) {
                shellHitPositions[base + 0] = additionalHitPositions[base + 0];
                shellHitPositions[base + 1] = additionalHitPositions[base + 1];
                shellHitPositions[base + 2] = additionalHitPositions[base + 2];
                shellHitNormals[base + 0] = additionalHitNormals[base + 0];
                shellHitNormals[base + 1] = additionalHitNormals[base + 1];
                shellHitNormals[base + 2] = additionalHitNormals[base + 2];
                shellHitColors[base + 0] = additionalHitColors[base + 0];
                shellHitColors[base + 1] = additionalHitColors[base + 1];
                shellHitColors[base + 2] = additionalHitColors[base + 2];
                if (shellHitMaterialParams && additionalHitMaterialParams) {
                    shellHitMaterialParams[base + 0] = additionalHitMaterialParams[base + 0];
                    shellHitMaterialParams[base + 1] = additionalHitMaterialParams[base + 1];
                    shellHitMaterialParams[base + 2] = additionalHitMaterialParams[base + 2];
                }
            }
        } else if (additionalHit) {
            shellHitPositions[base + 0] = additionalHitPositions[base + 0];
            shellHitPositions[base + 1] = additionalHitPositions[base + 1];
            shellHitPositions[base + 2] = additionalHitPositions[base + 2];
            shellHitNormals[base + 0] = additionalHitNormals[base + 0];
            shellHitNormals[base + 1] = additionalHitNormals[base + 1];
            shellHitNormals[base + 2] = additionalHitNormals[base + 2];
            shellHitColors[base + 0] = additionalHitColors[base + 0];
            shellHitColors[base + 1] = additionalHitColors[base + 1];
            shellHitColors[base + 2] = additionalHitColors[base + 2];
            if (shellHitMaterialParams && additionalHitMaterialParams) {
                shellHitMaterialParams[base + 0] = additionalHitMaterialParams[base + 0];
                shellHitMaterialParams[base + 1] = additionalHitMaterialParams[base + 1];
                shellHitMaterialParams[base + 2] = additionalHitMaterialParams[base + 2];
            }
            shellHitFlags[sampleIdx] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Prepare for next iteration: trace outer shell re-entry and update state.
// Implements: mask_for_remaining_rays = ~pred_intersection_mask & (x_outer_enter_mask_new | x_inner_mask)
// ---------------------------------------------------------------------------
__global__ void prepareNextIterationKernel(
        const float* exitPositions,
        const float* outerExitT,
        const float* rayDirections,
        const int* innerHitFlags,
        const int* hitIndices,
        int hitCount,
        MeshDeviceView outerShell,
        Material material,
        float* entryPositions,
        int* activeFlags,
        float* accumT,
        float* newEntryT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int sampleIdx = hitIndices[idx];

    // Skip if ray already found intersection (activeFlags was set to 0)
    if (activeFlags[sampleIdx] == 0) {
        return;
    }

    int base = sampleIdx * 3;
    Vec3 exitPos(exitPositions[base + 0],
                 exitPositions[base + 1],
                 exitPositions[base + 2]);
    Vec3 dir(rayDirections[base + 0],
             rayDirections[base + 1],
             rayDirections[base + 2]);

    // Shift exit point outward by epsilon (matching Python: x_outer_exit + ds_left * eps)
    Vec3 shiftedExit = exitPos + dir * kSegmentEpsilon;
    Ray ray(shiftedExit, dir);

    // Trace outer shell for re-entry (FORWARD_ONLY)
    HitInfo reentry;
    bool hitReentry = traceMeshWithMode(ray, outerShell, &reentry, TraceMode::FORWARD_ONLY, false, material);

    // Remaining rays condition (matching Python):
    // mask_for_remaining_rays = ~pred_intersection_mask & (x_outer_enter_mask_new | x_inner_mask)
    // At this point, activeFlags[sampleIdx] == 1 means no intersection was found
    // So we check: can re-enter outer shell OR hit inner shell
    bool canContinue = hitReentry || (innerHitFlags[sampleIdx] != 0);

    if (canContinue) {
        // Get re-entry distance (0 if no re-entry found, matching Python behavior
        // where x_outer_enter_t_new may be 0 or invalid for non-hits)
        float reentryDist = hitReentry ? reentry.t : 0.0f;

        // Update entry position for next iteration
        // (matching Python: x_outer_enter_new = x_outer_exit + ds_left * x_outer_enter_t_new)
        Vec3 newEntry = shiftedExit + dir * reentryDist;
        entryPositions[base + 0] = newEntry.x;
        entryPositions[base + 1] = newEntry.y;
        entryPositions[base + 2] = newEntry.z;

        // Update accumulated distance
        // (matching Python: accum_t = accum_t + x_outer_exit_t + x_outer_enter_t_new)
        accumT[sampleIdx] = accumT[sampleIdx] + outerExitT[sampleIdx] + reentryDist + 2.0f * kSegmentEpsilon;
        newEntryT[sampleIdx] = reentryDist;
        activeFlags[sampleIdx] = 1;
    } else {
        // Ray escapes - no more segments (no re-entry AND no inner shell hit)
        activeFlags[sampleIdx] = 0;
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
// Scatter point encoding outputs directly into the MLP input buffer.
// ---------------------------------------------------------------------------
__global__ void scatterPointEncodingsKernel(const __half* encOutput,
                                            uint32_t pointEncOutDims,
                                            uint32_t pointCount,
                                            float* mlpInput,
                                            uint32_t pointFeatureDims,
                                            uint32_t mlpInputDims,
                                            int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int mlpBase = idx * mlpInputDims;
    int encBase = idx * pointCount * pointEncOutDims;
    uint32_t offset = 0;
    for (uint32_t p = 0; p < pointCount; ++p) {
        int pointBase = encBase + p * pointEncOutDims;
        for (uint32_t d = 0; d < pointEncOutDims; ++d) {
            mlpInput[mlpBase + offset] = __half2float(encOutput[pointBase + d]);
            ++offset;
        }
    }
}

// ---------------------------------------------------------------------------
// Append direction encoding outputs after the point features in the MLP buffer.
// ---------------------------------------------------------------------------
__global__ void appendDirectionEncodingsKernel(const __half* dirEncOutput,
                                               uint32_t dirEncOutDims,
                                               float* mlpInput,
                                               uint32_t pointFeatureDims,
                                               uint32_t mlpInputDims,
                                               int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int mlpBase = idx * mlpInputDims + pointFeatureDims;
    int dirBase = idx * dirEncOutDims;
    for (uint32_t d = 0; d < dirEncOutDims; ++d) {
        mlpInput[mlpBase + d] = __half2float(dirEncOutput[dirBase + d]);
    }
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

RendererNeural::RendererNeural(Scene& scene, const NeuralNetworkConfig* nnConfig)
        : scene_(&scene),
          lightDir_(normalize(Vec3(1.0f, 1.5f, -1.0f))),
          useMidpointEncoding_(nnConfig ? nnConfig->use_midpoint_encoding : false) {
    if (!tcnn::cpp::has_networks()) {
        std::fprintf(stderr, "tiny-cuda-nn was built without network support.\n");
        return;
    }

    int log2HashmapSize = 14;
    if (nnConfig != nullptr) {
        log2HashmapSize = nnConfig->log2_hashmap_size;
    }

    // Point encoding: HashGrid matching Python config.
    tcnn::cpp::json pointEncConfig = {
            {"otype", "HashGrid"},
            {"n_levels", 8},
            {"n_features_per_level", 4},
            {"log2_hashmap_size", log2HashmapSize},
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

    uint32_t pointEncCount = useMidpointEncoding_ ? 3u : 2u;
    mlpInputDims_ = pointEncOutDims_ * pointEncCount + dirEncOutDims_;
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

void RendererNeural::traceNeuralSegmentsForRays(bool useCameraRays,
                                                const float* rayOrigins,
                                                const float* rayDirections,
                                                const int* rayActiveMask,
                                                const float* rayPdfs,
                                                size_t elementCount,
                                                const RenderParams& params,
                                                const MeshDeviceView& outerView,
                                                const MeshDeviceView& innerView,
                                                Vec3 outerMin,
                                                Vec3 outerInvExtent,
                                                float* outHitPositions,
                                                float* outHitNormals,
                                                float* outHitColors,
                                                float* outHitMaterialParams,
                                                int* outHitFlags) {
    if (elementCount == 0) {
        return;
    }

    dim3 block(8, 8);
    dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

    int elementCountInt = static_cast<int>(elementCount);
    const int compactBlock = 256;
    int compactGrid = static_cast<int>((elementCount + compactBlock - 1) / compactBlock);

    checkCuda(cudaMemset(outHitFlags, 0, elementCount * sizeof(int)), "cudaMemset hitFlags");
    checkCuda(cudaMemset(outHitPositions, 0, elementCount * 3 * sizeof(float)), "cudaMemset hitPositions");
    checkCuda(cudaMemset(outHitNormals, 0, elementCount * 3 * sizeof(float)), "cudaMemset hitNormals");
    checkCuda(cudaMemset(outHitColors, 0, elementCount * 3 * sizeof(float)), "cudaMemset hitColors");
    if (outHitMaterialParams) {
        checkCuda(cudaMemset(outHitMaterialParams, 0, elementCount * 3 * sizeof(float)), "cudaMemset hitMaterialParams");
    }

    const float* segmentRayDirections = nullptr;
    if (useCameraRays) {
        traceOuterShellEntryKernel<<<grid, block>>>(
                currentEntryPos_,
                outerExitT_,
                rayDirections_,
                rayActiveFlags_,
                accumT_,
                params,
                outerView);
        checkCuda(cudaGetLastError(), "traceOuterShellEntryKernel launch");
        segmentRayDirections = rayDirections_;
    } else {
        if (!rayOrigins || !rayDirections) {
            return;
        }
        traceOuterShellEntryFromRaysKernel<<<grid, block>>>(
                rayOrigins,
                rayDirections,
                rayActiveMask,
                rayPdfs,
                currentEntryPos_,
                outerExitT_,
                rayActiveFlags_,
                accumT_,
                params,
                outerView);
        checkCuda(cudaGetLastError(), "traceOuterShellEntryFromRaysKernel launch");
        segmentRayDirections = rayDirections;
    }

    checkCuda(cudaMemset(hitCount_, 0, sizeof(int)), "cudaMemset hitCount");
    compactInputsKernel<<<compactGrid, compactBlock>>>(
            rayActiveFlags_,
            elementCountInt,
            hitIndices_,
            hitCount_);
    checkCuda(cudaGetLastError(), "compactInputsKernel launch");

    int activeCount = 0;
    checkCuda(cudaMemcpy(&activeCount, hitCount_, sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy activeCount");

    for (int iter = 0; iter < kMaxSegmentIterations && activeCount > 0; ++iter) {
        size_t activeCountSize = static_cast<size_t>(activeCount);
        size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
        size_t paddedActiveCount = roundUp(activeCountSize, granularity);

        const int buildBlock = 256;
        int buildGrid = (activeCount + buildBlock - 1) / buildBlock;

        traceSegmentExitsKernel<<<buildGrid, buildBlock>>>(
                currentEntryPos_,
                segmentRayDirections,
                hitIndices_,
                activeCount,
                outerView,
                innerView,
                params.material,
                segmentExitPos_,
                outerExitT_,
                innerEnterT_,
                innerHitFlags_);
        checkCuda(cudaGetLastError(), "traceSegmentExitsKernel launch");

        int pointCount = useMidpointEncoding_ ? 3 : 2;
        buildSegmentNeuralInputsKernel<<<buildGrid, buildBlock>>>(
                currentEntryPos_,
                segmentExitPos_,
                segmentRayDirections,
                hitIndices_,
                activeCount,
                outerMin,
                outerInvExtent,
                compactedPointInputs_,
                compactedDirs_,
                pointCount);
        checkCuda(cudaGetLastError(), "buildSegmentNeuralInputsKernel launch");

        if (paddedActiveCount > activeCountSize) {
            size_t tail = paddedActiveCount - activeCountSize;
            size_t pointInputStride = pointCount * 3;
            checkCuda(cudaMemset(compactedPointInputs_ + activeCountSize * pointInputStride, 0,
                                 tail * pointInputStride * sizeof(float)),
                      "cudaMemset point inputs tail");
            checkCuda(cudaMemset(compactedDirs_ + activeCountSize * 3, 0, tail * 3 * sizeof(float)),
                      "cudaMemset dirs tail");
        }

        uint32_t pointBatchSize = static_cast<uint32_t>(paddedActiveCount * pointCount);
        uint32_t rayBatchSize = static_cast<uint32_t>(paddedActiveCount);

        {
            tcnn::cpp::Context ctx = pointEncoding_->forward(
                0, pointBatchSize,
                compactedPointInputs_,
                pointEncOutput_,
                pointEncParams_,
                false);
            (void)ctx;
        }

        uint32_t pointFeatureDims = pointEncOutDims_ * pointCount;
        scatterPointEncodingsKernel<<<buildGrid, buildBlock>>>(
                static_cast<const __half*>(pointEncOutput_),
                pointEncOutDims_,
                static_cast<uint32_t>(pointCount),
                mlpInput_,
                pointFeatureDims,
                mlpInputDims_,
                activeCount);
        checkCuda(cudaGetLastError(), "scatterPointEncodingsKernel launch");

        {
            tcnn::cpp::Context ctx = dirEncoding_->forward(
                0, rayBatchSize,
                compactedDirs_,
                dirEncOutput_,
                dirEncParams_,
                false);
            (void)ctx;
        }

        appendDirectionEncodingsKernel<<<buildGrid, buildBlock>>>(
                static_cast<const __half*>(dirEncOutput_),
                dirEncOutDims_,
                mlpInput_,
                pointFeatureDims,
                mlpInputDims_,
                activeCount);
        checkCuda(cudaGetLastError(), "appendDirectionEncodingsKernel launch");

        if (paddedActiveCount > activeCountSize) {
            size_t tail = paddedActiveCount - activeCountSize;
            checkCuda(cudaMemset(mlpInput_ + activeCountSize * mlpInputDims_, 0,
                                 tail * mlpInputDims_ * sizeof(float)),
                      "cudaMemset mlp input tail");
        }

        {
            tcnn::cpp::Context ctx = mlpNetwork_->forward(
                0, rayBatchSize,
                mlpInput_,
                outputs_,
                mlpParams_,
                false);
            (void)ctx;
        }

        applySegmentNeuralOutputKernel<<<buildGrid, buildBlock>>>(
                static_cast<const __half*>(outputs_),
                static_cast<int>(mlpOutputDims_),
                hitIndices_,
                activeCount,
                currentEntryPos_,
                segmentRayDirections,
                accumT_,
                innerHitFlags_,
                innerEnterT_,
                outerExitT_,
                outHitPositions,
                outHitNormals,
                outHitColors,
                outHitMaterialParams,
                outHitFlags,
                rayActiveFlags_,
                params.material);
        checkCuda(cudaGetLastError(), "applySegmentNeuralOutputKernel launch");

        prepareNextIterationKernel<<<buildGrid, buildBlock>>>(
                segmentExitPos_,
                outerExitT_,
                segmentRayDirections,
                innerHitFlags_,
                hitIndices_,
                activeCount,
                outerView,
                params.material,
                currentEntryPos_,
                rayActiveFlags_,
                accumT_,
                innerEnterT_);
        checkCuda(cudaGetLastError(), "prepareNextIterationKernel launch");

        checkCuda(cudaMemset(hitCount_, 0, sizeof(int)), "cudaMemset hitCount");
        compactInputsKernel<<<compactGrid, compactBlock>>>(
                rayActiveFlags_,
                elementCountInt,
                hitIndices_,
                hitCount_);
        checkCuda(cudaGetLastError(), "compactInputsKernel launch");

        checkCuda(cudaMemcpy(&activeCount, hitCount_, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy activeCount");
    }
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
    scene_->additionalMesh().uploadToDevice();

    // Select mesh for non-neural rendering (0=original, 1=inner, 2=outer).
    Mesh* classicMesh = &originalMesh;
    if (classicMeshIndex_ == 1 && innerShell.numTriangles() > 0) {
        classicMesh = &innerShell;
    } else if (classicMeshIndex_ == 2 && outerShell.numTriangles() > 0) {
        classicMesh = &outerShell;
    }

    EnvironmentMap& environment = scene_->environment();
    environment.uploadToDevice();
    MeshDeviceView classicView = classicMesh->deviceView();
    MeshDeviceView outerView = outerShell.deviceView();
    MeshDeviceView innerView = innerShell.deviceView();
    EnvironmentDeviceView envView = environment.deviceView();
    envView.rotation = envmapRotation_ - 90.0f;  // Convert from nbvh convention

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

    // Calculate scene scale from the classic mesh bounds for scale-adaptive epsilons.
    Vec3 sceneMin = classicMesh->boundsMin();
    Vec3 sceneMax = classicMesh->boundsMax();
    Vec3 sceneExtent = sceneMax - sceneMin;
    sceneScale_ = sqrtf(sceneExtent.x * sceneExtent.x +
                        sceneExtent.y * sceneExtent.y +
                        sceneExtent.z * sceneExtent.z);
    if (sceneScale_ < 1e-6f) {
        sceneScale_ = 1.0f;  // Fallback for invalid bounds.
    }
    static bool printedOnce = false;
    if (!printedOnce) {
        printf("Scene scale: %.6f (extent: %.6f, %.6f, %.6f)\n",
               sceneScale_, sceneExtent.x, sceneExtent.y, sceneExtent.z);
        printedOnce = true;
    }

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
        classicMeshIndex_ != lastClassicMeshIndex_ || envmapRotation_ != lastEnvmapRotation_) {
        resetAccum();
    }
    lastUseNeuralQuery_ = useNeuralQuery_;
    lastLambertView_ = lambertView_;
    lastBounceCount_ = maxBounces;
    lastSamplesPerPixel_ = samplesPerPixel;
    lastClassicMeshIndex_ = classicMeshIndex_;
    lastEnvmapRotation_ = envmapRotation_;
    lastCamPos_ = camPos;
    lastBasis_ = basis_;
    lastFovY_ = basis_.fovY;
    hasLastCamera_ = true;

    const Material& material = scene_->globalMaterial();

    RenderParams params;
    params.camPos = camPos;
    params.camForward = basis_.forward;
    params.camRight = basis_.right;
    params.camUp = basis_.up;
    params.lightDir = lightDir_;
    params.outerShellMin = outerMin;
    params.outerShellInvExtent = outerInvExtent;
    params.material = material;
    params.fovY = basis_.fovY;
    params.maxRadiance = 100.0f;
    params.sceneScale = sceneScale_;
    params.maxBounces = maxBounces;
    params.width = width_;
    params.height = height_;
    params.pixelCount = static_cast<int>(pixelCount);
    params.samplesPerPixel = samplesPerPixel;
    params.sampleOffset = accumSampleCount_;

    dim3 block(8, 8);
    dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

    bool neuralReady = useNeuralQuery_ && pointEncoding_ && mlpNetwork_ &&
                       outerShell.numTriangles() > 0;
    if (neuralReady) {
        traceNeuralSegmentsForRays(
                true,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                elementCount,
                params,
                outerView,
                innerView,
                outerMin,
                outerInvExtent,
                hitPositions_,
                hitNormals_,
                hitColors_,
                hitMaterialParams_,
                hitFlags_);

        // Always trace additional mesh (empty mesh results in all misses)
        MeshDeviceView additionalView = scene_->additionalMesh().deviceView();

        traceAdditionalMeshPrimaryRaysKernel<<<grid, block>>>(
                additionalHitPositions_,
                additionalHitNormals_,
                additionalHitColors_,
                additionalHitMaterialParams_,
                additionalHitFlags_,
                params,
                additionalView);
        checkCuda(cudaGetLastError(), "traceAdditionalMeshPrimaryRaysKernel launch");
        checkCuda(cudaDeviceSynchronize(), "traceAdditionalMeshPrimaryRaysKernel sync");

        // Always select closest hit (if additional mesh empty, shell hits win)
        selectClosestPrimaryHitKernel<<<grid, block>>>(
                hitPositions_,
                hitNormals_,
                hitColors_,
                hitMaterialParams_,
                hitFlags_,
                additionalHitPositions_,
                additionalHitNormals_,
                additionalHitColors_,
                additionalHitMaterialParams_,
                additionalHitFlags_,
                params);
        checkCuda(cudaGetLastError(), "selectClosestPrimaryHitKernel launch");
        checkCuda(cudaDeviceSynchronize(), "selectClosestPrimaryHitKernel sync");

        // 4. Path trace from neural hits using wavefront architecture.
        // (Step numbers continue from multi-segment iteration above)
        if (!lambertView_) {
            // Initialize path state from neural primary hits
            initializePathStateKernel<<<grid, block>>>(
                    pathThroughput_,
                    pathRadiance_,
                    pathActive_,
                    hitFlags_,
                    hitNormals_,
                    hitColors_,
                    params,
                    envView);
            checkCuda(cudaGetLastError(), "initializePathStateKernel launch");

            // Wavefront bounce loop for neural mode (neural segments + additional mesh)
            float* currentHitPos = hitPositions_;
            float* currentHitNormals = hitNormals_;
            float* currentHitColors = hitColors_;
            float* currentHitMaterialParams = hitMaterialParams_;
            int* currentHitFlags = hitFlags_;

            for (int bounce = 1; bounce <= maxBounces; ++bounce) {
                // Sample bounce directions (Disney BRDF - shared with GT)
                sampleBounceDirectionsKernel<<<grid, block>>>(
                        currentHitPos,
                        currentHitNormals,
                        currentHitColors,
                        currentHitMaterialParams,
                        currentHitFlags,
                        pathActive_,
                        params,
                        bounceOrigins_,
                        bounceDirections_,
                        bouncePdfs_,
                        bounceBRDFs_);
                checkCuda(cudaGetLastError(), "sampleBounceDirectionsKernel launch");

                traceNeuralSegmentsForRays(
                        false,
                        bounceOrigins_,
                        bounceDirections_,
                        pathActive_,
                        bouncePdfs_,
                        elementCount,
                        params,
                        outerView,
                        innerView,
                        outerMin,
                        outerInvExtent,
                        bouncePositions_,
                        bounceNormals_,
                        bounceColors_,
                        bounceMaterialParams_,
                        bounceHitFlags_);

                traceAdditionalMeshRaysKernel<<<grid, block>>>(
                        bounceOrigins_,
                        bounceDirections_,
                        bouncePdfs_,
                        additionalHitPositions_,
                        additionalHitNormals_,
                        additionalHitColors_,
                        additionalHitMaterialParams_,
                        additionalHitFlags_,
                        params,
                        additionalView);
                checkCuda(cudaGetLastError(), "traceAdditionalMeshRaysKernel launch");

                selectClosestHitKernel<<<grid, block>>>(
                        bouncePositions_,
                        bounceNormals_,
                        bounceColors_,
                        bounceMaterialParams_,
                        bounceHitFlags_,
                        additionalHitPositions_,
                        additionalHitNormals_,
                        additionalHitColors_,
                        additionalHitMaterialParams_,
                        additionalHitFlags_,
                        bounceOrigins_,
                        params);
                checkCuda(cudaGetLastError(), "selectClosestHitKernel launch");

                // Integrate bounce results (shared with GT)
                integrateBounceKernel<<<grid, block>>>(
                        pathThroughput_,
                        pathRadiance_,
                        pathActive_,
                        bounceHitFlags_,
                        bounceDirections_,
                        bounceBRDFs_,
                        bounce,
                        params,
                        envView);
                checkCuda(cudaGetLastError(), "integrateBounceKernel launch");

                // Swap buffers for next bounce
                currentHitPos = bouncePositions_;
                currentHitNormals = bounceNormals_;
                currentHitColors = bounceColors_;
                currentHitFlags = bounceHitFlags_;
                currentHitMaterialParams = bounceMaterialParams_;
            }

            // Finalize and output
            finalizePathTracingKernel<<<grid, block>>>(
                    devicePixels_,
                    accum_,
                    pathRadiance_,
                    params);
            checkCuda(cudaGetLastError(), "finalizePathTracingKernel launch");
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
        // --- Ground truth mesh path tracing (wavefront architecture) ---
        // 1. Trace primary rays
        intersectGroundTruthKernel<<<grid, block>>>(
                hitPositions_,
                hitNormals_,
                hitColors_,
                hitMaterialParams_,
                hitFlags_,
                params,
                classicView);
        checkCuda(cudaGetLastError(), "intersectGroundTruthKernel launch");

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
            // 2. Initialize path state
            initializePathStateKernel<<<grid, block>>>(
                    pathThroughput_,
                    pathRadiance_,
                    pathActive_,
                    hitFlags_,
                    hitNormals_,
                    hitColors_,
                    params,
                    envView);
            checkCuda(cudaGetLastError(), "initializePathStateKernel launch");

            // 3. Wavefront bounce loop
            float* currentHitPos = hitPositions_;
            float* currentHitNormals = hitNormals_;
            float* currentHitColors = hitColors_;
            float* currentHitMaterialParams = hitMaterialParams_;
            int* currentHitFlags = hitFlags_;

            for (int bounce = 1; bounce <= maxBounces; ++bounce) {
                // Sample bounce directions (Disney BRDF)
                sampleBounceDirectionsKernel<<<grid, block>>>(
                        currentHitPos,
                        currentHitNormals,
                        currentHitColors,
                        currentHitMaterialParams,
                        currentHitFlags,
                        pathActive_,
                        params,
                        bounceOrigins_,
                        bounceDirections_,
                        bouncePdfs_,
                        bounceBRDFs_);
                checkCuda(cudaGetLastError(), "sampleBounceDirectionsKernel launch");

                // Trace bounce rays against GT mesh
                traceGroundTruthBouncesKernel<<<grid, block>>>(
                        bounceOrigins_,
                        bounceDirections_,
                        bouncePdfs_,
                        classicView,
                        params,
                        bouncePositions_,
                        bounceNormals_,
                        bounceColors_,
                        bounceMaterialParams_,
                        bounceHitFlags_);
                checkCuda(cudaGetLastError(), "traceGroundTruthBouncesKernel launch");

                // Integrate bounce results
                integrateBounceKernel<<<grid, block>>>(
                        pathThroughput_,
                        pathRadiance_,
                        pathActive_,
                        bounceHitFlags_,
                        bounceDirections_,
                        bounceBRDFs_,
                        bounce,
                        params,
                        envView);
                checkCuda(cudaGetLastError(), "integrateBounceKernel launch");

                // Swap buffers for next bounce
                currentHitPos = bouncePositions_;
                currentHitNormals = bounceNormals_;
                currentHitColors = bounceColors_;
                currentHitMaterialParams = bounceMaterialParams_;
                currentHitFlags = bounceHitFlags_;
            }

            // 4. Finalize and output
            finalizePathTracingKernel<<<grid, block>>>(
                    devicePixels_,
                    accum_,
                    pathRadiance_,
                    params);
            checkCuda(cudaGetLastError(), "finalizePathTracingKernel launch");
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
    freePtr(compactedPointInputs_);
    freePtr(compactedDirs_);
    freePtr(pointEncOutput_);
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
    freePtr(hitMaterialParams_);
    freePtr(hitFlags_);
    freePtr(additionalHitPositions_);
    freePtr(additionalHitNormals_);
    freePtr(additionalHitColors_);
    freePtr(additionalHitMaterialParams_);
    freePtr(additionalHitFlags_);
    freePtr(outputs_);
    freePtr(bouncePositions_);
    freePtr(bounceNormals_);
    freePtr(bounceDirs_);
    freePtr(bounceColors_);
    freePtr(bounceMaterialParams_);
    freePtr(bounceHitFlags_);
    freePtr(bounce2Positions_);
    freePtr(bounce2Normals_);
    freePtr(bounce2Dirs_);
    freePtr(bounce2Colors_);
    freePtr(bounce2HitFlags_);
    freePtr(pathThroughput_);
    freePtr(pathRadiance_);
    freePtr(pathActive_);
    freePtr(bounceOrigins_);
    freePtr(bounceDirections_);
    freePtr(bouncePdfs_);
    freePtr(bounceBRDFs_);
    freePtr(rayActiveFlags_);
    freePtr(accumT_);
    freePtr(currentEntryPos_);
    freePtr(outerExitT_);
    freePtr(innerEnterT_);
    freePtr(innerHitFlags_);
    freePtr(segmentExitPos_);
    bufferElements_ = 0;
    accumPixels_ = 0;
    accumSampleCount_ = 0;
    hasLastCamera_ = false;
    pointEncPointCount_ = 0;
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
    pointEncPointCount_ = 0;
}

bool RendererNeural::ensureNetworkBuffers(size_t elementCount) {
    if (elementCount == 0) {
        return false;
    }
    uint32_t pointCount = useMidpointEncoding_ ? 3u : 2u;
    if (elementCount <= bufferElements_ &&
            hitPositions_ && hitNormals_ && hitColors_ && hitMaterialParams_ && hitFlags_ &&
            additionalHitPositions_ && additionalHitNormals_ && additionalHitColors_ && additionalHitMaterialParams_ && additionalHitFlags_ &&
            outerHitPositions_ && innerHitPositions_ && rayDirections_ && outerHitFlags_ &&
            compactedPointInputs_ && compactedDirs_ &&
            pointEncPointCount_ == pointCount &&
            hitIndices_ && hitCount_ &&
            bouncePositions_ && bounceNormals_ && bounceDirs_ && bounceColors_ && bounceMaterialParams_ && bounceHitFlags_ &&
            bounce2Positions_ && bounce2Normals_ && bounce2Dirs_ && bounce2Colors_ && bounce2HitFlags_ &&
            pathThroughput_ && pathRadiance_ && pathActive_ &&
            rayActiveFlags_ && accumT_ && currentEntryPos_ &&
            outerExitT_ && innerEnterT_ && innerHitFlags_ && segmentExitPos_) {
        return true;
    }

    // Free old buffers.
    auto freePtr = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    freePtr(compactedPointInputs_);
    freePtr(compactedDirs_);
    freePtr(pointEncOutput_);
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
    freePtr(hitMaterialParams_);
    freePtr(hitFlags_);
    freePtr(additionalHitPositions_);
    freePtr(additionalHitNormals_);
    freePtr(additionalHitColors_);
    freePtr(additionalHitMaterialParams_);
    freePtr(additionalHitFlags_);
    freePtr(outputs_);
    freePtr(bouncePositions_);
    freePtr(bounceNormals_);
    freePtr(bounceDirs_);
    freePtr(bounceColors_);
    freePtr(bounceMaterialParams_);
    freePtr(bounceHitFlags_);
    freePtr(bounce2Positions_);
    freePtr(bounce2Normals_);
    freePtr(bounce2Dirs_);
    freePtr(bounce2Colors_);
    freePtr(bounce2HitFlags_);
    freePtr(pathThroughput_);
    freePtr(pathRadiance_);
    freePtr(pathActive_);
    freePtr(bounceOrigins_);
    freePtr(bounceDirections_);
    freePtr(bouncePdfs_);
    freePtr(bounceBRDFs_);
    freePtr(rayActiveFlags_);
    freePtr(accumT_);
    freePtr(currentEntryPos_);
    freePtr(outerExitT_);
    freePtr(innerEnterT_);
    freePtr(innerHitFlags_);
    freePtr(segmentExitPos_);

    pointEncPointCount_ = 0;

    size_t vec3Bytes = elementCount * 3 * sizeof(float);
    size_t intBytes = elementCount * sizeof(int);

    // Shell tracing buffers.
    checkCuda(cudaMalloc(&outerHitPositions_, vec3Bytes), "cudaMalloc outerHitPositions");
    checkCuda(cudaMalloc(&innerHitPositions_, vec3Bytes), "cudaMalloc innerHitPositions");
    checkCuda(cudaMalloc(&rayDirections_, vec3Bytes), "cudaMalloc rayDirections");
    checkCuda(cudaMalloc(&outerHitFlags_, intBytes), "cudaMalloc outerHitFlags");

    // Encoding input buffers (entry/exit/midpoint per element).
    size_t pointInputStride = pointCount * 3;
    size_t pointInputBytes = elementCount * pointInputStride * sizeof(float);
    checkCuda(cudaMalloc(&compactedPointInputs_, pointInputBytes), "cudaMalloc compactedPointInputs");
    checkCuda(cudaMalloc(&compactedDirs_, vec3Bytes), "cudaMalloc compactedDirs");

    // Encoding output buffers (FP16) for all concatenated points.
    if (pointEncOutDims_ > 0) {
        size_t encOutBytes = elementCount * pointCount * pointEncOutDims_ * sizeof(__half);
        checkCuda(cudaMalloc(&pointEncOutput_, encOutBytes), "cudaMalloc pointEncOutput");
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
    checkCuda(cudaMalloc(&hitMaterialParams_, vec3Bytes), "cudaMalloc hitMaterialParams");
    checkCuda(cudaMalloc(&hitFlags_, intBytes), "cudaMalloc hitFlags");

    // Additional mesh hit buffers (for hybrid rendering).
    checkCuda(cudaMalloc(&additionalHitPositions_, vec3Bytes), "cudaMalloc additionalHitPositions");
    checkCuda(cudaMalloc(&additionalHitNormals_, vec3Bytes), "cudaMalloc additionalHitNormals");
    checkCuda(cudaMalloc(&additionalHitColors_, vec3Bytes), "cudaMalloc additionalHitColors");
    checkCuda(cudaMalloc(&additionalHitMaterialParams_, vec3Bytes), "cudaMalloc additionalHitMaterialParams");
    checkCuda(cudaMalloc(&additionalHitFlags_, intBytes), "cudaMalloc additionalHitFlags");

    // Bounce buffers.
    checkCuda(cudaMalloc(&bouncePositions_, vec3Bytes), "cudaMalloc bouncePositions");
    checkCuda(cudaMalloc(&bounceNormals_, vec3Bytes), "cudaMalloc bounceNormals");
    checkCuda(cudaMalloc(&bounceColors_, vec3Bytes), "cudaMalloc bounceColors");
    checkCuda(cudaMalloc(&bounceMaterialParams_, vec3Bytes), "cudaMalloc bounceMaterialParams");
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

    // Wavefront buffers.
    checkCuda(cudaMalloc(&bounceOrigins_, vec3Bytes), "cudaMalloc bounceOrigins");
    checkCuda(cudaMalloc(&bounceDirections_, vec3Bytes), "cudaMalloc bounceDirections");
    checkCuda(cudaMalloc(&bouncePdfs_, elementCount * sizeof(float)), "cudaMalloc bouncePdfs");
    checkCuda(cudaMalloc(&bounceBRDFs_, vec3Bytes), "cudaMalloc bounceBRDFs");

    // Multi-segment iteration state buffers.
    checkCuda(cudaMalloc(&rayActiveFlags_, intBytes), "cudaMalloc rayActiveFlags");
    checkCuda(cudaMalloc(&accumT_, elementCount * sizeof(float)), "cudaMalloc accumT");
    checkCuda(cudaMalloc(&currentEntryPos_, vec3Bytes), "cudaMalloc currentEntryPos");
    checkCuda(cudaMalloc(&outerExitT_, elementCount * sizeof(float)), "cudaMalloc outerExitT");
    checkCuda(cudaMalloc(&innerEnterT_, elementCount * sizeof(float)), "cudaMalloc innerEnterT");
    checkCuda(cudaMalloc(&innerHitFlags_, intBytes), "cudaMalloc innerHitFlags");
    checkCuda(cudaMalloc(&segmentExitPos_, vec3Bytes), "cudaMalloc segmentExitPos");

    pointEncPointCount_ = pointCount;

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

    auto zeroBuffer = [](auto* ptr, size_t bytes, const char* name) {
        if (ptr && bytes > 0) {
            checkCuda(cudaMemset(ptr, 0, bytes), name);
        }
    };

    if (bufferElements_ > 0) {
        zeroBuffer(pathRadiance_, bufferElements_ * sizeof(Vec3), "cudaMemset pathRadiance");
        zeroBuffer(pathThroughput_, bufferElements_ * sizeof(Vec3), "cudaMemset pathThroughput");
        zeroBuffer(pathActive_, bufferElements_ * sizeof(int), "cudaMemset pathActive");
        zeroBuffer(hitFlags_, bufferElements_ * sizeof(int), "cudaMemset hitFlags");
        zeroBuffer(hitPositions_, bufferElements_ * 3 * sizeof(float), "cudaMemset hitPositions");
        zeroBuffer(hitNormals_, bufferElements_ * 3 * sizeof(float), "cudaMemset hitNormals");
        zeroBuffer(hitColors_, bufferElements_ * 3 * sizeof(float), "cudaMemset hitColors");
        zeroBuffer(hitMaterialParams_, bufferElements_ * 3 * sizeof(float), "cudaMemset hitMaterialParams");
        zeroBuffer(additionalHitFlags_, bufferElements_ * sizeof(int), "cudaMemset additionalHitFlags");
        zeroBuffer(additionalHitPositions_, bufferElements_ * 3 * sizeof(float), "cudaMemset additionalHitPositions");
        zeroBuffer(additionalHitNormals_, bufferElements_ * 3 * sizeof(float), "cudaMemset additionalHitNormals");
        zeroBuffer(additionalHitColors_, bufferElements_ * 3 * sizeof(float), "cudaMemset additionalHitColors");
        zeroBuffer(additionalHitMaterialParams_, bufferElements_ * 3 * sizeof(float), "cudaMemset additionalHitMaterialParams");
    }
}
