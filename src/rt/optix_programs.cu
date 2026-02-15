// OptiX device programs: raygen, miss, closesthit.
// This file is compiled to PTX by CMake and embedded in the host binary.
//
// The PTX is loaded by optixCreateState() via optixModuleCreate().

#include <optix.h>
#include <optix_device.h>

#include "optix_launch_params.h"
#include "../mesh_traversal.cuh"

// ---------------------------------------------------------------------------
// Launch parameters (defined once, referenced by all programs)
// ---------------------------------------------------------------------------
// OptixLaunchParams contains Vec3 (user-defined ctor) so it cannot be declared
// as a __constant__ object directly (CUDA disallows dynamic initialisation of
// __constant__ variables).  Instead we store raw bytes and reinterpret-cast.
// __align__(8) ensures 8-byte-aligned constant memory so that pointer-sized
// fields inside OptixLaunchParams are naturally aligned when read back.
extern "C" {
    __constant__ __align__(8) unsigned char launchParamsRaw[sizeof(OptixLaunchParams)];
}
// Convenience reference — all device code below uses this name unchanged.
static __device__ __forceinline__ const OptixLaunchParams& launchParamsRef() {
    return *reinterpret_cast<const OptixLaunchParams*>(launchParamsRaw);
}
#define launchParams launchParamsRef()

// ---------------------------------------------------------------------------
// Helper: trace one ray against a GAS and return preliminary hit info
// ---------------------------------------------------------------------------
__device__ inline bool traceMeshOptiX(
        OptixTraversableHandle gas,
        const Ray& ray,
        float tMin,
        float tMax,
        TraceMode mode,
        uint32_t& outPrimIdx,
        float& outT,
        float& outU,
        float& outV) {

    uint32_t rayFlags = OPTIX_RAY_FLAG_NONE;
    if (mode == TraceMode::FORWARD_ONLY)  rayFlags = OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    if (mode == TraceMode::BACKWARD_ONLY) rayFlags = OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;

    // Payload registers: p0=t(bits), p1=u(bits), p2=v(bits), p3=primIdx, p4=hit
    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;

    optixTrace(gas,
               make_float3(ray.origin.x, ray.origin.y, ray.origin.z),
               make_float3(ray.direction.x, ray.direction.y, ray.direction.z),
               tMin, tMax, /*rayTime=*/0.0f,
               OptixVisibilityMask(0xFF),
               rayFlags,
               /*SBT offset=*/0, /*SBT stride=*/1, /*miss index=*/0,
               p0, p1, p2, p3, p4);

    if (p4 == 0) return false;

    outT       = __uint_as_float(p0);
    outU       = __uint_as_float(p1);
    outV       = __uint_as_float(p2);
    outPrimIdx = p3;
    return true;
}

// ---------------------------------------------------------------------------
// Closest-hit program (shared by all raygen programs)
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__default() {
    float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(__float_as_uint(bary.x));
    optixSetPayload_2(__float_as_uint(bary.y));
    optixSetPayload_3(optixGetPrimitiveIndex());
    optixSetPayload_4(1u);
}

// ---------------------------------------------------------------------------
// Miss program
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__default() {
    optixSetPayload_4(0u);
}

// ===========================================================================
// Raygen helper: write hit result to output buffers
// ===========================================================================
// Pack position, normal, and UV+materialId for later SW resolution.
// hitColors[0,1] = uv.x, uv.y; hitColors[2] = reinterpret_cast<float>(materialId)
__device__ __forceinline__ void writeHitResult(
        int sampleIdx,
        const HitInfo& hitInfo,
        const MeshDeviceView& /*mesh*/,
        const RenderParams& /*rp*/,
        float* hitPositions,
        float* hitNormals,
        float* hitColors,
        float* /*hitMaterialParams*/,
        int*   hitFlags) {
    int base = sampleIdx * 3;
    hitPositions[base + 0] = hitInfo.position.x;
    hitPositions[base + 1] = hitInfo.position.y;
    hitPositions[base + 2] = hitInfo.position.z;
    hitNormals[base + 0]   = hitInfo.shadingNormal.x;
    hitNormals[base + 1]   = hitInfo.shadingNormal.y;
    hitNormals[base + 2]   = hitInfo.shadingNormal.z;
    hitColors[base + 0]    = hitInfo.uv.x;
    hitColors[base + 1]    = hitInfo.uv.y;
    hitColors[base + 2]    = __int_as_float(hitInfo.materialId);
    hitFlags[sampleIdx]    = 1;
}

__device__ inline void writeMissResult(
        int sampleIdx,
        const RenderParams& rp,
        float* hitPositions,
        float* hitNormals,
        float* hitColors,
        float* hitMaterialParams,
        int*   hitFlags) {
    int base = sampleIdx * 3;
    hitPositions[base + 0] = 0.0f;  hitPositions[base + 1] = 0.0f;  hitPositions[base + 2] = 0.0f;
    hitNormals[base + 0]   = 0.0f;  hitNormals[base + 1]   = 0.0f;  hitNormals[base + 2]   = 0.0f;
    hitColors[base + 0]    = 0.0f;  hitColors[base + 1]    = 0.0f;  hitColors[base + 2]    = 0.0f;
    if (hitMaterialParams) {
        hitMaterialParams[base + 0] = rp.material.metallic.value;
        hitMaterialParams[base + 1] = rp.material.roughness.value;
        hitMaterialParams[base + 2] = rp.material.specular.value;
    }
    hitFlags[sampleIdx] = 0;
}

// ===========================================================================
// __raygen__primaryGT
// Replaces intersectGroundTruthKernel — primary rays against classic mesh.
// ===========================================================================
extern "C" __global__ void __raygen__primaryGT() {
    const RenderParams& rp = launchParams.renderParams;
    int x = static_cast<int>(optixGetLaunchIndex().x);
    int y = static_cast<int>(optixGetLaunchIndex().y);
    if (x >= rp.width || y >= rp.height) return;

    int pixelIdx = y * rp.width + x;
    for (int s = 0; s < rp.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * rp.pixelCount;
        uint32_t rng  = initRng(pixelIdx, rp.sampleOffset, s);
        Ray ray       = generatePrimaryRay(x, y, rp, rng);

        uint32_t primIdx; float t, u, v;
        bool hit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                  TraceMode::FORWARD_ONLY, primIdx, t, u, v);
        if (hit) {
            PreliminaryHitInfo pi; pi.t = t; pi.u = u; pi.v = v; pi.primIdx = primIdx;
            HitInfo hi = computeHitData(pi, primIdx, ray, launchParams.mesh);
            writeHitResult(sampleIdx, hi, launchParams.mesh, rp,
                           launchParams.hitPositions, launchParams.hitNormals,
                           launchParams.hitColors, launchParams.hitMaterialParams,
                           launchParams.hitFlags);
        } else {
            writeMissResult(sampleIdx, rp,
                            launchParams.hitPositions, launchParams.hitNormals,
                            launchParams.hitColors, launchParams.hitMaterialParams,
                            launchParams.hitFlags);
        }
    }
}

// ===========================================================================
// __raygen__bounceGT
// Replaces traceGroundTruthBouncesKernel — bounce rays against classic mesh.
// ===========================================================================
extern "C" __global__ void __raygen__bounceGT() {
    const RenderParams& rp = launchParams.renderParams;
    int x = static_cast<int>(optixGetLaunchIndex().x);
    int y = static_cast<int>(optixGetLaunchIndex().y);
    if (x >= rp.width || y >= rp.height) return;

    int pixelIdx = y * rp.width + x;
    for (int s = 0; s < rp.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * rp.pixelCount;
        int base      = sampleIdx * 3;

        if (launchParams.rayPdfs && launchParams.rayPdfs[sampleIdx] <= 0.0f) {
            launchParams.hitFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 origin(launchParams.rayOrigins[base + 0],
                    launchParams.rayOrigins[base + 1],
                    launchParams.rayOrigins[base + 2]);
        Vec3 dir(launchParams.rayDirections[base + 0],
                 launchParams.rayDirections[base + 1],
                 launchParams.rayDirections[base + 2]);
        Ray ray(origin, dir);

        uint32_t primIdx; float t, u, v;
        bool hit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                  TraceMode::FORWARD_ONLY, primIdx, t, u, v);
        if (hit) {
            PreliminaryHitInfo pi; pi.t = t; pi.u = u; pi.v = v; pi.primIdx = primIdx;
            HitInfo hi = computeHitData(pi, primIdx, ray, launchParams.mesh);
            writeHitResult(sampleIdx, hi, launchParams.mesh, rp,
                           launchParams.hitPositions, launchParams.hitNormals,
                           launchParams.hitColors, launchParams.hitMaterialParams,
                           launchParams.hitFlags);
        } else {
            launchParams.hitFlags[sampleIdx] = 0;
            if (launchParams.hitMaterialParams) {
                launchParams.hitMaterialParams[base + 0] = rp.material.metallic.value;
                launchParams.hitMaterialParams[base + 1] = rp.material.roughness.value;
                launchParams.hitMaterialParams[base + 2] = rp.material.specular.value;
            }
        }
    }
}

// ===========================================================================
// __raygen__shellEntry
// Replaces traceOuterShellEntryKernel — primary rays into outer shell.
// ===========================================================================
extern "C" __global__ void __raygen__shellEntry() {
    const RenderParams& rp = launchParams.renderParams;
    int x = static_cast<int>(optixGetLaunchIndex().x);
    int y = static_cast<int>(optixGetLaunchIndex().y);
    if (x >= rp.width || y >= rp.height) return;

    int pixelIdx = y * rp.width + x;
    for (int s = 0; s < rp.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * rp.pixelCount;
        int base      = sampleIdx * 3;
        uint32_t rng  = initRng(pixelIdx, rp.sampleOffset, s);
        Ray ray       = generatePrimaryRay(x, y, rp, rng);

        // Store ray direction
        launchParams.storedRayDirections[base + 0] = ray.direction.x;
        launchParams.storedRayDirections[base + 1] = ray.direction.y;
        launchParams.storedRayDirections[base + 2] = ray.direction.z;

        uint32_t primIdx; float t, u, v;
        bool hit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                  TraceMode::FORWARD_ONLY, primIdx, t, u, v);
        if (hit) {
            Vec3 entryPos = ray.at(t);
            launchParams.entryPositions[base + 0] = entryPos.x;
            launchParams.entryPositions[base + 1] = entryPos.y;
            launchParams.entryPositions[base + 2] = entryPos.z;
            launchParams.entryT[sampleIdx]     = t;
            launchParams.activeFlags[sampleIdx] = 1;
            launchParams.accumT[sampleIdx]      = t;
        } else {
            launchParams.entryPositions[base + 0] = 0.0f;
            launchParams.entryPositions[base + 1] = 0.0f;
            launchParams.entryPositions[base + 2] = 0.0f;
            launchParams.entryT[sampleIdx]     = 0.0f;
            launchParams.activeFlags[sampleIdx] = 0;
            launchParams.accumT[sampleIdx]      = 0.0f;
        }
    }
}

// ===========================================================================
// __raygen__shellEntryFromRays
// Replaces traceOuterShellEntryFromRaysKernel — arbitrary rays into outer shell,
// handles rays that start inside the shell by exiting first.
// ===========================================================================
extern "C" __global__ void __raygen__shellEntryFromRays() {
    const RenderParams& rp    = launchParams.renderParams;
    const float kEps          = 1e-8f;
    int x = static_cast<int>(optixGetLaunchIndex().x);
    int y = static_cast<int>(optixGetLaunchIndex().y);
    if (x >= rp.width || y >= rp.height) return;

    int pixelIdx = y * rp.width + x;
    for (int s = 0; s < rp.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * rp.pixelCount;
        int base      = sampleIdx * 3;

        if (launchParams.rayActiveMask && !launchParams.rayActiveMask[sampleIdx]) {
            launchParams.entryPositions[base+0] = 0.0f;
            launchParams.entryPositions[base+1] = 0.0f;
            launchParams.entryPositions[base+2] = 0.0f;
            launchParams.entryT[sampleIdx]      = 0.0f;
            launchParams.activeFlags[sampleIdx] = 0;
            launchParams.accumT[sampleIdx]      = 0.0f;
            continue;
        }
        if (launchParams.rayPdfs && launchParams.rayPdfs[sampleIdx] <= 0.0f) {
            launchParams.entryPositions[base+0] = 0.0f;
            launchParams.entryPositions[base+1] = 0.0f;
            launchParams.entryPositions[base+2] = 0.0f;
            launchParams.entryT[sampleIdx]      = 0.0f;
            launchParams.activeFlags[sampleIdx] = 0;
            launchParams.accumT[sampleIdx]      = 0.0f;
            continue;
        }

        Vec3 origin(launchParams.rayOrigins[base+0],
                    launchParams.rayOrigins[base+1],
                    launchParams.rayOrigins[base+2]);
        Vec3 dir(launchParams.rayDirections[base+0],
                 launchParams.rayDirections[base+1],
                 launchParams.rayDirections[base+2]);
        Ray ray(origin, dir);

        uint32_t primIdx; float t, u, v;
        float baseOffset  = 0.0f;
        Vec3 entryOrigin  = origin;
        bool hitOuter     = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                           TraceMode::FORWARD_ONLY, primIdx, t, u, v);
        if (!hitOuter) {
            // Try exit (backward trace) then retry
            uint32_t ep; float et, eu, ev;
            bool hitExit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                          TraceMode::BACKWARD_ONLY, ep, et, eu, ev);
            if (hitExit) {
                baseOffset  = et + kEps;
                entryOrigin = origin + dir * baseOffset;
                Ray shiftedRay(entryOrigin, dir);
                hitOuter = traceMeshOptiX(launchParams.gas, shiftedRay, 1e-6f, 1e30f,
                                          TraceMode::FORWARD_ONLY, primIdx, t, u, v);
            }
        }

        if (hitOuter) {
            Vec3 entryPos   = entryOrigin + dir * t;
            float totalEntryT = baseOffset + t;
            launchParams.entryPositions[base+0] = entryPos.x;
            launchParams.entryPositions[base+1] = entryPos.y;
            launchParams.entryPositions[base+2] = entryPos.z;
            launchParams.entryT[sampleIdx]      = totalEntryT;
            launchParams.activeFlags[sampleIdx] = 1;
            launchParams.accumT[sampleIdx]      = totalEntryT;
        } else {
            launchParams.entryPositions[base+0] = 0.0f;
            launchParams.entryPositions[base+1] = 0.0f;
            launchParams.entryPositions[base+2] = 0.0f;
            launchParams.entryT[sampleIdx]      = 0.0f;
            launchParams.activeFlags[sampleIdx] = 0;
            launchParams.accumT[sampleIdx]      = 0.0f;
        }
    }
}

// ===========================================================================
// __raygen__shellExit
// Replaces traceSegmentExitsKernel — 1D launch over compacted hit list.
// Traces outer shell backward (exit) and inner shell (any), picks closest.
// ===========================================================================
extern "C" __global__ void __raygen__shellExit() {
    const float kEps = 1e-8f;
    int idx = static_cast<int>(optixGetLaunchIndex().x);
    if (idx >= launchParams.hitCount) return;

    int sampleIdx = launchParams.hitIndices[idx];
    int base      = sampleIdx * 3;

    Vec3 entryPos(launchParams.entryPositions[base+0],
                  launchParams.entryPositions[base+1],
                  launchParams.entryPositions[base+2]);
    Vec3 dir(launchParams.storedRayDirections[base+0],
             launchParams.storedRayDirections[base+1],
             launchParams.storedRayDirections[base+2]);

    Vec3 shiftedEntry = entryPos + dir * kEps;
    Ray  ray(shiftedEntry, dir);

    // Outer exit (backward)
    uint32_t ep; float exitT, eu, ev;
    bool hitOuterExit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                       TraceMode::BACKWARD_ONLY, ep, exitT, eu, ev);
    if (!hitOuterExit) exitT = kEps;
    launchParams.outerExitT[sampleIdx] = exitT;

    // Inner shell (any mode, gas2)
    uint32_t ip; float innerT, iu, iv;
    bool hitInner = traceMeshOptiX(launchParams.gas2, ray, 1e-6f, 1e30f,
                                   TraceMode::ANY, ip, innerT, iu, iv);
    if (hitInner) {
        launchParams.innerEnterT[sampleIdx]  = innerT;
        launchParams.innerHitFlags[sampleIdx] = 1;
    } else {
        innerT = 1e30f;
        launchParams.innerEnterT[sampleIdx]  = innerT;
        launchParams.innerHitFlags[sampleIdx] = 0;
    }

    // Exit position = nearer of outer_exit or inner_enter
    bool innerBeforeOuter = hitInner && (innerT < exitT);
    Vec3 exitPos = shiftedEntry + dir * (innerBeforeOuter ? innerT : exitT);

    // Reuse hitPositions to store exit positions (same as segmentExitPos_ in SW path)
    launchParams.hitPositions[base+0] = exitPos.x;
    launchParams.hitPositions[base+1] = exitPos.y;
    launchParams.hitPositions[base+2] = exitPos.z;
}

// ===========================================================================
// __raygen__additionalPrimary
// Replaces traceAdditionalMeshPrimaryRaysKernel.
// ===========================================================================
extern "C" __global__ void __raygen__additionalPrimary() {
    const RenderParams& rp = launchParams.renderParams;
    int x = static_cast<int>(optixGetLaunchIndex().x);
    int y = static_cast<int>(optixGetLaunchIndex().y);
    if (x >= rp.width || y >= rp.height) return;

    int pixelIdx = y * rp.width + x;
    for (int s = 0; s < rp.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * rp.pixelCount;
        int base      = sampleIdx * 3;

        if (launchParams.mesh.numTriangles == 0) {
            launchParams.hitFlags[sampleIdx] = 0;
            continue;
        }

        uint32_t rng = initRng(pixelIdx, rp.sampleOffset, s);
        Ray ray      = generatePrimaryRay(x, y, rp, rng);

        uint32_t primIdx; float t, u, v;
        bool hit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                  TraceMode::FORWARD_ONLY, primIdx, t, u, v);
        if (hit) {
            PreliminaryHitInfo pi; pi.t = t; pi.u = u; pi.v = v; pi.primIdx = primIdx;
            HitInfo hi = computeHitData(pi, primIdx, ray, launchParams.mesh);
            writeHitResult(sampleIdx, hi, launchParams.mesh, rp,
                           launchParams.hitPositions, launchParams.hitNormals,
                           launchParams.hitColors, launchParams.hitMaterialParams,
                           launchParams.hitFlags);
        } else {
            launchParams.hitFlags[sampleIdx] = 0;
            if (launchParams.hitMaterialParams) {
                launchParams.hitMaterialParams[base+0] = rp.material.metallic.value;
                launchParams.hitMaterialParams[base+1] = rp.material.roughness.value;
                launchParams.hitMaterialParams[base+2] = rp.material.specular.value;
            }
        }
    }
}

// ===========================================================================
// __raygen__additionalBounce
// Replaces traceAdditionalMeshRaysKernel — arbitrary rays against additional mesh.
// ===========================================================================
extern "C" __global__ void __raygen__additionalBounce() {
    const RenderParams& rp = launchParams.renderParams;
    int x = static_cast<int>(optixGetLaunchIndex().x);
    int y = static_cast<int>(optixGetLaunchIndex().y);
    if (x >= rp.width || y >= rp.height) return;

    int pixelIdx = y * rp.width + x;
    for (int s = 0; s < rp.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * rp.pixelCount;
        int base      = sampleIdx * 3;

        if (launchParams.mesh.numTriangles == 0) {
            launchParams.hitFlags[sampleIdx] = 0;
            continue;
        }
        if (launchParams.rayPdfs && launchParams.rayPdfs[sampleIdx] <= 0.0f) {
            launchParams.hitFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 origin(launchParams.rayOrigins[base+0],
                    launchParams.rayOrigins[base+1],
                    launchParams.rayOrigins[base+2]);
        Vec3 dir(launchParams.rayDirections[base+0],
                 launchParams.rayDirections[base+1],
                 launchParams.rayDirections[base+2]);
        Ray ray(origin, dir);

        uint32_t primIdx; float t, u, v;
        // cullBackfaces=false for bounce rays against additional mesh (matching SW)
        bool hit = traceMeshOptiX(launchParams.gas, ray, 1e-6f, 1e30f,
                                  TraceMode::ANY, primIdx, t, u, v);
        if (hit) {
            PreliminaryHitInfo pi; pi.t = t; pi.u = u; pi.v = v; pi.primIdx = primIdx;
            HitInfo hi = computeHitData(pi, primIdx, ray, launchParams.mesh);
            writeHitResult(sampleIdx, hi, launchParams.mesh, rp,
                           launchParams.hitPositions, launchParams.hitNormals,
                           launchParams.hitColors, launchParams.hitMaterialParams,
                           launchParams.hitFlags);
        } else {
            launchParams.hitFlags[sampleIdx] = 0;
            if (launchParams.hitMaterialParams) {
                launchParams.hitMaterialParams[base+0] = rp.material.metallic.value;
                launchParams.hitMaterialParams[base+1] = rp.material.roughness.value;
                launchParams.hitMaterialParams[base+2] = rp.material.specular.value;
            }
        }
    }
}

// ===========================================================================
// __raygen__earlyTermination
// Replaces checkBounceEarlyTerminationKernel.
// Body intentionally left as a no-op (logic is commented out in SW path too).
// ===========================================================================
extern "C" __global__ void __raygen__earlyTermination() {
    // No-op: early termination logic not yet activated.
    // HW plumbing is in place for future activation.
}
