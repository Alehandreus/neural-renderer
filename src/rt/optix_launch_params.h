#pragma once

// Shared host+device header: included by optix_state.cu (host side) and
// optix_programs.cu (device side, compiled to PTX).
//
// All structs must be trivially copyable and have the same layout on host and device.

#include <optix.h>

// Pull in MeshDeviceView and RenderParams.
// When compiled as PTX (optix_programs.cu) these come from the same src/ tree.
#include "../mesh.h"
#include "../render_params.h"

// ---------------------------------------------------------------------------
// Ray payload: passed through optixTrace registers (5 uint32 values).
// ---------------------------------------------------------------------------
struct RayPayload {
    float    t;       // Hit distance (or 1e30 on miss)
    float    u, v;    // Barycentric coords
    uint32_t primIdx; // Triangle index
    uint32_t hit;     // 1 = hit, 0 = miss
};

// ---------------------------------------------------------------------------
// Per-launch parameters — uploaded to device once per optixLaunch call.
// Access on device via optixGetLaunchParamsBuffer.
// ---------------------------------------------------------------------------
struct OptixLaunchParams {
    // Which GAS to trace against (mesh-specific handles)
    OptixTraversableHandle gas;   // Primary mesh for this launch
    OptixTraversableHandle gas2;  // Secondary mesh (shellExit needs outer+inner)

    RenderParams renderParams;

    // Input rays / masks (same semantics as corresponding SW kernel params)
    const float* rayOrigins;
    const float* rayDirections;
    float*       entryPositions;
    const int*   rayActiveMask;
    const float* rayPdfs;
    const int*   hitIndices;
    int          hitCount;

    // Primary-hit output buffers
    float* hitPositions;
    float* hitNormals;
    float* hitColors;
    float* hitMaterialParams;
    int*   hitFlags;
    float* hitDistances;

    // Shell-segment state buffers
    float* entryT;
    float* storedRayDirections;
    float* outerExitT;
    float* innerEnterT;
    int*   innerHitFlags;
    float* accumT;
    int*   activeFlags;

    // Mesh attribute pointers for computeHitData in raygen/closesthit
    MeshDeviceView mesh;
    MeshDeviceView mesh2;
};

// ---------------------------------------------------------------------------
// SBT record headers (standard OptiX boilerplate, no per-record data needed)
// ---------------------------------------------------------------------------
struct RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct HitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
