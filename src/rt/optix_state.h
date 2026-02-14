#pragma once

// Host-only OptiX state.  Device code (optix_programs.cu) does NOT include
// this header — it only needs optix_launch_params.h.

#include <cuda.h>
#include <optix.h>

#include "optix_launch_params.h"

// ---------------------------------------------------------------------------
// Per-mesh GAS (Geometry Acceleration Structure)
// ---------------------------------------------------------------------------
struct OptixGas {
    OptixTraversableHandle handle     = 0;
    CUdeviceptr            buffer     = 0;
    size_t                 bufferSize = 0;
};

// ---------------------------------------------------------------------------
// Top-level OptiX state owned by RendererNeural
// ---------------------------------------------------------------------------
struct OptixState {
    OptixDeviceContext      context   = nullptr;
    OptixModule             module    = nullptr;
    OptixPipeline           pipeline  = nullptr;
    OptixShaderBindingTable sbt       = {};

    // One GAS per mesh role
    OptixGas gasClassic;
    OptixGas gasOuter;
    OptixGas gasInner;
    OptixGas gasAdditional;

    // Per-launch params buffer (device)
    CUdeviceptr dLaunchParams = 0;

    // Base of the raygen SBT records array.
    // Set sbt.raygenRecord = dRaygenRecordsBase + index * sizeof(RaygenRecord)
    // before each optixLaunch to select a specific program.
    CUdeviceptr dRaygenRecordsBase = 0;
};

// Raygen program indices — must match kRaygenNames[] order in optix_state.cu.
enum class RaygenIndex : unsigned int {
    PrimaryGT          = 0,
    BounceGT           = 1,
    ShellEntry         = 2,
    ShellEntryFromRays = 3,
    ShellExit          = 4,
    AdditionalPrimary  = 5,
    AdditionalBounce   = 6,
    EarlyTermination   = 7,
};

// ---------------------------------------------------------------------------
// Public API — called from RendererNeural
// ---------------------------------------------------------------------------

// Initialise OptiX context, module, pipeline and SBT from embedded PTX source.
// ptxSource must be a null-terminated string of PTX text.
OptixState* optixCreateState(const char* ptxSource);

// Destroy all OptiX objects and free memory.
void optixDestroyState(OptixState* state);

// Build (or rebuild) a GAS for the geometry in meshView.
// deviceVertices and deviceIndices must already be uploaded (uploadToDevice was called).
void optixBuildGas(OptixState* state,
                   OptixGas& gas,
                   CUdeviceptr deviceVertices,
                   int numVertices,
                   CUdeviceptr deviceIndices,
                   int numTriangles);

// Destroy a single GAS and free its device buffer.
void optixDestroyGas(OptixGas& gas);
