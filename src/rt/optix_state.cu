// OptiX state management: context, GAS, pipeline, SBT.
// This file is compiled as a regular CUDA translation unit (not PTX).

#include "optix_state.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

static void checkOptix(OptixResult result, const char* context) {
    if (result != OPTIX_SUCCESS) {
        std::fprintf(stderr, "OptiX error (%s): %s\n", context, optixGetErrorString(result));
        std::exit(1);
    }
}

static void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

static void checkCuDrv(CUresult result, const char* context) {
    if (result != CUDA_SUCCESS) {
        const char* str = nullptr;
        cuGetErrorString(result, &str);
        std::fprintf(stderr, "CUDA driver error (%s): %s\n", context, str ? str : "unknown");
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Log callback
// ---------------------------------------------------------------------------

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    if (level <= 3) {
        std::fprintf(stderr, "[OptiX %s] %s\n", tag, message);
    }
}

// ---------------------------------------------------------------------------
// Raygen program names — must match extern "C" __global__ names in optix_programs.cu
// ---------------------------------------------------------------------------

static const char* kRaygenNames[] = {
    "__raygen__primaryGT",
    "__raygen__bounceGT",
    "__raygen__shellEntry",
    "__raygen__shellEntryFromRays",
    "__raygen__shellExit",
    "__raygen__additionalPrimary",
    "__raygen__additionalBounce",
    "__raygen__earlyTermination",
};
static constexpr int kNumRaygen = 8;

// ---------------------------------------------------------------------------
// optixCreateState
// ---------------------------------------------------------------------------

OptixState* optixCreateState(const char* ptxSource) {
    auto* s = new OptixState();

    // --- Init OptiX ---
    checkOptix(optixInit(), "optixInit");

    CUcontext cuCtx = nullptr;  // Use current context
    OptixDeviceContextOptions ctxOpts = {};
    ctxOpts.logCallbackFunction = optixLogCallback;
    ctxOpts.logCallbackLevel    = 4;
    checkOptix(optixDeviceContextCreate(cuCtx, &ctxOpts, &s->context), "optixDeviceContextCreate");

    // --- Module ---
    OptixModuleCompileOptions moduleOpts = {};
    moduleOpts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineOpts = {};
    pipelineOpts.usesMotionBlur                   = 0;
    pipelineOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineOpts.numPayloadValues                 = 5;  // t, u, v, primIdx, hit
    pipelineOpts.numAttributeValues               = 2;  // u, v (built-in triangle attribs)
    pipelineOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOpts.pipelineLaunchParamsVariableName = "launchParamsRaw";

    char log[2048];
    size_t logSize = sizeof(log);

    checkOptix(optixModuleCreate(
        s->context, &moduleOpts, &pipelineOpts,
        ptxSource, std::strlen(ptxSource),
        log, &logSize, &s->module),
        "optixModuleCreate");
    if (logSize > 1) {
        std::fprintf(stderr, "[OptiX module log] %s\n", log);
    }

    // --- Program groups ---
    // 8 raygen + 1 miss + 1 closesthit
    std::vector<OptixProgramGroup> programGroups;
    programGroups.reserve(kNumRaygen + 2);

    // Raygen groups
    std::vector<OptixProgramGroup> raygenGroups(kNumRaygen);
    for (int i = 0; i < kNumRaygen; ++i) {
        OptixProgramGroupDesc desc = {};
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = s->module;
        desc.raygen.entryFunctionName = kRaygenNames[i];

        OptixProgramGroupOptions pgOpts = {};
        logSize = sizeof(log);
        checkOptix(optixProgramGroupCreate(
            s->context, &desc, 1, &pgOpts, log, &logSize, &raygenGroups[i]),
            "optixProgramGroupCreate raygen");
        programGroups.push_back(raygenGroups[i]);
    }

    // Miss group
    OptixProgramGroup missGroup = {};
    {
        OptixProgramGroupDesc desc = {};
        desc.kind               = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module        = s->module;
        desc.miss.entryFunctionName = "__miss__default";

        OptixProgramGroupOptions pgOpts = {};
        logSize = sizeof(log);
        checkOptix(optixProgramGroupCreate(
            s->context, &desc, 1, &pgOpts, log, &logSize, &missGroup),
            "optixProgramGroupCreate miss");
        programGroups.push_back(missGroup);
    }

    // Closesthit group
    OptixProgramGroup closesthitGroup = {};
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH            = s->module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__default";

        OptixProgramGroupOptions pgOpts = {};
        logSize = sizeof(log);
        checkOptix(optixProgramGroupCreate(
            s->context, &desc, 1, &pgOpts, log, &logSize, &closesthitGroup),
            "optixProgramGroupCreate closesthit");
        programGroups.push_back(closesthitGroup);
    }

    // --- Pipeline ---
    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 1;

    logSize = sizeof(log);
    checkOptix(optixPipelineCreate(
        s->context, &pipelineOpts, &linkOpts,
        programGroups.data(), static_cast<unsigned int>(programGroups.size()),
        log, &logSize, &s->pipeline),
        "optixPipelineCreate");
    if (logSize > 1) {
        std::fprintf(stderr, "[OptiX pipeline log] %s\n", log);
    }

    // Stack sizes — use OptiX utility helpers (API stable across OptiX 7.x–9.x)
    OptixStackSizes stackSizes = {};
    for (auto& pg : programGroups) {
        checkOptix(optixUtilAccumulateStackSizes(pg, &stackSizes, s->pipeline),
                   "optixUtilAccumulateStackSizes");
    }
    unsigned int dcsFromTraversal = 0, dcsFromState = 0, continuationSize = 0;
    checkOptix(optixUtilComputeStackSizes(
        &stackSizes,
        /*maxTraceDepth=*/1,
        /*maxCCDepth=*/0,
        /*maxDCDepth=*/0,
        &dcsFromTraversal, &dcsFromState, &continuationSize),
        "optixUtilComputeStackSizes");
    checkOptix(optixPipelineSetStackSize(
        s->pipeline,
        dcsFromTraversal, dcsFromState, continuationSize,
        /*maxTraversableGraphDepth=*/1),
        "optixPipelineSetStackSize");

    // --- SBT ---
    // Pack raygen records (one per raygen program)
    std::vector<RaygenRecord> raygenRecordsCPU(kNumRaygen);
    for (int i = 0; i < kNumRaygen; ++i) {
        checkOptix(optixSbtRecordPackHeader(raygenGroups[i], &raygenRecordsCPU[i]),
                   "optixSbtRecordPackHeader raygen");
    }

    CUdeviceptr dRaygenRecords = 0;
    size_t raygenRecordBytes = sizeof(RaygenRecord) * kNumRaygen;
    checkCuDrv(cuMemAlloc(&dRaygenRecords, raygenRecordBytes), "cuMemAlloc raygen SBT");
    checkCuDrv(cuMemcpyHtoD(dRaygenRecords, raygenRecordsCPU.data(), raygenRecordBytes),
               "cuMemcpyHtoD raygen SBT");

    MissRecord missRecordCPU = {};
    checkOptix(optixSbtRecordPackHeader(missGroup, &missRecordCPU), "optixSbtRecordPackHeader miss");
    CUdeviceptr dMissRecord = 0;
    checkCuDrv(cuMemAlloc(&dMissRecord, sizeof(MissRecord)), "cuMemAlloc miss SBT");
    checkCuDrv(cuMemcpyHtoD(dMissRecord, &missRecordCPU, sizeof(MissRecord)), "cuMemcpyHtoD miss SBT");

    HitgroupRecord hitRecordCPU = {};
    checkOptix(optixSbtRecordPackHeader(closesthitGroup, &hitRecordCPU), "optixSbtRecordPackHeader hit");
    CUdeviceptr dHitRecord = 0;
    checkCuDrv(cuMemAlloc(&dHitRecord, sizeof(HitgroupRecord)), "cuMemAlloc hit SBT");
    checkCuDrv(cuMemcpyHtoD(dHitRecord, &hitRecordCPU, sizeof(HitgroupRecord)), "cuMemcpyHtoD hit SBT");

    // Store base; per-launch code sets sbt.raygenRecord = dRaygenRecordsBase + i*stride
    s->dRaygenRecordsBase              = dRaygenRecords;
    s->sbt.raygenRecord                = dRaygenRecords;
    s->sbt.missRecordBase              = dMissRecord;
    s->sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    s->sbt.missRecordCount             = 1;
    s->sbt.hitgroupRecordBase          = dHitRecord;
    s->sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    s->sbt.hitgroupRecordCount         = 1;

    // Allocate launch params buffer on device
    checkCuDrv(cuMemAlloc(&s->dLaunchParams, sizeof(OptixLaunchParams)), "cuMemAlloc launchParams");

    // Destroy program groups (no longer needed after pipeline creation)
    for (auto& pg : programGroups) {
        optixProgramGroupDestroy(pg);
    }

    return s;
}

// ---------------------------------------------------------------------------
// optixDestroyState
// ---------------------------------------------------------------------------

void optixDestroyState(OptixState* state) {
    if (!state) return;

    // Free GAS buffers
    for (OptixGas* gas : {&state->gasClassic, &state->gasOuter, &state->gasInner, &state->gasAdditional}) {
        optixDestroyGas(*gas);
    }

    if (state->dLaunchParams) cuMemFree(state->dLaunchParams);

    // Free SBT device memory (raygenRecord == dRaygenRecordsBase, freed once)
    if (state->dRaygenRecordsBase) cuMemFree(state->dRaygenRecordsBase);
    if (state->sbt.missRecordBase) cuMemFree(state->sbt.missRecordBase);
    if (state->sbt.hitgroupRecordBase) cuMemFree(state->sbt.hitgroupRecordBase);

    if (state->pipeline) optixPipelineDestroy(state->pipeline);
    if (state->module)   optixModuleDestroy(state->module);
    if (state->context)  optixDeviceContextDestroy(state->context);

    delete state;
}

// ---------------------------------------------------------------------------
// optixBuildGas
// ---------------------------------------------------------------------------

void optixBuildGas(OptixState* state,
                   OptixGas& gas,
                   CUdeviceptr deviceVertices,
                   int numVertices,
                   CUdeviceptr deviceIndices,
                   int numTriangles) {
    if (!state || !deviceVertices || !deviceIndices || numTriangles <= 0) return;

    // Destroy previous GAS if present
    optixDestroyGas(gas);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    buildInput.triangleArray.numVertices         = static_cast<unsigned int>(numVertices);
    buildInput.triangleArray.vertexBuffers       = &deviceVertices;

    buildInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes  = sizeof(unsigned int) * 3;
    buildInput.triangleArray.numIndexTriplets    = static_cast<unsigned int>(numTriangles);
    buildInput.triangleArray.indexBuffer         = deviceIndices;

    unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags         = &geomFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOpts = {};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    checkOptix(optixAccelComputeMemoryUsage(state->context, &accelOpts, &buildInput, 1, &sizes),
               "optixAccelComputeMemoryUsage");

    CUdeviceptr dTemp = 0, dOutput = 0, dCompactedSize = 0;
    checkCuDrv(cuMemAlloc(&dTemp,   sizes.tempSizeInBytes),   "cuMemAlloc GAS temp");
    checkCuDrv(cuMemAlloc(&dOutput, sizes.outputSizeInBytes), "cuMemAlloc GAS output");

    // Emit compacted size so we can shrink the buffer
    OptixAccelEmitDesc emitDesc = {};
    checkCuDrv(cuMemAlloc(&dCompactedSize, sizeof(uint64_t)), "cuMemAlloc compactedSize");
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = dCompactedSize;

    OptixTraversableHandle tempHandle = 0;
    checkOptix(optixAccelBuild(
        state->context, /*stream=*/nullptr,
        &accelOpts, &buildInput, 1,
        dTemp, sizes.tempSizeInBytes,
        dOutput, sizes.outputSizeInBytes,
        &tempHandle, &emitDesc, 1),
        "optixAccelBuild");

    checkCuda(cudaDeviceSynchronize(), "GAS build sync");

    uint64_t compactedSize = 0;
    checkCuDrv(cuMemcpyDtoH(&compactedSize, dCompactedSize, sizeof(uint64_t)), "cuMemcpyDtoH compactedSize");

    // Compact
    checkCuDrv(cuMemAlloc(&gas.buffer, compactedSize), "cuMemAlloc GAS compact");
    gas.bufferSize = compactedSize;
    checkOptix(optixAccelCompact(state->context, nullptr, tempHandle, gas.buffer, compactedSize, &gas.handle),
               "optixAccelCompact");

    checkCuda(cudaDeviceSynchronize(), "GAS compact sync");

    cuMemFree(dTemp);
    cuMemFree(dOutput);
    cuMemFree(dCompactedSize);
}

// ---------------------------------------------------------------------------
// optixDestroyGas
// ---------------------------------------------------------------------------

void optixDestroyGas(OptixGas& gas) {
    if (gas.buffer) {
        cuMemFree(gas.buffer);
        gas.buffer     = 0;
        gas.bufferSize = 0;
        gas.handle     = 0;
    }
}
