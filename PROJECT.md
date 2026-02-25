# cuda-rendering: Neural Mesh Rendering — Project Reference

## High-Level Overview

This project is a **neural surface rendering framework** that replaces traditional mesh geometry with a learned neural implicit surface inside a volumetric shell. The core idea:

- A 3D object is enclosed in two shells (inner, outer).
- A neural network learns to predict whether a ray segment through those shells hits the surface, and if so, at what distance and with what normal.
- At runtime, rays are cast against the shell geometry; the hit segment is fed into the neural network instead of testing against the full mesh.
- Bounce/secondary rays use the ground-truth mesh for accurate lighting.

This allows high-quality rendering of complex geometry with compressed, fast neural inference instead of large BVH traversal.

---

## Architecture Map

```
JSON Config ──► Scene (4 meshes + env map)
                 │
                 ▼
           RendererNeural
           ├── Primary rays ──► outer shell (GT or neural)
           │                    └── neural: network query (entry, inner, dir → hit, dist, normal)
           │                    └── GT: mesh BVH / OptiX GAS
           ├── Bounce rays  ──► GT mesh (BVH / OptiX)
           │   └── Disney BRDF importance sampling
           ├── Accumulation buffer (HDR float)
           └── Tonemapping + denoising → sRGB pixels
                                           │
                             ImGui viewer / evaluate tool
```

---

## Directory Structure

```
cuda-rendering/
├── src/                        # All source code
│   ├── vec3.h                  # Math primitives
│   ├── ray.h                   # Ray struct
│   ├── hit_info.h              # Intersection result structs
│   ├── bvh_data.h              # BVH node layout
│   ├── material.h              # Disney PBR material params
│   ├── render_params.h         # Per-launch camera/scene state
│   ├── mesh.h / mesh.cu        # Mesh: CPU+GPU data, BVH, OptiX GAS
│   ├── scene.h / scene.cu      # Scene: 4 meshes + environment map
│   ├── mesh_bvh.cpp            # Software BVH construction (SAH)
│   ├── mesh_loader.h/.cpp      # GLTF/OBJ/FBX loading
│   ├── config_loader.h/.cpp    # JSON scene config I/O
│   ├── mesh_intersection.cuh   # Ray-triangle, material, texture sampling
│   ├── mesh_traversal.cuh      # BVH traversal, primary ray gen, RNG
│   ├── disney_brdf.cuh         # Disney principled BRDF eval + sampling
│   ├── denoiser.cuh            # Joint bilateral denoiser (normal-guided)
│   ├── image_utils.h           # PNG/EXR I/O, PSNR, FLIP metrics
│   ├── cuda_renderer_neural.h  # RendererNeural class interface
│   ├── cuda_renderer_neural.cu # Main renderer: kernels + neural inference
│   ├── input_controller.h/.cpp # WASD + mouse camera controller
│   ├── viewer.cu               # Interactive ImGui viewer (main app)
│   ├── evaluate.cu             # Batch GT vs neural evaluation tool
│   ├── compare_images.cu       # Standalone image comparison tool
│   └── rt/                     # OptiX hardware ray tracing
│       ├── optix_launch_params.h   # Shared host+device params struct
│       ├── optix_state.h/.cu       # Host: context, pipeline, SBT
│       └── optix_programs.cu       # Device: raygen, closesthit, miss
├── cmake/                      # PTX compilation + embedding helpers
├── ext/                        # Git submodules (see Dependencies)
├── plans/                      # Design documents
├── models.py                   # PyTorch model definition (training)
├── *.json                      # Scene configs (camera, chess, statuette…)
└── CMakeLists.txt              # Build system
```

---

## Data Structures

### Math (`vec3.h`)
- **`Vec3`** — 3D float vector; host+device; arithmetic operators, dot, cross, normalize, lerp.
- **`Vec2`** — 2D float vector for UVs.

### Ray (`ray.h`)
```cpp
struct Ray { Vec3 origin, direction; Vec3 at(float t); };
```

### Intersection (`hit_info.h`)
- **`PreliminaryHitInfo`** — minimal BVH result: `t`, `u`, `v`, `primIdx`.
- **`HitInfo`** — full result: position, geometric/shading normal, tangent, UV, material params.

### BVH (`bvh_data.h`)
```cpp
struct BvhNode { Vec3 boundsMin, boundsMax; int left, right, first, count, isLeaf; };
```
Used for the iterative software BVH traversal stack.

### Material (`material.h`)
- **`MaterialParam`** — scalar value that is either a constant float or a UV-sampled texture.
- **`MaterialParamVec3`** — same for RGB.
- **`Material`** — full Disney PBR: metallic, roughness, specular, anisotropy, sheen, clearcoat, IOR, transmission, emission, normal map, base color.

### Render State (`render_params.h`)
```cpp
struct RenderParams {
    Vec3 camPos, camForward, camRight, camUp;
    Vec3 lightDir, outerShellMin, outerShellInvExtent;
    Material material;
    int width, height, samplesPerPixel, maxBounces;
    float fovY, maxRadiance, sceneScale;
    bool useConstantNeuralColor, useDirectEnvColor;
};
```
Uploaded to the GPU once per frame; accessed in all kernels.

### Scene Shell Structure (`scene.h`)
```cpp
class Scene {
    Mesh originalMesh_;    // main reference geometry (GT rendering & bounces)
    Mesh innerShell_;      // inner neural sampling surface
    Mesh outerShell_;      // outer neural sampling surface
    Mesh additionalMesh_;  // auxiliary geometry
    EnvironmentMap envMap_;
};
```
- **`EnvironmentMap`**: RGBE HDR loaded to device texture; supports bilinear filtering + yaw rotation.

---

## Neural Network

### What It Replaces
Instead of tracing rays against the full `originalMesh_`, rays are traced against the `outerShell_`. The hit segment (ray from outer shell entry through inner shell) is fed to the neural network.

### Input / Output
| Slot | Meaning | Dim |
|------|---------|-----|
| entry XYZ | outer shell intersection point | 3 |
| inner XYZ | inner shell intersection point | 3 |
| direction XYZ | normalized ray direction | 3 |
| **→ presence** | does ray hit surface? | 1 |
| **→ distance** | distance along segment to hit | 1 |
| **→ normal** | surface normal at hit | 3 |

### Architecture
- **3× HashGrid encodings** (one per XYZ input, learned spatial features).
- **Spherical Harmonics** encoding for direction.
- **Tiny-cuda-nn FP16 MLP** (fully-fused; fast).
- Loaded from a binary checkpoint at runtime.

### Training (`models.py`)
Defines the PyTorch model with the same architecture. Weights are exported to `.bin` and loaded by the C++ runtime via tiny-cuda-nn.

---

## Rendering Pipeline

### Primary Rays

**Ground-truth mode:** standard BVH/OptiX traversal against `originalMesh_`.

**Neural mode:**
1. Cast ray against `outerShell_` → get entry point.
2. Cast ray against `innerShell_` → get exit point.
3. Query neural net → `(presence, distance, normal)`.
4. If `presence`, set hit at `entry + distance * dir`.

### Wavefront Path Tracing (`cuda_renderer_neural.cu`)

All active rays are processed in parallel. Per-bounce state lives in flat GPU arrays (`pathThroughput`, `pathRadiance`, `pathActive`, `bounceOrigins`, etc.).

**Per-sample loop:**

```
resetAccum
for sample in [0, samplesPerPixel):
    generatePrimaryRays
    intersect:
        GT mode:     intersectGroundTruthKernel (originalMesh_ BVH/OptiX)
        Neural mode: traceNeuralSegmentsForRays (outer→inner shell, network query)
                     + traceAdditionalMeshPrimaryRaysKernel, selectClosestPrimaryHitKernel
    initializePathState (throughput=1, sample env on miss, read albedo/normal)
    for bounce in [1, maxBounces]:
        sampleBounceDirectionsKernel   (Disney BRDF sample)
        checkBounceEarlyTerminationKernel  (neural only: kill ray if re-entering shell)
        traceBounces:
            GT mode:     traceGroundTruthBouncesKernel (originalMesh_)
            Neural mode: traceNeuralSegmentsForRays + traceAdditionalMeshRaysKernel
                         + selectClosestHitKernel
        integrateBounceKernel  (update throughput, env sample on miss, Russian roulette)
        swap hit buffers for next bounce
    addToAccum
finalizePathTracingKernel (tonemap + sRGB)
```

### Kernel Summary

| Kernel | Purpose |
|--------|---------|
| `intersectGroundTruthKernel` | Trace primary rays vs GT mesh BVH |
| `traceOuterShellEntryKernel` | Find outer shell entry for primary neural rays |
| `traceOuterShellEntryFromRaysKernel` | Same but for arbitrary bounce rays (handles inside-shell start) |
| `traceSegmentExitsKernel` | Find outer-exit and inner-enter per active ray |
| `traceGroundTruthBouncesKernel` | Trace bounce rays vs GT mesh (GT mode) |
| `traceAdditionalMeshPrimaryRaysKernel` / `traceAdditionalMeshRaysKernel` | Trace vs additionalMesh_ |
| `checkBounceEarlyTerminationKernel` | Neural mode: terminate bounce if it re-enters shell region |
| `initializePathStateKernel` | Set throughput=1, sample env on miss, read hit albedo/normal |
| `sampleBounceDirectionsKernel` | Disney BRDF importance sample; write bounce ray + BRDF weight |
| `integrateBounceKernel` | Update throughput, env sample on miss, Russian roulette |
| `selectClosestPrimaryHitKernel` / `selectClosestHitKernel` | Merge neural + additional mesh hits |
| `finalizePathTracingKernel` | Average accum, tonemap, sRGB encode |

### OptiX Path (`rt/`)

When `USE_OPTIX=ON`, hardware BVH replaces the software traversal:

- **`optix_state.cu`** — builds OptiX context, loads PTX, creates GAS per mesh, builds pipeline and SBT.
- **`optix_programs.cu`** — device programs:
  - `raygen*` — loops over rays, calls `optixTrace`.
  - `closesthit` — writes `t, u, v, primIdx` to payload.
  - `miss` — marks ray as no-hit.
- Multiple raygen programs for different trace modes: `PrimaryGT`, `BounceGT`, `ShellEntry`, `ShellExit`.

**OptiX PTX gotcha (already resolved):** must compile with `compute_89` virtual arch and `CUDA_ARCHITECTURES OFF` on the PTX target. See MEMORY.md for details.

---

## Key Algorithms

### Disney Principled BRDF (`disney_brdf.cuh`)
Full implementation:
- **Diffuse:** Lambertian with retroreflective correction.
- **Specular:** GGX microfacet (isotropic + anisotropic via GTR2).
- **Clearcoat:** GTR1 distribution.
- **Fresnel:** Schlick for conductor, dielectric for transmission.
- **Sampling:** mixture of 3 lobes; returns direction, PDF, and weight.
- `sampleDisneyBrdf()` — draw one sample.
- `evalDisneyBrdf()` — evaluate at given direction.

### BVH Construction (`mesh_bvh.cpp`)
- Uses `bvh::v2` library with SAH splits and high-quality settings.
- Reorders triangles and material maps to BVH leaf order for coherent access.
- Returns flat `BvhNode[]` suitable for iterative GPU traversal.

### BVH Traversal (`mesh_traversal.cuh`)
- Iterative DFS with an explicit stack (no recursion).
- AABB test: slab method with per-axis overlap.
- `TraceMode`: `FORWARD_ONLY` (front faces / entry), `BACKWARD_ONLY` (exit), `ANY` (closest).
- `traceMeshSoftware()` — returns `PreliminaryHitInfo`.

### Ray-Triangle Intersection (`mesh_intersection.cuh`)
- Möller–Trumbore algorithm.
- `computeHitInfo()` — barycentric interpolation of normals/UVs; tangent frame for normal maps.
- `sampleTexture()` — bilinear/nearest with wrap-around UV.
- `resolveMaterialParam()` — constant vs texture lookup.

### Denoiser (`denoiser.cuh`)
Joint bilateral filter guided by primary-hit normals and albedo:
- Spatial kernel radius = 5, σ_spatial = 3.
- Normal weight: `dot(n_i, n_j)^64`.
- Albedo weight: Gaussian on albedo difference.
- Reads linear HDR accumulation; writes sRGB.

---

## Build Targets

| Target | Description |
|--------|-------------|
| `viewer` | Interactive ImGui app with real-time rendering |
| `evaluate` | Batch render → PSNR + FLIP metrics vs reference |
| `compare_images` | Standalone image quality comparison |

```bash
cmake -B build -DUSE_OPTIX=ON -DOPTIX_PTX_ARCH=compute_89
cmake --build build --target viewer
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_OPTIX` | `OFF` | Enable OptiX hardware ray tracing |
| `OPTIX_PTX_ARCH` | `compute_89` | Virtual CUDA arch for PTX (do not use `compute_120`; see MEMORY.md) |
| `PROFILE_KERNELS` | `OFF` | Per-kernel GPU timing (zero overhead when OFF) |

With `PROFILE_KERNELS=ON`, `RendererNeural` records CUDA events around every major kernel dispatch. At the end of each frame, elapsed times are accumulated by kernel category and exposed via `lastFrameTimings()`. The viewer shows a **GPU Kernel Timings** ImGui panel (top-right) listing per-kernel total ms and ns/ray for the last frame.

Kernel categories (`KernelId` enum in `cuda_renderer_neural.h`):

| Category | When active |
|----------|-------------|
| GT primary intersect | GT mode: primary rays vs `originalMesh_` |
| GT bounce intersect | GT mode: bounce rays ×bounces |
| Shell intersection | Neural mode: all shell BVH traversals (outer entry + segment exits ×iters) |
| Neural forward pass | Neural mode: `network_->inference_mixed_precision` ×iters |
| Additional mesh (primary/bounce) | Hybrid: trace vs `additionalMesh_` |
| Select closest (primary/bounce) | Merge neural + additional mesh hits |
| Init path state | Per sample: throughput init, env miss |
| Sample bounce dirs | Disney BRDF sampling ×bounces |
| Bounce early term. | Neural mode: early termination check ×bounces |
| Integrate bounce | Throughput update, Russian roulette ×bounces |
| Finalize / lambert | Tonemap + sRGB encode |

```bash
cmake -B build -DUSE_OPTIX=ON -DOPTIX_PTX_ARCH=compute_89 -DPROFILE_KERNELS=ON
cmake --build build --target viewer
```

---

## Config Format (JSON)

```jsonc
{
  "scene": {
    "original_mesh": { "path": "mesh.glb", "scale": 1.0 },
    "inner_shell":   { "path": "inner.obj" },
    "outer_shell":   { "path": "outer.obj" },
    "additional_mesh": { "path": "floor.glb" }
  },
  "environment": {
    "hdri_path": "env.hdr",
    "rotation": 0.0,       // yaw in radians
    "strength": 1.0
  },
  "checkpoint_path": "weights.bin",
  "camera": {
    "matrix": [ /* 16 floats, column-major 4×4 */ ],
    "yfov": 0.785           // radians (~45°)
  },
  "rendering": {
    "total_samples": 2048,
    "bounce_count": 3,
    "width": 1920,    // initial window width in pixels
    "height": 1080    // initial window height in pixels
  },
  "material": { /* optional overrides */ }
}
```

Saved camera JSON files additionally store `position`, `yaw`, `pitch` for convenience.

---

## Dependencies (ext/)

| Library | Role |
|---------|------|
| **tiny-cuda-nn** | FP16 neural network (encoding + MLP) |
| **bvh** | Software BVH construction + traversal |
| **glfw** | Window / OpenGL context |
| **imgui** | UI framework |
| **assimp** | OBJ/FBX loading |
| **tinygltf** | GLTF/GLB + PBR materials |
| **tinyexr** | EXR image I/O |
| **stb** | PNG/JPG I/O |
| **flip-cuda** | FLIP perceptual image metric |
| **nfd** | Native file dialogs |
| **OptiX 9.0** | Hardware RT (optional, `/opt/optix/include`) |

---

## File-by-File Function Reference

### `mesh.cu`
- `Mesh::loadFromFile()` — dispatch to GLTF or Assimp loader.
- `Mesh::uploadToDevice()` — sync CPU buffers to GPU if dirty.
- `Mesh::buildBvh()` — call `mesh_bvh.cpp`, store flat node array.
- `Mesh::buildGAS()` — build OptiX GAS from vertex/index buffers.
- `Mesh::deviceView()` — return lightweight struct of raw device pointers.

### `scene.cu`
- `Scene::loadFromConfig()` — load all meshes + env map from `RendererConfig`.
- `Scene::uploadAll()` — call `uploadToDevice()` + `buildBvh()` on each mesh.
- `EnvironmentMap::load()` — read RGBE HDR, upload to CUDA texture.
- `EnvironmentMap::sample()` — equirectangular lookup with rotation.

### `config_loader.cpp`
- `loadConfig(path)` → `RendererConfig` — parse JSON, resolve relative paths.
- `saveConfig(path, config)` — write JSON.
- `cameraMatrixToViewParams()` / `viewParamsToCameraMatrix()` — convert between 4×4 matrix and (pos, yaw, pitch).

### `mesh_loader.cpp`
- `loadMeshGLTF(path)` → `Mesh` — TinyGLTF parse; extract PBR materials, textures, normals, UVs.
- `loadMeshAssimp(path)` → `Mesh` — Assimp parse; geometry only.

### `cuda_renderer_neural.cu`
- `RendererNeural::RendererNeural()` — allocate device buffers, load network checkpoint.
- `RendererNeural::render(scene, params)` — top-level entry; runs full sample loop.
- `RendererNeural::traceNeuralSegmentsForRays()` — batch neural network query for a set of rays.
- `RendererNeural::resizeBuffers(w, h)` — reallocate per-pixel state on resolution change.
- `RendererNeural::saveImage(path)` — write current accumulation buffer to PNG/EXR.

### `viewer.cu`
- `main()` — create GLFW window, OpenGL context, ImGui; load scene; enter render loop.
- `renderLoop()` — per-frame: handle input → call `renderer.render()` → blit pixels via CUDA-GL interop.
- `drawUI()` — ImGui panels for neural toggle, sample/bounce count, material sliders, camera info.

### `evaluate.cu`
- `main()` — load config, render GT + neural, compute PSNR + FLIP, save images.
- Progress bar with ETA printed to stdout.

### `input_controller.cpp`
- `InputController::update(dt)` — read GLFW key/mouse state; update camera position + yaw/pitch.
- Captures/releases mouse on ESC.

### `rt/optix_state.cu`
- `OptixState::init()` — create context, load PTX module, create raygen/CH/miss program groups.
- `OptixState::buildGAS(mesh)` — build geometry acceleration structure.
- `OptixState::buildPipeline()` — link programs, set stack sizes.
- `OptixState::buildSBT()` — fill shader binding table with per-program data.
- `OptixState::launch(params, w, h)` — upload params, call `optixLaunch`.

### `rt/optix_programs.cu`
- `__raygen__primaryGT()` — generate primary rays, call `optixTrace`, write result.
- `__raygen__bounceGT()` — read bounce ray buffer, trace, write hit.
- `__raygen__shellEntry/Exit()` — trace against shell geometry for neural segment endpoints.
- `__closesthit__mesh()` — pack `t, u, v, primIdx` into payload registers.
- `__miss__()` — mark no-hit.

---

## GPU Memory Layout

Per-pixel wavefront state (all flat arrays of size `width × height`):

| Buffer | Type | Contents |
|--------|------|---------|
| `accum` | `Vec3[]` | HDR accumulated radiance |
| `devicePixels` | `uchar4[]` | sRGB display output |
| `hitPositions` | `Vec3[]` | primary hit world pos (×3 for entry/mid/inner) |
| `hitNormals` | `Vec3[]` | shading normals |
| `hitColors` | `Vec3[]` | albedo / base color |
| `hitMaterialParams` | `float3[]` | metallic, roughness, specular |
| `hitFlags` | `int[]` | valid hit bitmask |
| `pathThroughput` | `Vec3[]` | current path weight |
| `pathRadiance` | `Vec3[]` | accumulated emission |
| `pathActive` | `int[]` | 1 if path still alive |
| `bounceOrigins/Directions` | `Vec3[]` | next ray to trace |
| `bouncePDFs / brdfWeights` | `float/Vec3[]` | for MIS or direct accumulation |

Network I/O buffers:
- `networkInputs_` — FP32 batch inputs.
- `outputs_` — FP16 MLP outputs (presence, distance, normal).

---

## Known Issues / Notes

- `__constant__ launchParamsRaw[sizeof(OptixLaunchParams)]` pattern used because `Vec3` has a non-trivial constructor (see MEMORY.md).
- OptiX GAS build for the outer/inner shells is separate from the main mesh GAS.
