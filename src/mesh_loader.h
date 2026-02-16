#pragma once

#include <cstdio>
#include <string>

#include "mesh.h"

// Load GLTF/GLB file with full material and texture support using TinyGLTF
// Materials, textures, and per-primitive material mapping are loaded from the file
// Sets mesh.hasMeshMaterials_ = true
bool LoadGltfWithMaterials(const std::string& path,
                           Mesh* outMesh,
                           std::string* error,
                           bool normalize = false,
                           float scale = 1.0f);

// Load mesh from OBJ/FBX/other formats using Assimp
// No textures or materials are loaded - use global material from config/ImGui
// Sets mesh.hasMeshMaterials_ = false
bool LoadMeshFromFile(const std::string& path,
                      Mesh* outMesh,
                      std::string* error,
                      bool normalize = false,
                      float scale = 1.0f);

// Auto-detect format and load appropriately:
// - .gltf/.glb -> LoadGltfWithMaterials
// - others -> LoadMeshFromFile
bool LoadMeshAuto(const std::string& path,
                  Mesh* outMesh,
                  std::string* error,
                  bool normalize = false,
                  float scale = 1.0f);

// Generate a UV sphere mesh (for testing/default geometry)
void GenerateUvSphere(Mesh* outMesh, int stacks, int slices, float radius);

// Convenience wrapper: auto-load a mesh with an error label and optional nearest-neighbor textures.
inline bool LoadMeshLabeled(const char* path, Mesh* mesh, const char* label,
                             bool normalize, bool nearestTex, float scale = 1.0f) {
    if (!path || path[0] == '\0') return false;
    std::string loadError;
    bool loaded = LoadMeshAuto(path, mesh, &loadError, normalize, scale);
    if (loaded) {
        mesh->setTextureNearest(nearestTex);
    } else {
        std::fprintf(stderr, "Failed to load %s mesh '%s': %s\n", label, path, loadError.c_str());
    }
    return loaded;
}
