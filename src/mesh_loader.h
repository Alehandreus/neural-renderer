#pragma once

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
