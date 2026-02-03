#pragma once

#include <string>

#include "mesh.h"

bool LoadMeshFromFile(const std::string& path, Mesh* outMesh, std::string* error, bool normalize = false, float scale = 1.0f);
bool LoadTexturedGltfFromFile(const std::string& path,
                              Mesh* outMesh,
                              std::string* error,
                              bool normalize = false,
                              bool nearestFilter = false,
                              float scale = 1.0f);
void GenerateUvSphere(Mesh* outMesh, int stacks, int slices, float radius);
