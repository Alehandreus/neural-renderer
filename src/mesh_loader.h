#pragma once

#include <string>

#include "mesh.h"

bool LoadMeshFromFile(const std::string& path, Mesh* outMesh, std::string* error, bool normalize = false);
void GenerateUvSphere(Mesh* outMesh, int stacks, int slices, float radius);
