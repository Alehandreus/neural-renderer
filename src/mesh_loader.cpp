#include "mesh_loader.h"

#include <cmath>
#include <cfloat>
#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace {

Vec3 toVec3(const aiVector3D& v) {
    return Vec3(v.x, v.y, v.z);
}

void expandBounds(const Vec3& v, Vec3* minV, Vec3* maxV) {
    minV->x = fminf(minV->x, v.x);
    minV->y = fminf(minV->y, v.y);
    minV->z = fminf(minV->z, v.z);
    maxV->x = fmaxf(maxV->x, v.x);
    maxV->y = fmaxf(maxV->y, v.y);
    maxV->z = fmaxf(maxV->z, v.z);
}

void normalizeTriangles(std::vector<Triangle>& triangles) {
    if (triangles.empty()) {
        return;
    }

    Vec3 minV(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 maxV(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const Triangle& tri : triangles) {
        expandBounds(tri.v0, &minV, &maxV);
        expandBounds(tri.v1, &minV, &maxV);
        expandBounds(tri.v2, &minV, &maxV);
    }

    Vec3 size = maxV - minV;
    float maxExtent = fmaxf(size.x, fmaxf(size.y, size.z));
    if (maxExtent <= 0.0f) {
        return;
    }

    Vec3 center = (minV + maxV) * 0.5f;
    float scale = 2.0f / maxExtent * 5.0f;
    for (Triangle& tri : triangles) {
        Vec3 v0 = (tri.v0 - center) * scale;
        Vec3 v1 = (tri.v1 - center) * scale;
        Vec3 v2 = (tri.v2 - center) * scale;
        tri = makeTriangle(v0, v1, v2);
    }
}

}  // namespace

bool LoadMeshFromFile(const std::string& path,
                      Mesh* outMesh,
                      std::string* error,
                      bool normalize) {
    if (!outMesh) {
        if (error) {
            *error = "Output mesh pointer is null.";
        }
        return false;
    }

    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_ImproveCacheLocality |
            aiProcess_PreTransformVertices |
            aiProcess_GenNormals;
    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->HasMeshes()) {
        if (error) {
            *error = importer.GetErrorString();
        }
        return false;
    }

    std::vector<Triangle> triangles;
    triangles.reserve(1024);

    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        if (!mesh || mesh->mNumFaces == 0 || mesh->mNumVertices == 0 || !mesh->mVertices || !mesh->mFaces) {
            continue;
        }

        for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
            const aiFace& face = mesh->mFaces[faceIndex];
            if (face.mNumIndices != 3) {
                continue;
            }

            Vec3 v0 = toVec3(mesh->mVertices[face.mIndices[0]]);
            Vec3 v1 = toVec3(mesh->mVertices[face.mIndices[1]]);
            Vec3 v2 = toVec3(mesh->mVertices[face.mIndices[2]]);
            triangles.push_back(makeTriangle(v0, v1, v2));
        }
    }

    if (triangles.empty()) {
        if (error) {
            *error = "No triangles found in mesh.";
        }
        return false;
    }

    if (normalize) {
        normalizeTriangles(triangles);
    }
    outMesh->setTriangles(std::move(triangles));
    return true;
}

void GenerateUvSphere(Mesh* outMesh, int stacks, int slices, float radius) {
    if (!outMesh) {
        return;
    }

    if (stacks < 2 || slices < 3) {
        outMesh->setTriangles({});
        return;
    }

    const float kPi = 3.14159265358979323846f;
    std::vector<Triangle> triangles;
    triangles.reserve(static_cast<size_t>(stacks) * static_cast<size_t>(slices) * 2);

    for (int i = 0; i < stacks; ++i) {
        float v0 = static_cast<float>(i) / static_cast<float>(stacks);
        float v1 = static_cast<float>(i + 1) / static_cast<float>(stacks);
        float phi0 = v0 * kPi;
        float phi1 = v1 * kPi;

        for (int j = 0; j < slices; ++j) {
            float u0 = static_cast<float>(j) / static_cast<float>(slices);
            float u1 = static_cast<float>(j + 1) / static_cast<float>(slices);
            float theta0 = u0 * 2.0f * kPi;
            float theta1 = u1 * 2.0f * kPi;

            Vec3 p00(
                    radius * sinf(phi0) * cosf(theta0),
                    radius * cosf(phi0),
                    radius * sinf(phi0) * sinf(theta0));
            Vec3 p01(
                    radius * sinf(phi0) * cosf(theta1),
                    radius * cosf(phi0),
                    radius * sinf(phi0) * sinf(theta1));
            Vec3 p10(
                    radius * sinf(phi1) * cosf(theta0),
                    radius * cosf(phi1),
                    radius * sinf(phi1) * sinf(theta0));
            Vec3 p11(
                    radius * sinf(phi1) * cosf(theta1),
                    radius * cosf(phi1),
                    radius * sinf(phi1) * sinf(theta1));

            triangles.push_back(makeTriangle(p00, p10, p11));
            triangles.push_back(makeTriangle(p00, p11, p01));
        }
    }

    outMesh->setTriangles(std::move(triangles));
}
