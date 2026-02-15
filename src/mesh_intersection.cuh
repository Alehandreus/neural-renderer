#pragma once

#include "hit_info.h"
#include "mesh.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"

// =============================================================================
// Ray-Triangle Intersection for Indexed Geometry
// =============================================================================

// Intersect ray with a single triangle (Möller–Trumbore)
// Returns minimal hit data (t, u, v)
__device__ inline PreliminaryHitInfo intersectTriangleIndexed(
    const Ray& ray,
    const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
    PreliminaryHitInfo pi;
    pi.t = 1e30f;

    const float kEpsilon = 1e-8f;
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = cross(ray.direction, e2);
    float det = dot(e1, pvec);

    if (fabsf(det) < kEpsilon) return pi;

    float invDet = 1.0f / det;
    Vec3 tvec = ray.origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return pi;

    Vec3 qvec = cross(tvec, e1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return pi;

    float t = dot(e2, qvec) * invDet;
    if (t <= kEpsilon) return pi;

    pi.t = t;
    pi.u = u;
    pi.v = v;
    return pi;
}

// =============================================================================
// Material Lookup
// =============================================================================

// Binary search to find primitive index containing given triangle
__device__ inline uint32_t findPrimitiveForTriangle(
    uint32_t triIdx,
    const uint32_t* materialMap,
    int numPrimitives)
{
    if (numPrimitives <= 0 || !materialMap) return 0;

    uint32_t left = 0, right = static_cast<uint32_t>(numPrimitives);
    while (left < right) {
        uint32_t mid = left + (right - left) / 2;
        if (materialMap[mid] <= triIdx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left > 0 ? left - 1 : 0;
}

__device__ inline int getMaterialIdForTriangle(
    uint32_t triIdx,
    const MeshDeviceView& mesh)
{
    if (!mesh.hasMeshMaterials || mesh.numPrimitives <= 0) {
        return -1;  // Use global material
    }

    uint32_t primIdx = findPrimitiveForTriangle(triIdx, mesh.materialMap, mesh.numPrimitives);
    if (primIdx < static_cast<uint32_t>(mesh.numPrimitives) && mesh.materialIds) {
        return mesh.materialIds[primIdx];
    }
    return -1;
}

// =============================================================================
// Orthonormal Basis for TBN
// =============================================================================

__device__ inline void orthoBasis(Vec3& tangent, Vec3& bitangent, const Vec3& normal)
{
    bitangent = Vec3(0.0f, 0.0f, 0.0f);
    if (fabsf(normal.x) < 0.6f) {
        bitangent.x = 1.0f;
    } else if (fabsf(normal.y) < 0.6f) {
        bitangent.y = 1.0f;
    } else {
        bitangent.z = 1.0f;
    }
    tangent = normalize(cross(bitangent, normal));
    bitangent = normalize(cross(normal, tangent));
}

// =============================================================================
// Texture Sampling Helpers
// =============================================================================

__device__ __forceinline__ Vec3 sampleTextureRawDev(const TextureDeviceView& tex,
                                           float u, float v,
                                           bool nearestFilter)
{
    if (!tex.pixels || tex.width <= 0 || tex.height <= 0 || tex.channels <= 0) {
        return Vec3(-1.0f, -1.0f, -1.0f);
    }

    u = u - floorf(u);
    v = v - floorf(v);

    auto fetch = [&](int xi, int yi) {
        int idx = (yi * tex.width + xi) * tex.channels;
        float r = tex.channels > 0 ? tex.pixels[idx + 0] * (1.0f / 255.0f) : 0.0f;
        float g = tex.channels > 1 ? tex.pixels[idx + 1] * (1.0f / 255.0f) : 0.0f;
        float b = tex.channels > 2 ? tex.pixels[idx + 2] * (1.0f / 255.0f) : 0.0f;
        return Vec3(r, g, b);
    };

    if (nearestFilter) {
        int x = static_cast<int>(u * static_cast<float>(tex.width));
        int y = static_cast<int>(v * static_cast<float>(tex.height));
        x = max(0, min(x, tex.width - 1));
        y = max(0, min(y, tex.height - 1));
        return fetch(x, y);
    }

    float x = u * static_cast<float>(tex.width - 1);
    float y = v * static_cast<float>(tex.height - 1);
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, tex.width - 1);
    int y1 = min(y0 + 1, tex.height - 1);
    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);

    Vec3 c00 = fetch(x0, y0);
    Vec3 c10 = fetch(x1, y0);
    Vec3 c01 = fetch(x0, y1);
    Vec3 c11 = fetch(x1, y1);
    Vec3 c0 = lerp(c00, c10, tx);
    Vec3 c1 = lerp(c01, c11, tx);
    return lerp(c0, c1, ty);
}

__device__ inline Vec3 srgbToLinearDev(Vec3 c) {
    auto convert = [](float v) {
        if (v <= 0.04045f) {
            return v / 12.92f;
        }
        return powf((v + 0.055f) / 1.055f, 2.4f);
    };
    return Vec3(convert(c.x), convert(c.y), convert(c.z));
}

// =============================================================================
// Compute Full Hit Data from Preliminary Hit
// =============================================================================

__device__ inline HitInfo computeHitData(
    const PreliminaryHitInfo& pi,
    uint32_t primIdx,
    const Ray& ray,
    const MeshDeviceView& mesh)
{
    HitInfo its;
    its.t = pi.t;

    uint3 idx = mesh.indices[primIdx];
    Vec3 v0 = mesh.vertices[idx.x];
    Vec3 v1 = mesh.vertices[idx.y];
    Vec3 v2 = mesh.vertices[idx.z];

    float w = 1.0f - pi.u - pi.v;

    // Geometric normal (flat)
    its.geometricNormal = normalize(cross(v1 - v0, v2 - v0));

    // Shading normal (smooth, interpolated from per-vertex normals)
    if (mesh.hasNormals && mesh.normals) {
        its.shadingNormal = normalize(
            mesh.normals[idx.x] * w +
            mesh.normals[idx.y] * pi.u +
            mesh.normals[idx.z] * pi.v);
    } else {
        its.shadingNormal = its.geometricNormal;
    }

    // Position
    its.position = v0 * w + v1 * pi.u + v2 * pi.v;

    // UV coordinates
    if (mesh.hasTexcoords && mesh.texcoords) {
        its.uv = mesh.texcoords[idx.x] * w +
                 mesh.texcoords[idx.y] * pi.u +
                 mesh.texcoords[idx.z] * pi.v;
    } else {
        its.uv = Vec2(pi.u, pi.v);
    }

    // Material lookup
    its.materialId = getMaterialIdForTriangle(primIdx, mesh);

    // Apply normal map if material has one
    if (its.materialId >= 0 && its.materialId < mesh.numMaterials && mesh.materials) {
        const Material& mat = mesh.materials[its.materialId];
        if (mat.normal.textured && mat.normal.textureId < static_cast<uint32_t>(mesh.numTextures)) {
            Vec3 texNormal = sampleTextureRawDev(
                mesh.textures[mat.normal.textureId],
                its.uv.x, its.uv.y, mesh.textureNearest != 0);

            if (texNormal.x >= 0.0f) {  // Valid sample
                // Convert [0,1] -> [-1,1] tangent space
                Vec3 tsNormal = normalize(texNormal * 2.0f - Vec3(1.0f, 1.0f, 1.0f));

                // Build TBN from shading normal
                Vec3 T, B;
                orthoBasis(T, B, its.shadingNormal);

                // Transform to world space
                its.shadingNormal = normalize(
                    T * tsNormal.x +
                    B * tsNormal.y +
                    its.shadingNormal * tsNormal.z);
            }
        }
    }

    return its;
}

// =============================================================================
// Sample Material Parameter
// =============================================================================

__device__ __forceinline__ float sampleMaterialParam(
    const MaterialParam& param,
    Vec2 uv,
    const MeshDeviceView& mesh)
{
    if (!param.textured) {
        return param.value;
    }
    if (param.textureId >= static_cast<uint32_t>(mesh.numTextures) || !mesh.textures) {
        return param.value;
    }

    Vec3 sampled = sampleTextureRawDev(mesh.textures[param.textureId], uv.x, uv.y, mesh.textureNearest != 0);
    if (sampled.x < 0.0f) {
        return param.value;
    }

    // Select channel
    if (param.channel == 0) return sampled.x;
    if (param.channel == 1) return sampled.y;
    if (param.channel == 2) return sampled.z;
    return sampled.x;
}

__device__ __forceinline__ Vec3 sampleMaterialParamVec3(
    const MaterialParamVec3& param,
    Vec2 uv,
    const MeshDeviceView& mesh,
    bool applySrgb = true)
{
    if (!param.textured) {
        return param.value;
    }
    if (param.textureId >= static_cast<uint32_t>(mesh.numTextures) || !mesh.textures) {
        return param.value;
    }

    const TextureDeviceView& tex = mesh.textures[param.textureId];
    Vec3 raw = sampleTextureRawDev(tex, uv.x, uv.y, mesh.textureNearest != 0);
    if (raw.x < 0.0f) {
        return param.value;
    }

    // Check if texture is sRGB
    if (applySrgb && tex.srgb) {
        return srgbToLinearDev(raw);
    }
    return raw;
}

// =============================================================================
// Get Resolved Material at Hit Point
// =============================================================================

// Simple struct to hold resolved (sampled) material values
struct ResolvedMaterial {
    Vec3 base_color;
    float metallic;
    float roughness;
    float specular;
    float specular_tint;
    float anisotropy;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    Vec3 emission;
    float emission_scale;
    float ior;
    float specular_transmission;
};

__device__ __forceinline__ ResolvedMaterial resolveMaterial(
    const Material& mat,
    Vec2 uv,
    const MeshDeviceView& mesh)
{
    ResolvedMaterial r;
    r.base_color = sampleMaterialParamVec3(mat.base_color, uv, mesh, false);
    r.metallic = sampleMaterialParam(mat.metallic, uv, mesh);
    r.roughness = sampleMaterialParam(mat.roughness, uv, mesh);
    r.specular = sampleMaterialParam(mat.specular, uv, mesh);
    r.specular_tint = sampleMaterialParam(mat.specular_tint, uv, mesh);
    r.anisotropy = sampleMaterialParam(mat.anisotropy, uv, mesh);
    r.sheen = sampleMaterialParam(mat.sheen, uv, mesh);
    r.sheen_tint = sampleMaterialParam(mat.sheen_tint, uv, mesh);
    r.clearcoat = sampleMaterialParam(mat.clearcoat, uv, mesh);
    r.clearcoat_gloss = sampleMaterialParam(mat.clearcoat_gloss, uv, mesh);
    r.emission = sampleMaterialParamVec3(mat.base_emission, uv, mesh, false);
    r.emission_scale = mat.emission_scale;
    r.ior = mat.ior;
    r.specular_transmission = mat.specular_transmission;
    return r;
}
