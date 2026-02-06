#pragma once

#include "vec3.h"
#include <cstdint>

enum class ColorSpace { LINEAR, SRGB };

// Scalar material parameter: either a constant value or texture reference
struct MaterialParam {
    float value = 0.0f;
    uint32_t textureId = 0;
    uint8_t channel = 0;      // 0=R, 1=G, 2=B, 3=A
    bool textured = false;

    __host__ __device__ static MaterialParam constant(float v) {
        MaterialParam p;
        p.value = v;
        p.textureId = 0;
        p.channel = 0;
        p.textured = false;
        return p;
    }

    __host__ __device__ static MaterialParam texture(uint32_t id, uint8_t ch = 0) {
        MaterialParam p;
        p.value = 0.0f;
        p.textureId = id;
        p.channel = ch;
        p.textured = true;
        return p;
    }
};

// Vec3 material parameter: either a constant color or texture reference (uses RGB)
struct MaterialParamVec3 {
    Vec3 value{0.0f, 0.0f, 0.0f};
    uint32_t textureId = 0;
    bool textured = false;

    __host__ __device__ static MaterialParamVec3 constant(Vec3 v) {
        MaterialParamVec3 p;
        p.value = v;
        p.textureId = 0;
        p.textured = false;
        return p;
    }

    __host__ __device__ static MaterialParamVec3 texture(uint32_t id) {
        MaterialParamVec3 p;
        p.value = Vec3(0.0f, 0.0f, 0.0f);
        p.textureId = id;
        p.textured = true;
        return p;
    }
};

// Disney Principled BRDF material with explicit texture flags
struct Material {
    // Base color (albedo)
    MaterialParamVec3 base_color;

    // PBR parameters
    MaterialParam metallic;
    MaterialParam roughness;
    MaterialParam specular;
    MaterialParam specular_tint;
    MaterialParam anisotropy;

    // Sheen
    MaterialParam sheen;
    MaterialParam sheen_tint;

    // Clearcoat
    MaterialParam clearcoat;
    MaterialParam clearcoat_gloss;

    // Normal map
    MaterialParamVec3 normal;

    // Emission
    MaterialParamVec3 base_emission;
    float emission_scale = 1.0f;

    // Transmission
    float ior = 1.5f;
    float specular_transmission = 0.0f;

    // Create default material with sensible values
    __host__ __device__ static Material defaultMaterial() {
        Material mat;
        mat.base_color = MaterialParamVec3::constant(Vec3(0.8f, 0.8f, 0.8f));
        mat.metallic = MaterialParam::constant(0.0f);
        mat.roughness = MaterialParam::constant(0.0f);
        mat.specular = MaterialParam::constant(0.0f);
        mat.specular_tint = MaterialParam::constant(0.0f);
        mat.anisotropy = MaterialParam::constant(0.0f);
        mat.sheen = MaterialParam::constant(0.0f);
        mat.sheen_tint = MaterialParam::constant(0.0f);
        mat.clearcoat = MaterialParam::constant(0.0f);
        mat.clearcoat_gloss = MaterialParam::constant(1.0f);
        mat.normal = MaterialParamVec3::constant(Vec3(0.0f, 0.0f, 0.0f));
        mat.normal.textured = false;
        mat.base_emission = MaterialParamVec3::constant(Vec3(0.0f, 0.0f, 0.0f));
        mat.emission_scale = 1.0f;
        mat.ior = 1.5f;
        mat.specular_transmission = 0.0f;
        return mat;
    }

    // Check if any parameter uses textures
    __host__ __device__ bool hasAnyTexture() const {
        return base_color.textured || metallic.textured || roughness.textured ||
               specular.textured || specular_tint.textured || anisotropy.textured ||
               sheen.textured || sheen_tint.textured || clearcoat.textured ||
               clearcoat_gloss.textured || normal.textured || base_emission.textured;
    }
};
