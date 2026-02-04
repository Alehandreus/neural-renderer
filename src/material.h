#pragma once

#include "vec3.h"

// Disney Principled BRDF material parameters
// Matching nbvh/include/base/material.h
struct Material {
    // Disney Material
    Vec3 base_color{1.0, 1.0, 1.0};
    // Vec3 base_color{0.906, 0.906, 0.906};
    // Vec3 base_color{0.906, 0.906, 0.906};
    // Vec3 base_color{0.9f, 0.9f, 0.9f};
    // float metallic = 0.1f;
    // float specular = 0.2f;
    // float roughness = 0.2f;

    float metallic = 0.0f;
    float specular = 0.0f;
    float roughness = 1.0f;    

    float specular_tint = 0.0f;
    float anisotropy = 0.0f;

    float sheen = 0.0f;
    float sheen_tint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoat_gloss = 0.0f;

    Vec3 base_emission{0.0f, 0.0f, 0.0f};
    float emission_scale = 1.0f;

    Vec3 normal{0.0f, 0.0f, 0.0f};
    float ior = 1.5f;

    float specular_transmission = 0.0f;
};
