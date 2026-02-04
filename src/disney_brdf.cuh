#pragma once

#include "material.h"
#include "vec3.h"

// Disney Principled BRDF implementation
// Based on nbvh/include/cuda/base/material.cuh

#define INV_PI 0.31830988618379067154f

__device__ inline float sqr(float x) {
    return x * x;
}

__device__ inline float lerpf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

__device__ inline float saturate(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ inline float luminance(Vec3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

__device__ inline float schlick_weight(float cos_theta) {
    float m = saturate(1.0f - cos_theta);
    float m2 = m * m;
    return m2 * m2 * m;  // m^5
}

__device__ inline float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
    float g = sqr(eta_t) / sqr(eta_i) - 1.0f + sqr(cos_theta_i);
    if (g < 0.0f) {
        return 1.0f;
    }
    g = sqrtf(g);
    float a = (g - cos_theta_i) / (g + cos_theta_i);
    float b = (cos_theta_i * (g + cos_theta_i) - 1.0f) / (cos_theta_i * (g - cos_theta_i) + 1.0f);
    return 0.5f * a * a * (1.0f + b * b);
}

// GTR1 (Generalized Trowbridge-Reitz with gamma=1) - used for clearcoat
__device__ inline float gtr_1(float cos_theta_h, float alpha) {
    if (alpha >= 1.0f) {
        return INV_PI;
    }
    float alpha_sqr = alpha * alpha;
    return INV_PI * (alpha_sqr - 1.0f) / (logf(alpha_sqr) * (1.0f + (alpha_sqr - 1.0f) * cos_theta_h * cos_theta_h));
}

// GTR2 (Generalized Trowbridge-Reitz with gamma=2, equivalent to GGX)
__device__ inline float gtr_2(float cos_theta_h, float alpha) {
    float alpha_sqr = alpha * alpha;
    return INV_PI * alpha_sqr / sqr(1.0f + (alpha_sqr - 1.0f) * cos_theta_h * cos_theta_h);
}

// GTR2 anisotropic
__device__ inline float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, Vec2 alpha) {
    return INV_PI / (alpha.x * alpha.y * sqr(sqr(h_dot_x / alpha.x) + sqr(h_dot_y / alpha.y) + h_dot_n * h_dot_n));
}

// Smith shadowing/masking term for GGX
__device__ inline float smith_shadowing_ggx(float n_dot_o, float alpha_g) {
    float a = alpha_g * alpha_g;
    float b = n_dot_o * n_dot_o;
    return 1.0f / (n_dot_o + sqrtf(a + b - a * b));
}

// Smith shadowing/masking term for GGX anisotropic
__device__ inline float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, Vec2 alpha) {
    return 1.0f / (n_dot_o + sqrtf(sqr(o_dot_x * alpha.x) + sqr(o_dot_y * alpha.y) + sqr(n_dot_o)));
}

__device__ inline bool same_hemisphere(Vec3 wo, Vec3 wi, Vec3 n) {
    return dot(wo, n) * dot(wi, n) > 0.0f;
}

// ---------------------------------------------------------------------------
// Disney BRDF Components
// ---------------------------------------------------------------------------

__device__ inline Vec3 disney_diffuse(const Material& mat,
                                      Vec3 n,
                                      Vec3 wo,
                                      Vec3 wi) {
    Vec3 w_h = normalize(wi + wo);
    float n_dot_o = fabsf(dot(wo, n));
    float n_dot_i = fabsf(dot(wi, n));
    float i_dot_h = dot(wi, w_h);
    float fd90 = 0.5f + 2.0f * mat.roughness * i_dot_h * i_dot_h;
    float fi = schlick_weight(n_dot_i);
    float fo = schlick_weight(n_dot_o);
    return mat.base_color * INV_PI * lerpf(1.0f, fd90, fi) * lerpf(1.0f, fd90, fo);
}

__device__ inline Vec3 disney_sheen(const Material& mat,
                                    Vec3 n,
                                    Vec3 wo,
                                    Vec3 wi) {
    Vec3 w_h = normalize(wi + wo);
    float lum = luminance(mat.base_color);
    Vec3 tint = lum > 0.0f ? mat.base_color / lum : Vec3(1.0f, 1.0f, 1.0f);
    Vec3 sheen_color = lerp(Vec3(1.0f, 1.0f, 1.0f), tint, mat.sheen_tint);
    float f = schlick_weight(dot(wi, w_h));
    return sheen_color * (f * mat.sheen);
}

__device__ inline float disney_clear_coat(const Material& mat,
                                          Vec3 n,
                                          Vec3 wo,
                                          Vec3 wi) {
    Vec3 w_h = normalize(wi + wo);
    float alpha = lerpf(0.1f, 0.001f, mat.clearcoat_gloss);
    float d = gtr_1(dot(n, w_h), alpha);
    float f = lerpf(0.04f, 1.0f, schlick_weight(dot(wi, n)));
    float g = smith_shadowing_ggx(dot(n, wi), 0.25f) * smith_shadowing_ggx(dot(n, wo), 0.25f);
    return 0.25f * mat.clearcoat * d * f * g;
}

__device__ inline Vec3 disney_microfacet_isotropic(const Material& mat,
                                                   Vec3 n,
                                                   Vec3 wo,
                                                   Vec3 wi) {
    Vec3 w_h = normalize(wi + wo);
    float lum = luminance(mat.base_color);
    Vec3 tint = lum > 0.0f ? mat.base_color / lum : Vec3(1.0f, 1.0f, 1.0f);
    Vec3 spec = lerp(mat.specular * 0.08f * lerp(Vec3(1.0f, 1.0f, 1.0f), tint, mat.specular_tint), mat.base_color, mat.metallic);

    float alpha = fmaxf(0.001f, mat.roughness * mat.roughness);
    float d = gtr_2(dot(n, w_h), alpha);
    Vec3 f = lerp(spec, Vec3(1.0f, 1.0f, 1.0f), schlick_weight(dot(wi, w_h)));
    float g = smith_shadowing_ggx(dot(n, wi), alpha) * smith_shadowing_ggx(dot(n, wo), alpha);
    return f * (d * g);
}

__device__ inline Vec3 disney_microfacet_anisotropic(const Material& mat,
                                                     Vec3 n,
                                                     Vec3 wo,
                                                     Vec3 wi,
                                                     Vec3 tangent,
                                                     Vec3 bitangent) {
    Vec3 w_h = normalize(wi + wo);
    float lum = luminance(mat.base_color);
    Vec3 tint = lum > 0.0f ? mat.base_color / lum : Vec3(1.0f, 1.0f, 1.0f);
    Vec3 spec = lerp(mat.specular * 0.08f * lerp(Vec3(1.0f, 1.0f, 1.0f), tint, mat.specular_tint), mat.base_color, mat.metallic);

    float aspect = sqrtf(1.0f - mat.anisotropy * 0.9f);
    float a = mat.roughness * mat.roughness;
    Vec2 alpha = Vec2(fmaxf(0.001f, a / aspect), fmaxf(0.001f, a * aspect));
    float d = gtr_2_aniso(dot(n, w_h), fabsf(dot(w_h, tangent)), fabsf(dot(w_h, bitangent)), alpha);
    Vec3 f = lerp(spec, Vec3(1.0f, 1.0f, 1.0f), schlick_weight(dot(wi, w_h)));
    float g = smith_shadowing_ggx_aniso(dot(n, wi), fabsf(dot(wi, tangent)), fabsf(dot(wi, bitangent)), alpha) *
              smith_shadowing_ggx_aniso(dot(n, wo), fabsf(dot(wo, tangent)), fabsf(dot(wo, bitangent)), alpha);
    return f * (d * g);
}

__device__ inline Vec3 disney_microfacet_transmission_isotropic(const Material& mat,
                                                                Vec3 n,
                                                                Vec3 wo,
                                                                Vec3 wi) {
    float o_dot_n = dot(wo, n);
    float i_dot_n = dot(wi, n);
    if (o_dot_n == 0.0f || i_dot_n == 0.0f) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    bool entering = o_dot_n > 0.0f;
    float eta_o = entering ? 1.0f : mat.ior;
    float eta_i = entering ? mat.ior : 1.0f;
    Vec3 w_h = normalize(wo + wi * eta_i / eta_o);

    float alpha = fmaxf(0.001f, mat.roughness * mat.roughness);
    float d = gtr_2(fabsf(dot(n, w_h)), alpha);

    float f = fresnel_dielectric(fabsf(dot(wi, n)), eta_o, eta_i);
    float g = smith_shadowing_ggx(fabsf(dot(n, wi)), alpha) * smith_shadowing_ggx(fabsf(dot(n, wo)), alpha);

    float i_dot_h = dot(wi, w_h);
    float o_dot_h = dot(wo, w_h);

    float c = fabsf(o_dot_h) / fabsf(dot(wo, n)) * fabsf(i_dot_h) / fabsf(dot(wi, n)) * sqr(eta_o) /
              sqr(eta_o * o_dot_h + eta_i * i_dot_h);

    return mat.base_color * (c * (1.0f - f) * g * d);
}

// Main Disney BRDF evaluation
__device__ inline Vec3 disney_eval(const Material& mat,
                                   Vec3 n,
                                   Vec3 wo,
                                   Vec3 wi,
                                   Vec3 tangent,
                                   Vec3 bitangent) {
    if (!same_hemisphere(wo, wi, n)) {
        if (mat.specular_transmission > 0.0f) {
            Vec3 spec_trans = disney_microfacet_transmission_isotropic(mat, n, wo, wi);
            return spec_trans * ((1.0f - mat.metallic) * mat.specular_transmission);
        }
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    float coat = disney_clear_coat(mat, n, wo, wi);
    Vec3 sheen = disney_sheen(mat, n, wo, wi);
    Vec3 diffuse = disney_diffuse(mat, n, wo, wi);
    Vec3 gloss;
    if (mat.anisotropy == 0.0f) {
        gloss = disney_microfacet_isotropic(mat, n, wo, wi);
    } else {
        gloss = disney_microfacet_anisotropic(mat, n, wo, wi, tangent, bitangent);
    }
    return (diffuse + sheen) * ((1.0f - mat.metallic) * (1.0f - mat.specular_transmission)) + gloss + Vec3(coat, coat, coat);
}

// ---------------------------------------------------------------------------
// Disney BRDF Sampling
// ---------------------------------------------------------------------------

// Sample GGX distribution (GTR2)
__device__ inline Vec3 sample_ggx(Vec3 n, float alpha, float u1, float u2) {
    float phi = 2.0f * 3.14159265358979323846f * u1;
    float cos_theta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    Vec3 h;
    h.x = sin_theta * cosf(phi);
    h.y = sin_theta * sinf(phi);
    h.z = cos_theta;

    // Transform to world space
    Vec3 up = fabsf(n.z) < 0.999f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangent = normalize(cross(up, n));
    Vec3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

// Sample cosine-weighted hemisphere
__device__ inline Vec3 sample_cosine_hemisphere(Vec3 n, float u1, float u2) {
    float phi = 2.0f * 3.14159265358979323846f * u1;
    float cos_theta = sqrtf(u2);
    float sin_theta = sqrtf(1.0f - u2);

    Vec3 h;
    h.x = sin_theta * cosf(phi);
    h.y = sin_theta * sinf(phi);
    h.z = cos_theta;

    // Transform to world space
    Vec3 up = fabsf(n.z) < 0.999f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangent = normalize(cross(up, n));
    Vec3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

// Disney BRDF importance sampling
__device__ inline Vec3 disney_sample(const Material& mat,
                                     Vec3 n,
                                     Vec3 wo,
                                     float u1,
                                     float u2,
                                     float u3,
                                     float* pdf_out) {
    // Simple lobe selection based on material properties
    float diffuse_weight = (1.0f - mat.metallic) * (1.0f - mat.specular_transmission);

    // float specular_weight = 1.0f;
    float F0 = 0.08f * mat.specular;
    float specular_weight = F0 + (1.0f - F0) * mat.metallic;

    float total_weight = diffuse_weight + specular_weight;

    float diffuse_prob = diffuse_weight / total_weight;

    Vec3 wi;
    if (u3 < diffuse_prob) {
        // Sample diffuse lobe (cosine-weighted hemisphere)
        wi = sample_cosine_hemisphere(n, u1, u2);
        if (pdf_out) {
            float n_dot_i = fmaxf(0.0f, dot(n, wi));
            *pdf_out = n_dot_i * INV_PI * diffuse_prob;
        }
    } else {
        // Sample specular lobe (GGX)
        float alpha = fmaxf(0.001f, mat.roughness * mat.roughness);
        Vec3 h = sample_ggx(n, alpha, u1, u2);
        wi = normalize(wo * -1.0f + h * (2.0f * dot(wo, h)));

        if (pdf_out) {
            float n_dot_h = fmaxf(0.0f, dot(n, h));
            float h_dot_o = fmaxf(0.0f, dot(h, wo));
            float D = gtr_2(n_dot_h, alpha);
            *pdf_out = (D * n_dot_h / (4.0f * h_dot_o)) * (1.0f - diffuse_prob);
        }
    }

    return wi;
}

// PDF for Disney BRDF sampling
__device__ inline float disney_pdf(const Material& mat,
                                   Vec3 n,
                                   Vec3 wo,
                                   Vec3 wi) {
    if (!same_hemisphere(wo, wi, n)) {
        return 0.0f;
    }

    float diffuse_weight = (1.0f - mat.metallic) * (1.0f - mat.specular_transmission);
    float specular_weight = 1.0f;
    float total_weight = diffuse_weight + specular_weight;

    float diffuse_prob = diffuse_weight / total_weight;
    float specular_prob = 1.0f - diffuse_prob;

    // Diffuse PDF
    float n_dot_i = fmaxf(0.0f, dot(n, wi));
    float pdf_diffuse = n_dot_i * INV_PI;

    // Specular PDF
    Vec3 h = normalize(wi + wo);
    float n_dot_h = fmaxf(0.0f, dot(n, h));
    float h_dot_o = fmaxf(0.0f, dot(h, wo));
    float alpha = fmaxf(0.001f, mat.roughness * mat.roughness);
    float D = gtr_2(n_dot_h, alpha);
    float pdf_specular = D * n_dot_h / (4.0f * h_dot_o);

    return pdf_diffuse * diffuse_prob + pdf_specular * specular_prob;
}
