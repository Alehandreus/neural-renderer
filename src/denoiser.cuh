#pragma once

#include <cuda_runtime.h>
#include "vec3.h"

// ---------------------------------------------------------------------------
// sRGB encoding helpers
// Defined here so both the denoiser kernel and the anonymous-namespace kernels
// in cuda_renderer_neural.cu can share the same implementation.
// ---------------------------------------------------------------------------

__device__ inline float linearToSrgb(float v) {
    v = fmaxf(0.0f, v);
    float result;
    if (v <= 0.0031308f) {
        result = 12.92f * v;
    } else {
        result = 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
    }
    return fminf(1.0f, result);
}

__device__ inline Vec3 encodeSrgb(Vec3 c) {
    return Vec3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z));
}

// ---------------------------------------------------------------------------
// Joint bilateral denoiser parameters
// ---------------------------------------------------------------------------

constexpr int   kDenoiseRadius = 5;     // kernel half-radius → 11×11 window
constexpr float kSigmaSpatial  = 3.0f;  // spatial Gaussian spread
constexpr float kNormalAlpha   = 64.0f; // normal similarity sharpness
constexpr float kSigmaAlbedo   = 0.1f;  // albedo similarity spread

// ---------------------------------------------------------------------------
// bilateralDenoiseKernel
//
// Reads the running linear-HDR accumulation buffer (accum / accumCount),
// applies a joint bilateral filter guided by primary-hit normals and albedo,
// and writes the denoised sRGB result to output.
//
// guideNormals / guideAlbedo: 3 floats per pixel (sample-0 primary hits,
//   stored as buf[pixelIdx*3 + component]).
// ---------------------------------------------------------------------------
__global__ void bilateralDenoiseKernel(
        const Vec3* accum, int accumCount,
        const float* guideNormals, const float* guideAlbedo,
        uchar4* output, int width, int height) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= width || cy >= height) return;

    int centerIdx = cy * width + cx;
    float invAccum = 1.0f / fmaxf(1.0f, (float)accumCount);

    Vec3 centerColor = accum[centerIdx] * invAccum;

    float cnx = guideNormals[centerIdx * 3 + 0];
    float cny = guideNormals[centerIdx * 3 + 1];
    float cnz = guideNormals[centerIdx * 3 + 2];
    float cax = guideAlbedo[centerIdx * 3 + 0];
    float cay = guideAlbedo[centerIdx * 3 + 1];
    float caz = guideAlbedo[centerIdx * 3 + 2];

    const float invSigmaSpatial2 = 1.0f / (2.0f * kSigmaSpatial * kSigmaSpatial);
    const float invSigmaAlbedo2  = 1.0f / (2.0f * kSigmaAlbedo  * kSigmaAlbedo);

    Vec3  weightedSum(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    for (int dy = -kDenoiseRadius; dy <= kDenoiseRadius; ++dy) {
        int ny = cy + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -kDenoiseRadius; dx <= kDenoiseRadius; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= width) continue;

            int nIdx = ny * width + nx;
            Vec3 neighborColor = accum[nIdx] * invAccum;

            // Spatial weight
            float dist2   = (float)(dx * dx + dy * dy);
            float wSpatial = expf(-dist2 * invSigmaSpatial2);

            // Normal similarity weight (dot product clamped to [0,1])
            float nnx = guideNormals[nIdx * 3 + 0];
            float nny = guideNormals[nIdx * 3 + 1];
            float nnz = guideNormals[nIdx * 3 + 2];
            float normalDot = fmaxf(0.0f, fminf(1.0f, cnx*nnx + cny*nny + cnz*nnz));
            float wNormal   = powf(normalDot, kNormalAlpha);

            // Albedo similarity weight
            float dax = guideAlbedo[nIdx * 3 + 0] - cax;
            float day = guideAlbedo[nIdx * 3 + 1] - cay;
            float daz = guideAlbedo[nIdx * 3 + 2] - caz;
            float wAlbedo = expf(-(dax*dax + day*day + daz*daz) * invSigmaAlbedo2);

            float w = wSpatial * wNormal * wAlbedo;
            weightedSum += neighborColor * w;
            totalWeight += w;
        }
    }

    Vec3 color = (totalWeight > 0.0f) ? (weightedSum * (1.0f / totalWeight)) : centerColor;
    color = encodeSrgb(color);

    output[centerIdx] = make_uchar4(
            (unsigned char)(color.x * 255.0f),
            (unsigned char)(color.y * 255.0f),
            (unsigned char)(color.z * 255.0f),
            255);
}
