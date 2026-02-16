#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "FLIP.h"

#include "stb_image.h"
#include "stb_image_write.h"

// Pixels per degree for the FLIP metric.
inline float calculatePPD(float monitorDistance, float resolutionX, float monitorWidth) {
    return monitorDistance * (resolutionX / monitorWidth) * (static_cast<float>(FLIP::PI) / 180.0f);
}

// Compute FLIP error between two images. Writes a Magma-colormap visualization to outputPath
// (may be nullptr to skip saving). Returns mean FLIP error.
inline float computeFlip(const std::vector<uchar4>& ref, const std::vector<uchar4>& test,
                          int width, int height, const char* outputPath) {
    struct {
        float PPD                = 0.0f;
        float monitorDistance    = 0.7f;
        float monitorWidth       = 0.7f;
        float monitorResolutionX = 3840.0f;
    } flipOptions;

    FLIP::image<FLIP::color3> reference(width, height);
    FLIP::image<FLIP::color3> testImage(width, height);
    FLIP::image<float> errorMapFLIP(width, height, 0.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            reference.set(x, y, FLIP::color3(ref[idx].x / 255.0f, ref[idx].y / 255.0f, ref[idx].z / 255.0f));
            testImage.set(x, y, FLIP::color3(test[idx].x / 255.0f, test[idx].y / 255.0f, test[idx].z / 255.0f));
        }
    }

    // Images are already in sRGB space — do NOT apply LinearRGB2sRGB().
    flipOptions.PPD = calculatePPD(flipOptions.monitorDistance, flipOptions.monitorResolutionX, flipOptions.monitorWidth);
    errorMapFLIP.FLIP(reference, testImage, flipOptions.PPD);

    pooling<float> pooledValues;
    float maxError = 0.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float err = errorMapFLIP.get(x, y);
            pooledValues.update(x, y, err);
            if (err > maxError) maxError = err;
        }
    }

    if (outputPath) {
        FLIP::image<FLIP::color3> magmaMap(FLIP::MapMagma, 256);
        FLIP::image<FLIP::color3> ldr_flip(width, height);
        ldr_flip.copyFloat2Color3(errorMapFLIP);
        ldr_flip.colorMap(errorMapFLIP, magmaMap);

        std::vector<unsigned char> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                FLIP::color3 c = ldr_flip.get(x, y);
                rgb[idx * 3 + 0] = static_cast<unsigned char>(c.x * 255.0f);
                rgb[idx * 3 + 1] = static_cast<unsigned char>(c.y * 255.0f);
                rgb[idx * 3 + 2] = static_cast<unsigned char>(c.z * 255.0f);
            }
        }

        if (stbi_write_png(outputPath, width, height, 3, rgb.data(), width * 3) == 0) {
            std::fprintf(stderr, "Failed to write FLIP visualization: %s\n", outputPath);
        } else {
            std::printf("Saved FLIP visualization: %s\n", outputPath);
        }
    }

    std::printf("FLIP max error: %.4f\n", maxError);
    return pooledValues.getMean();
}

// Compute PSNR (dB) between two images. Returns 100 if images are identical.
inline float computePsnr(const std::vector<uchar4>& ref, const std::vector<uchar4>& test,
                          int width, int height) {
    double mse = 0.0;
    size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);

    for (size_t i = 0; i < count; ++i) {
        double dr = static_cast<double>(ref[i].x) - static_cast<double>(test[i].x);
        double dg = static_cast<double>(ref[i].y) - static_cast<double>(test[i].y);
        double db = static_cast<double>(ref[i].z) - static_cast<double>(test[i].z);
        mse += (dr * dr + dg * dg + db * db) / 3.0;
    }
    mse /= static_cast<double>(count);

    if (mse < 1e-10) return 100.0f;
    return static_cast<float>(10.0 * std::log10((255.0 * 255.0) / mse));
}

// Save uchar4 pixel buffer as a 3-channel PNG.
inline bool savePng(const char* path, const std::vector<uchar4>& pixels, int width, int height) {
    std::vector<unsigned char> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
        rgb[i * 3 + 0] = pixels[i].x;
        rgb[i * 3 + 1] = pixels[i].y;
        rgb[i * 3 + 2] = pixels[i].z;
    }
    if (stbi_write_png(path, width, height, 3, rgb.data(), width * 3) == 0) {
        std::fprintf(stderr, "Failed to write PNG: %s\n", path);
        return false;
    }
    std::printf("Saved: %s\n", path);
    return true;
}

// Load a PNG/JPG image into a uchar4 buffer.
inline bool loadImage(const char* path, std::vector<uchar4>& pixels, int& width, int& height) {
    stbi_set_flip_vertically_on_load(false);
    int channels = 0;
    unsigned char* data = stbi_load(path, &width, &height, &channels, 3);
    if (!data) {
        std::fprintf(stderr, "Failed to load image: %s (%s)\n", path, stbi_failure_reason());
        return false;
    }
    size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    pixels.resize(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) {
        pixels[i] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2], 255};
    }
    stbi_image_free(data);
    return true;
}

// Flip pixel buffer vertically in-place.
inline void flipVertically(std::vector<uchar4>& pixels, int width, int height) {
    for (int y = 0; y < height / 2; ++y) {
        int oppositeY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            std::swap(pixels[y * width + x], pixels[oppositeY * width + x]);
        }
    }
}
