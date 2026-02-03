#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "FLIP.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Helper to calculate pixels per degree.
float calculatePPD(float monitorDistance, float resolutionX, float monitorWidth) {
    return monitorDistance * (resolutionX / monitorWidth) * (static_cast<float>(FLIP::PI) / 180.0f);
}

// Compute FLIP error between two images and save visualization.
float computeFlip(const std::vector<uchar4>& ref, const std::vector<uchar4>& test, int width, int height, const char* outputPath) {
    // FLIP parameters.
    struct {
        float PPD                = 0.0f;     // If PPD==0.0, computed from parameters below.
        float monitorDistance    = 0.7f;     // Unit: meters.
        float monitorWidth       = 0.7f;     // Unit: meters.
        float monitorResolutionX = 3840.0f;  // Unit: pixels.
    } flipOptions;

    // Create FLIP images.
    FLIP::image<FLIP::color3> reference(width, height);
    FLIP::image<FLIP::color3> testImage(width, height);
    FLIP::image<float> errorMapFLIP(width, height, 0.0f);

    // Convert uchar4 to FLIP::color3 (normalize to [0,1]).
    // PNG/JPG images are already in sRGB color space, so no conversion needed.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            reference.set(x, y, FLIP::color3(
                ref[idx].x / 255.0f,
                ref[idx].y / 255.0f,
                ref[idx].z / 255.0f));
            testImage.set(x, y, FLIP::color3(
                test[idx].x / 255.0f,
                test[idx].y / 255.0f,
                test[idx].z / 255.0f));
        }
    }

    // Images loaded from disk (PNG/JPG) are already in sRGB space.
    // DO NOT call LinearRGB2sRGB() - that would apply sRGB curve twice!

    // Calculate PPD.
    flipOptions.PPD = calculatePPD(flipOptions.monitorDistance, flipOptions.monitorResolutionX, flipOptions.monitorWidth);

    // Compute FLIP.
    errorMapFLIP.FLIP(reference, testImage, flipOptions.PPD);

    // Compute mean error and find max.
    pooling<float> pooledValues;
    float maxError = 0.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float err = errorMapFLIP.get(x, y);
            pooledValues.update(x, y, err);
            if (err > maxError) maxError = err;
        }
    }

    // Save FLIP visualization using Magma colormap.
    if (outputPath) {
        FLIP::image<FLIP::color3> magmaMap(FLIP::MapMagma, 256);
        FLIP::image<FLIP::color3> ldr_flip(width, height);

        ldr_flip.copyFloat2Color3(errorMapFLIP);
        ldr_flip.colorMap(errorMapFLIP, magmaMap);

        // Convert FLIP::color3 to uchar4 for PNG saving.
        std::vector<uchar4> flipVis(static_cast<size_t>(width) * static_cast<size_t>(height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                FLIP::color3 c = ldr_flip.get(x, y);
                flipVis[idx] = {
                    static_cast<unsigned char>(c.x * 255.0f),
                    static_cast<unsigned char>(c.y * 255.0f),
                    static_cast<unsigned char>(c.z * 255.0f),
                    255
                };
            }
        }

        // Save as PNG.
        std::vector<unsigned char> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (size_t i = 0; i < flipVis.size(); ++i) {
            rgb[i * 3 + 0] = flipVis[i].x;
            rgb[i * 3 + 1] = flipVis[i].y;
            rgb[i * 3 + 2] = flipVis[i].z;
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

// Compute PSNR between two images.
float computePsnr(const std::vector<uchar4>& ref, const std::vector<uchar4>& test, int width, int height) {
    double mse = 0.0;
    size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);

    for (size_t i = 0; i < count; ++i) {
        double dr = static_cast<double>(ref[i].x) - static_cast<double>(test[i].x);
        double dg = static_cast<double>(ref[i].y) - static_cast<double>(test[i].y);
        double db = static_cast<double>(ref[i].z) - static_cast<double>(test[i].z);
        mse += (dr * dr + dg * dg + db * db) / 3.0;
    }

    mse /= static_cast<double>(count);

    if (mse < 1e-10) {
        return 100.0f;  // Images are identical.
    }

    double psnr = 10.0 * std::log10((255.0 * 255.0) / mse);
    return static_cast<float>(psnr);
}

// Load image using stb_image.
bool loadImage(const char* path, std::vector<uchar4>& pixels, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(path, &width, &height, &channels, 0);
    if (!data) {
        std::fprintf(stderr, "Failed to load image: %s\n", path);
        return false;
    }

    // Convert to uchar4 format.
    size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    pixels.resize(pixelCount);

    if (channels == 1) {
        // Grayscale - replicate to RGB.
        for (size_t i = 0; i < pixelCount; ++i) {
            unsigned char val = data[i];
            pixels[i] = {val, val, val, 255};
        }
    } else if (channels == 3) {
        // RGB - add alpha.
        for (size_t i = 0; i < pixelCount; ++i) {
            pixels[i] = {
                data[i * 3 + 0],
                data[i * 3 + 1],
                data[i * 3 + 2],
                255
            };
        }
    } else if (channels == 4) {
        // RGBA - use directly (ignore alpha for comparison).
        for (size_t i = 0; i < pixelCount; ++i) {
            pixels[i] = {
                data[i * 4 + 0],
                data[i * 4 + 1],
                data[i * 4 + 2],
                255
            };
        }
    } else {
        std::fprintf(stderr, "Unsupported channel count: %d\n", channels);
        stbi_image_free(data);
        return false;
    }

    stbi_image_free(data);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <reference_image> <test_image> [flip_output.png]\n", argv[0]);
        std::fprintf(stderr, "  Computes PSNR and FLIP metrics between two images.\n");
        std::fprintf(stderr, "  Optionally saves FLIP error visualization to third argument.\n");
        return 1;
    }

    const char* refPath = argv[1];
    const char* testPath = argv[2];
    const char* flipOutputPath = (argc >= 4) ? argv[3] : nullptr;

    std::printf("=== Image Comparison Tool ===\n");
    std::printf("Reference: %s\n", refPath);
    std::printf("Test:      %s\n", testPath);

    // Load reference image.
    std::vector<uchar4> refPixels;
    int refWidth, refHeight;
    if (!loadImage(refPath, refPixels, refWidth, refHeight)) {
        return 1;
    }
    std::printf("Loaded reference: %dx%d\n", refWidth, refHeight);

    // Load test image.
    std::vector<uchar4> testPixels;
    int testWidth, testHeight;
    if (!loadImage(testPath, testPixels, testWidth, testHeight)) {
        return 1;
    }
    std::printf("Loaded test:      %dx%d\n", testWidth, testHeight);

    // Verify dimensions match.
    if (refWidth != testWidth || refHeight != testHeight) {
        std::fprintf(stderr, "Error: Image dimensions do not match!\n");
        std::fprintf(stderr, "  Reference: %dx%d\n", refWidth, refHeight);
        std::fprintf(stderr, "  Test:      %dx%d\n", testWidth, testHeight);
        return 1;
    }

    // Compute PSNR.
    std::printf("\n=== Computing Metrics ===\n");
    float psnr = computePsnr(refPixels, testPixels, refWidth, refHeight);
    std::printf("PSNR: %.2f dB\n", psnr);

    // Compute FLIP.
    float flipError = computeFlip(refPixels, testPixels, refWidth, refHeight, flipOutputPath);
    std::printf("FLIP: %.4f (mean)\n", flipError);

    std::printf("\nComparison complete.\n");
    return 0;
}
