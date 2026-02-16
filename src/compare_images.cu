#include <cstdio>
#include <vector>

#include "image_utils.h"

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

    std::vector<uchar4> refPixels;
    int refWidth, refHeight;
    if (!loadImage(refPath, refPixels, refWidth, refHeight)) return 1;
    std::printf("Loaded reference: %dx%d\n", refWidth, refHeight);

    std::vector<uchar4> testPixels;
    int testWidth, testHeight;
    if (!loadImage(testPath, testPixels, testWidth, testHeight)) return 1;
    std::printf("Loaded test:      %dx%d\n", testWidth, testHeight);

    if (refWidth != testWidth || refHeight != testHeight) {
        std::fprintf(stderr, "Error: Image dimensions do not match!\n");
        std::fprintf(stderr, "  Reference: %dx%d\n", refWidth, refHeight);
        std::fprintf(stderr, "  Test:      %dx%d\n", testWidth, testHeight);
        return 1;
    }

    // Some renderers/write paths disagree on image origin. Detect and fix this
    // before computing FLIP so compare_images matches evaluate semantics.
    float psnrNoFlip = computePsnr(refPixels, testPixels, refWidth, refHeight);
    std::vector<uchar4> testPixelsFlipped = testPixels;
    flipVertically(testPixelsFlipped, refWidth, refHeight);
    float psnrFlipY = computePsnr(refPixels, testPixelsFlipped, refWidth, refHeight);
    if (psnrFlipY > psnrNoFlip + 3.0f) {
        testPixels.swap(testPixelsFlipped);
        std::printf("Detected Y-flipped test image (PSNR %.2f -> %.2f dB). Using flipped orientation.\n",
                    psnrNoFlip, psnrFlipY);
    }

    std::printf("\n=== Computing Metrics ===\n");
    float psnr = computePsnr(refPixels, testPixels, refWidth, refHeight);
    std::printf("PSNR: %.2f dB\n", psnr);

    float flipError = computeFlip(refPixels, testPixels, refWidth, refHeight, flipOutputPath);
    std::printf("FLIP: %.4f (mean)\n", flipError);

    std::printf("\nComparison complete.\n");
    return 0;
}
