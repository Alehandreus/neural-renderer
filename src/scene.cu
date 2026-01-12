#include "scene.h"

#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "stb_image.h"
#include "tinyexr.h"

namespace {

void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

bool readBytes(std::ifstream& file, unsigned char* data, size_t count) {
    file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count));
    return static_cast<size_t>(file.gcount()) == count;
}

Vec3 decodeRgbe(unsigned char r, unsigned char g, unsigned char b, unsigned char e) {
    if (e == 0) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    float scale = std::ldexp(1.0f, static_cast<int>(e) - (128 + 8));
    return Vec3(r * scale, g * scale, b * scale);
}

bool loadHdrImage(const std::string& path,
                  std::vector<Vec3>* pixels,
                  int* outWidth,
                  int* outHeight,
                  std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) {
            *error = "Failed to open HDR file.";
        }
        return false;
    }

    std::string line;
    bool formatOk = false;
    while (std::getline(file, line)) {
        if (line.empty()) {
            break;
        }
        if (line.rfind("FORMAT=", 0) == 0) {
            if (line.find("32-bit_rle_rgbe") != std::string::npos) {
                formatOk = true;
            }
        }
    }

    if (!formatOk) {
        if (error) {
            *error = "Unsupported HDR format (expected 32-bit_rle_rgbe).";
        }
        return false;
    }

    if (!std::getline(file, line)) {
        if (error) {
            *error = "Missing HDR resolution line.";
        }
        return false;
    }

    int width = 0;
    int height = 0;
    if (std::sscanf(line.c_str(), "-Y %d +X %d", &height, &width) != 2) {
        if (error) {
            *error = "Invalid HDR resolution line.";
        }
        return false;
    }

    if (width <= 0 || height <= 0) {
        if (error) {
            *error = "Invalid HDR dimensions.";
        }
        return false;
    }

    pixels->assign(static_cast<size_t>(width) * static_cast<size_t>(height), Vec3());
    std::vector<unsigned char> scanline(static_cast<size_t>(width) * 4);

    for (int y = 0; y < height; ++y) {
        unsigned char header[4] = {};
        if (!readBytes(file, header, 4)) {
            if (error) {
                *error = "Unexpected end of HDR file.";
            }
            return false;
        }

        bool isRle = header[0] == 2 && header[1] == 2 && (header[2] & 0x80) == 0;
        int scanlineWidth = (static_cast<int>(header[2]) << 8) | header[3];
        if (!isRle || scanlineWidth != width || width < 8 || width > 0x7fff) {
            (*pixels)[0] = decodeRgbe(header[0], header[1], header[2], header[3]);
            const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
            for (size_t i = 1; i < total; ++i) {
                unsigned char rgbe[4] = {};
                if (!readBytes(file, rgbe, 4)) {
                    if (error) {
                        *error = "Unexpected end of HDR file (flat data).";
                    }
                    return false;
                }
                (*pixels)[i] = decodeRgbe(rgbe[0], rgbe[1], rgbe[2], rgbe[3]);
            }
            *outWidth = width;
            *outHeight = height;
            return true;
        }

        for (int channel = 0; channel < 4; ++channel) {
            int x = 0;
            while (x < width) {
                unsigned char count = 0;
                if (!readBytes(file, &count, 1)) {
                    if (error) {
                        *error = "Unexpected end of HDR file (RLE count).";
                    }
                    return false;
                }
                if (count > 128) {
                    int run = count - 128;
                    unsigned char value = 0;
                    if (!readBytes(file, &value, 1)) {
                        if (error) {
                            *error = "Unexpected end of HDR file (RLE value).";
                        }
                        return false;
                    }
                    for (int i = 0; i < run && x < width; ++i) {
                        scanline[static_cast<size_t>(channel) * width + x] = value;
                        ++x;
                    }
                } else {
                    int run = count;
                    if (run == 0 || x + run > width) {
                        if (error) {
                            *error = "Invalid HDR RLE run.";
                        }
                        return false;
                    }
                    if (!readBytes(file, scanline.data() + static_cast<size_t>(channel) * width + x,
                                   static_cast<size_t>(run))) {
                        if (error) {
                            *error = "Unexpected end of HDR file (RLE data).";
                        }
                        return false;
                    }
                    x += run;
                }
            }
        }

        for (int x = 0; x < width; ++x) {
            unsigned char r = scanline[static_cast<size_t>(x)];
            unsigned char g = scanline[static_cast<size_t>(width + x)];
            unsigned char b = scanline[static_cast<size_t>(2 * width + x)];
            unsigned char e = scanline[static_cast<size_t>(3 * width + x)];
            (*pixels)[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                    decodeRgbe(r, g, b, e);
        }
    }

    *outWidth = width;
    *outHeight = height;
    return true;
}

char toLowerAscii(char value) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
}

bool endsWithIgnoreCase(const std::string& value, const char* suffix) {
    const size_t suffixLen = std::strlen(suffix);
    if (value.size() < suffixLen) {
        return false;
    }
    const size_t offset = value.size() - suffixLen;
    for (size_t i = 0; i < suffixLen; ++i) {
        if (toLowerAscii(value[offset + i]) != toLowerAscii(suffix[i])) {
            return false;
        }
    }
    return true;
}

bool loadExrImage(const std::string& path,
                  std::vector<Vec3>* pixels,
                  int* outWidth,
                  int* outHeight,
                  std::string* error) {
    float* rgba = nullptr;
    int width = 0;
    int height = 0;
    const char* err = nullptr;
    int result = LoadEXR(&rgba, &width, &height, path.c_str(), &err);
    if (result != TINYEXR_SUCCESS) {
        if (error) {
            *error = err ? err : "Failed to load EXR file.";
        }
        if (err) {
            FreeEXRErrorMessage(err);
        }
        return false;
    }

    if (width <= 0 || height <= 0) {
        if (error) {
            *error = "Invalid EXR dimensions.";
        }
        std::free(rgba);
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    pixels->assign(pixelCount, Vec3());
    for (size_t i = 0; i < pixelCount; ++i) {
        size_t base = i * 4;
        (*pixels)[i] = Vec3(rgba[base], rgba[base + 1], rgba[base + 2]);
    }

    std::free(rgba);
    *outWidth = width;
    *outHeight = height;
    return true;
}

bool loadPngImage(const std::string& path,
                  std::vector<Vec3>* pixels,
                  int* outWidth,
                  int* outHeight,
                  std::string* error) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        if (error) {
            *error = stbi_failure_reason() ? stbi_failure_reason() : "Failed to load PNG.";
        }
        return false;
    }
    if (width <= 0 || height <= 0) {
        if (error) {
            *error = "Invalid PNG dimensions.";
        }
        stbi_image_free(data);
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    pixels->assign(pixelCount, Vec3());
    for (size_t i = 0; i < pixelCount; ++i) {
        size_t base = i * 3;
        float r = data[base] / 255.0f;
        float g = data[base + 1] / 255.0f;
        float b = data[base + 2] / 255.0f;
        (*pixels)[i] = Vec3(r, g, b);
    }
    stbi_image_free(data);
    *outWidth = width;
    *outHeight = height;
    return true;
}

Vec3 computeAverageColor(const std::vector<Vec3>& pixels) {
    if (pixels.empty()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    Vec3 sum(0.0f, 0.0f, 0.0f);
    for (const Vec3& c : pixels) {
        sum += c;
    }
    return sum / static_cast<float>(pixels.size());
}

float computeLogAverageLuminance(const std::vector<Vec3>& pixels) {
    if (pixels.empty()) {
        return 0.0f;
    }
    const double kEps = 1e-4;
    double sum = 0.0;
    for (const Vec3& c : pixels) {
        double lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        sum += std::log(kEps + lum);
    }
    return static_cast<float>(std::exp(sum / static_cast<double>(pixels.size())));
}

}  // namespace

EnvironmentMap::~EnvironmentMap() {
    releaseDevice();
}

bool EnvironmentMap::loadFromFile(const std::string& path, std::string* error) {
    std::vector<Vec3> pixels;
    int width = 0;
    int height = 0;
    std::string localError;
    bool loaded = false;
    bool isLdr = false;
    if (endsWithIgnoreCase(path, ".exr")) {
        loaded = loadExrImage(path, &pixels, &width, &height, &localError);
    } else if (endsWithIgnoreCase(path, ".png") ||
               endsWithIgnoreCase(path, ".jpg") ||
               endsWithIgnoreCase(path, ".jpeg")) {
        loaded = loadPngImage(path, &pixels, &width, &height, &localError);
        isLdr = loaded;
    } else {
        loaded = loadHdrImage(path, &pixels, &width, &height, &localError);
    }
    if (!loaded) {
        if (error) {
            *error = localError;
        }
        return false;
    }

    pixels_ = std::move(pixels);
    width_ = width;
    height_ = height;
    averageColor_ = computeAverageColor(pixels_);
    averageLuminance_ = computeLogAverageLuminance(pixels_);
    isLdr_ = isLdr;
    path_ = path;
    deviceDirty_ = true;
    return true;
}

bool EnvironmentMap::uploadToDevice() {
    if (!isValid()) {
        return false;
    }

    int count = width_ * height_;
    if (!devicePixels_ || deviceCount_ != count) {
        releaseDevice();
        checkCuda(cudaMalloc(&devicePixels_, static_cast<size_t>(count) * sizeof(Vec3)), "cudaMalloc envmap");
        deviceCount_ = count;
        deviceDirty_ = true;
    }

    if (deviceDirty_) {
        checkCuda(cudaMemcpy(
                          devicePixels_,
                          pixels_.data(),
                          static_cast<size_t>(count) * sizeof(Vec3),
                          cudaMemcpyHostToDevice),
                  "cudaMemcpy envmap");
        deviceDirty_ = false;
    }

    return true;
}

void EnvironmentMap::releaseDevice() {
    if (devicePixels_) {
        cudaFree(devicePixels_);
        devicePixels_ = nullptr;
    }
    deviceCount_ = 0;
    deviceDirty_ = true;
}
