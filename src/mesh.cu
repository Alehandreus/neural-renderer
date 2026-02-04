#include "mesh.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace {

void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

}  // namespace

Mesh::~Mesh() {
    releaseDevice();
}

void Mesh::setTriangles(std::vector<Triangle> triangles) {
    triangles_ = std::move(triangles);
    nodes_.clear();
    deviceDirty_ = true;
    deviceNodesDirty_ = true;
    bvhDirty_ = true;
    boundsDirty_ = true;
}

void Mesh::setTextures(std::vector<MeshTexture> textures) {
    textures_ = std::move(textures);
    texturesDirty_ = true;
}

bool Mesh::uploadToDevice() {
    if (bvhDirty_) {
        buildBvh();
    }

    int count = static_cast<int>(triangles_.size());
    if (count == 0 || nodes_.empty()) {
        releaseDevice();
        deviceDirty_ = false;
        deviceNodesDirty_ = false;
        return false;
    }

    if (deviceCount_ != count || deviceTriangles_ == nullptr) {
        releaseDevice();
        checkCuda(cudaMalloc(&deviceTriangles_, static_cast<size_t>(count) * sizeof(Triangle)), "cudaMalloc");
        deviceCount_ = count;
        deviceDirty_ = true;
    }

    if (deviceDirty_) {
        checkCuda(cudaMemcpy(
            deviceTriangles_,
            triangles_.data(),
            static_cast<size_t>(count) * sizeof(Triangle),
            cudaMemcpyHostToDevice),
            "cudaMemcpy");
        deviceDirty_ = false;
    }

    int nodeCount = static_cast<int>(nodes_.size());
    if (deviceNodeCount_ != nodeCount || deviceNodes_ == nullptr) {
        if (deviceNodes_) {
            cudaFree(deviceNodes_);
            deviceNodes_ = nullptr;
        }
        checkCuda(cudaMalloc(&deviceNodes_, static_cast<size_t>(nodeCount) * sizeof(BvhNode)), "cudaMalloc");
        deviceNodeCount_ = nodeCount;
        deviceNodesDirty_ = true;
    }

    if (deviceNodesDirty_) {
        checkCuda(cudaMemcpy(
            deviceNodes_,
            nodes_.data(),
            static_cast<size_t>(nodeCount) * sizeof(BvhNode),
            cudaMemcpyHostToDevice),
            "cudaMemcpy");
        deviceNodesDirty_ = false;
    }

    if (texturesDirty_ || deviceTextureCount_ != static_cast<int>(textures_.size())) {
        if (deviceTextures_) {
            cudaFree(deviceTextures_);
            deviceTextures_ = nullptr;
        }
        for (unsigned char* ptr : deviceTexturePixels_) {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        deviceTexturePixels_.clear();
        deviceTextureCount_ = 0;

        if (!textures_.empty()) {
            deviceTexturePixels_.resize(textures_.size(), nullptr);
            std::vector<TextureDeviceView> hostViews;
            hostViews.reserve(textures_.size());
            for (size_t i = 0; i < textures_.size(); ++i) {
                const MeshTexture& tex = textures_[i];
                size_t byteCount = static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) *
                        static_cast<size_t>(tex.channels);
                unsigned char* devicePixels = nullptr;
                if (byteCount > 0 && !tex.pixels.empty()) {
                    checkCuda(cudaMalloc(&devicePixels, byteCount), "cudaMalloc texture pixels");
                    checkCuda(cudaMemcpy(
                        devicePixels,
                        tex.pixels.data(),
                        byteCount,
                        cudaMemcpyHostToDevice),
                        "cudaMemcpy texture pixels");
                }
                deviceTexturePixels_[i] = devicePixels;
                hostViews.push_back(TextureDeviceView{devicePixels, tex.width, tex.height, tex.channels});
            }
            if (!hostViews.empty()) {
                checkCuda(cudaMalloc(
                    &deviceTextures_,
                    hostViews.size() * sizeof(TextureDeviceView)),
                    "cudaMalloc texture views");
                checkCuda(cudaMemcpy(
                    deviceTextures_,
                    hostViews.data(),
                    hostViews.size() * sizeof(TextureDeviceView),
                    cudaMemcpyHostToDevice),
                    "cudaMemcpy texture views");
                deviceTextureCount_ = static_cast<int>(hostViews.size());
            }
        }
        texturesDirty_ = false;
    }

    return true;
}

void Mesh::releaseDevice() {
    if (deviceTriangles_) {
        cudaFree(deviceTriangles_);
        deviceTriangles_ = nullptr;
    }
    deviceCount_ = 0;
    if (deviceNodes_) {
        cudaFree(deviceNodes_);
        deviceNodes_ = nullptr;
    }
    deviceNodeCount_ = 0;
    if (deviceTextures_) {
        cudaFree(deviceTextures_);
        deviceTextures_ = nullptr;
    }
    for (unsigned char* ptr : deviceTexturePixels_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    deviceTexturePixels_.clear();
    deviceTextureCount_ = 0;
}

bool Mesh::hasVertexColors() const {
    const float kEpsilon = 0.01f;
    for (const Triangle& tri : triangles_) {
        // Check if any vertex color is significantly different from white
        if (std::abs(tri.c0.x - 1.0f) > kEpsilon || std::abs(tri.c0.y - 1.0f) > kEpsilon || std::abs(tri.c0.z - 1.0f) > kEpsilon ||
            std::abs(tri.c1.x - 1.0f) > kEpsilon || std::abs(tri.c1.y - 1.0f) > kEpsilon || std::abs(tri.c1.z - 1.0f) > kEpsilon ||
            std::abs(tri.c2.x - 1.0f) > kEpsilon || std::abs(tri.c2.y - 1.0f) > kEpsilon || std::abs(tri.c2.z - 1.0f) > kEpsilon) {
            return true;
        }
    }
    return false;
}

void Mesh::overrideVertexColors(Vec3 color) {
    for (Triangle& tri : triangles_) {
        tri.c0 = color;
        tri.c1 = color;
        tri.c2 = color;
    }
    deviceDirty_ = true;
}
