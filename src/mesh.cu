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
}
