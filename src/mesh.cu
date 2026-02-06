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

Mesh::Mesh(Mesh&& other) noexcept {
    *this = std::move(other);
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        releaseDevice();

        vertices_ = std::move(other.vertices_);
        normals_ = std::move(other.normals_);
        texcoords_ = std::move(other.texcoords_);
        indices_ = std::move(other.indices_);

        materials_ = std::move(other.materials_);
        materialMap_ = std::move(other.materialMap_);
        materialIds_ = std::move(other.materialIds_);
        hasMeshMaterials_ = other.hasMeshMaterials_;

        textures_ = std::move(other.textures_);
        textureNearest_ = other.textureNearest_;

        bvhNodes_ = std::move(other.bvhNodes_);
        boundsMin_ = other.boundsMin_;
        boundsMax_ = other.boundsMax_;

        bvhDirty_ = other.bvhDirty_;
        boundsDirty_ = other.boundsDirty_;
        geometryDirty_ = other.geometryDirty_;
        materialsDirty_ = other.materialsDirty_;
        texturesDirty_ = other.texturesDirty_;
        bvhNodesDirty_ = other.bvhNodesDirty_;

        // Reset other's dirty flags so it doesn't try to upload
        other.hasMeshMaterials_ = false;
        other.textureNearest_ = false;
    }
    return *this;
}

Mesh::~Mesh() {
    releaseDevice();
}

void Mesh::clear() {
    releaseDevice();

    vertices_.clear();
    normals_.clear();
    texcoords_.clear();
    indices_.clear();

    materials_.clear();
    materialMap_.clear();
    materialIds_.clear();
    hasMeshMaterials_ = false;

    textures_.clear();
    textureNearest_ = false;

    bvhNodes_.clear();
    boundsMin_ = Vec3();
    boundsMax_ = Vec3();

    bvhDirty_ = true;
    boundsDirty_ = true;
    geometryDirty_ = true;
    materialsDirty_ = true;
    texturesDirty_ = true;
    bvhNodesDirty_ = true;
}

bool Mesh::uploadToDevice() {
    if (bvhDirty_) {
        buildBvh();
    }

    int numTris = numTriangles();
    int numVerts = numVertices();

    if (numTris == 0 || numVerts == 0 || bvhNodes_.empty()) {
        releaseDevice();
        geometryDirty_ = false;
        bvhNodesDirty_ = false;
        return false;
    }

    // Upload geometry
    if (geometryDirty_) {
        // Free old geometry
        if (deviceVertices_) { cudaFree(deviceVertices_); deviceVertices_ = nullptr; }
        if (deviceNormals_) { cudaFree(deviceNormals_); deviceNormals_ = nullptr; }
        if (deviceTexcoords_) { cudaFree(deviceTexcoords_); deviceTexcoords_ = nullptr; }
        if (deviceIndices_) { cudaFree(deviceIndices_); deviceIndices_ = nullptr; }

        // Vertices
        checkCuda(cudaMalloc(&deviceVertices_, numVerts * sizeof(Vec3)), "cudaMalloc vertices");
        checkCuda(cudaMemcpy(deviceVertices_, vertices_.data(), numVerts * sizeof(Vec3), cudaMemcpyHostToDevice), "cudaMemcpy vertices");

        // Normals (optional)
        if (!normals_.empty()) {
            checkCuda(cudaMalloc(&deviceNormals_, normals_.size() * sizeof(Vec3)), "cudaMalloc normals");
            checkCuda(cudaMemcpy(deviceNormals_, normals_.data(), normals_.size() * sizeof(Vec3), cudaMemcpyHostToDevice), "cudaMemcpy normals");
        }

        // Texcoords (optional)
        if (!texcoords_.empty()) {
            checkCuda(cudaMalloc(&deviceTexcoords_, texcoords_.size() * sizeof(Vec2)), "cudaMalloc texcoords");
            checkCuda(cudaMemcpy(deviceTexcoords_, texcoords_.data(), texcoords_.size() * sizeof(Vec2), cudaMemcpyHostToDevice), "cudaMemcpy texcoords");
        }

        // Indices
        checkCuda(cudaMalloc(&deviceIndices_, numTris * sizeof(uint3)), "cudaMalloc indices");
        checkCuda(cudaMemcpy(deviceIndices_, indices_.data(), numTris * sizeof(uint3), cudaMemcpyHostToDevice), "cudaMemcpy indices");

        deviceNumTriangles_ = numTris;
        deviceNumVertices_ = numVerts;
        geometryDirty_ = false;
    }

    // Upload BVH nodes
    if (bvhNodesDirty_) {
        if (deviceBvhNodes_) { cudaFree(deviceBvhNodes_); deviceBvhNodes_ = nullptr; }

        int nodeCount = static_cast<int>(bvhNodes_.size());
        checkCuda(cudaMalloc(&deviceBvhNodes_, nodeCount * sizeof(BvhNode)), "cudaMalloc BVH nodes");
        checkCuda(cudaMemcpy(deviceBvhNodes_, bvhNodes_.data(), nodeCount * sizeof(BvhNode), cudaMemcpyHostToDevice), "cudaMemcpy BVH nodes");

        deviceNumBvhNodes_ = nodeCount;
        bvhNodesDirty_ = false;
    }

    // Upload materials
    if (materialsDirty_) {
        if (deviceMaterials_) { cudaFree(deviceMaterials_); deviceMaterials_ = nullptr; }
        if (deviceMaterialMap_) { cudaFree(deviceMaterialMap_); deviceMaterialMap_ = nullptr; }
        if (deviceMaterialIds_) { cudaFree(deviceMaterialIds_); deviceMaterialIds_ = nullptr; }

        if (!materials_.empty()) {
            checkCuda(cudaMalloc(&deviceMaterials_, materials_.size() * sizeof(Material)), "cudaMalloc materials");
            checkCuda(cudaMemcpy(deviceMaterials_, materials_.data(), materials_.size() * sizeof(Material), cudaMemcpyHostToDevice), "cudaMemcpy materials");
            deviceNumMaterials_ = static_cast<int>(materials_.size());
        }

        if (!materialMap_.empty()) {
            checkCuda(cudaMalloc(&deviceMaterialMap_, materialMap_.size() * sizeof(uint32_t)), "cudaMalloc materialMap");
            checkCuda(cudaMemcpy(deviceMaterialMap_, materialMap_.data(), materialMap_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy materialMap");
        }

        if (!materialIds_.empty()) {
            checkCuda(cudaMalloc(&deviceMaterialIds_, materialIds_.size() * sizeof(int)), "cudaMalloc materialIds");
            checkCuda(cudaMemcpy(deviceMaterialIds_, materialIds_.data(), materialIds_.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy materialIds");
            deviceNumPrimitives_ = static_cast<int>(materialIds_.size());
        }

        materialsDirty_ = false;
    }

    // Upload textures
    if (texturesDirty_) {
        // Free old textures
        if (deviceTextures_) { cudaFree(deviceTextures_); deviceTextures_ = nullptr; }
        for (unsigned char* ptr : deviceTexturePixels_) {
            if (ptr) cudaFree(ptr);
        }
        deviceTexturePixels_.clear();
        deviceNumTextures_ = 0;

        if (!textures_.empty()) {
            deviceTexturePixels_.resize(textures_.size(), nullptr);
            std::vector<TextureDeviceView> hostViews;
            hostViews.reserve(textures_.size());

            for (size_t i = 0; i < textures_.size(); ++i) {
                const MeshTexture& tex = textures_[i];
                size_t byteCount = static_cast<size_t>(tex.width) * tex.height * tex.channels;
                unsigned char* devicePixels = nullptr;

                if (byteCount > 0 && !tex.pixels.empty()) {
                    checkCuda(cudaMalloc(&devicePixels, byteCount), "cudaMalloc texture pixels");
                    checkCuda(cudaMemcpy(devicePixels, tex.pixels.data(), byteCount, cudaMemcpyHostToDevice), "cudaMemcpy texture pixels");
                }
                deviceTexturePixels_[i] = devicePixels;

                hostViews.push_back(TextureDeviceView{
                    devicePixels,
                    tex.width,
                    tex.height,
                    tex.channels,
                    tex.colorSpace == ColorSpace::SRGB ? 1 : 0
                });
            }

            if (!hostViews.empty()) {
                checkCuda(cudaMalloc(&deviceTextures_, hostViews.size() * sizeof(TextureDeviceView)), "cudaMalloc texture views");
                checkCuda(cudaMemcpy(deviceTextures_, hostViews.data(), hostViews.size() * sizeof(TextureDeviceView), cudaMemcpyHostToDevice), "cudaMemcpy texture views");
                deviceNumTextures_ = static_cast<int>(hostViews.size());
            }
        }
        texturesDirty_ = false;
    }

    return true;
}

MeshDeviceView Mesh::deviceView() const {
    MeshDeviceView view;

    // Geometry
    view.vertices = deviceVertices_;
    view.normals = deviceNormals_;
    view.texcoords = deviceTexcoords_;
    view.indices = deviceIndices_;
    view.numTriangles = deviceNumTriangles_;
    view.numVertices = deviceNumVertices_;

    // BVH
    view.bvhNodes = deviceBvhNodes_;
    view.numBvhNodes = deviceNumBvhNodes_;

    // Materials
    view.materials = deviceMaterials_;
    view.materialMap = deviceMaterialMap_;
    view.materialIds = deviceMaterialIds_;
    view.numPrimitives = deviceNumPrimitives_;
    view.numMaterials = deviceNumMaterials_;

    // Textures
    view.textures = deviceTextures_;
    view.numTextures = deviceNumTextures_;
    view.textureNearest = textureNearest_ ? 1 : 0;

    // Flags
    view.hasMeshMaterials = hasMeshMaterials_ ? 1 : 0;
    view.hasNormals = !normals_.empty() ? 1 : 0;
    view.hasTexcoords = !texcoords_.empty() ? 1 : 0;

    return view;
}

void Mesh::releaseDevice() {
    if (deviceVertices_) { cudaFree(deviceVertices_); deviceVertices_ = nullptr; }
    if (deviceNormals_) { cudaFree(deviceNormals_); deviceNormals_ = nullptr; }
    if (deviceTexcoords_) { cudaFree(deviceTexcoords_); deviceTexcoords_ = nullptr; }
    if (deviceIndices_) { cudaFree(deviceIndices_); deviceIndices_ = nullptr; }
    deviceNumTriangles_ = 0;
    deviceNumVertices_ = 0;

    if (deviceBvhNodes_) { cudaFree(deviceBvhNodes_); deviceBvhNodes_ = nullptr; }
    deviceNumBvhNodes_ = 0;

    if (deviceMaterials_) { cudaFree(deviceMaterials_); deviceMaterials_ = nullptr; }
    if (deviceMaterialMap_) { cudaFree(deviceMaterialMap_); deviceMaterialMap_ = nullptr; }
    if (deviceMaterialIds_) { cudaFree(deviceMaterialIds_); deviceMaterialIds_ = nullptr; }
    deviceNumMaterials_ = 0;
    deviceNumPrimitives_ = 0;

    if (deviceTextures_) { cudaFree(deviceTextures_); deviceTextures_ = nullptr; }
    for (unsigned char* ptr : deviceTexturePixels_) {
        if (ptr) cudaFree(ptr);
    }
    deviceTexturePixels_.clear();
    deviceNumTextures_ = 0;
}
