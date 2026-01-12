#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "bvh_data.h"
#include "triangle.h"

struct MeshDeviceView {
    const Triangle* triangles = nullptr;
    int triangleCount = 0;
    const BvhNode* nodes = nullptr;
    int nodeCount = 0;
    const struct TextureDeviceView* textures = nullptr;
    int textureCount = 0;
    int textureNearest = 0;
};

struct TextureDeviceView {
    const unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
};

struct MeshTexture {
    std::vector<unsigned char> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
};

class Mesh {
public:
    Mesh() = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    ~Mesh();

    void setTriangles(std::vector<Triangle> triangles);
    void setTextures(std::vector<MeshTexture> textures);
    void setTextureNearest(bool nearest) { textureNearest_ = nearest; }
    bool textureNearest() const { return textureNearest_; }
    const std::vector<Triangle>& triangles() const { return triangles_; }
    int triangleCount() const { return static_cast<int>(triangles_.size()); }
    const std::vector<MeshTexture>& textures() const { return textures_; }
    int textureCount() const { return static_cast<int>(textures_.size()); }

    void buildBvh();
    bool hasBvh() const { return !nodes_.empty(); }
    int nodeCount() const { return static_cast<int>(nodes_.size()); }
    size_t bvhStorageBytes() const { return nodes_.size() * sizeof(BvhNode); }
    Vec3 boundsMin() const { return boundsMin_; }
    Vec3 boundsMax() const { return boundsMax_; }

    bool uploadToDevice();
    MeshDeviceView deviceView() const {
        return MeshDeviceView{deviceTriangles_, deviceCount_, deviceNodes_, deviceNodeCount_,
                              deviceTextures_, deviceTextureCount_, textureNearest_ ? 1 : 0};
    }
    void releaseDevice();

private:
    std::vector<Triangle> triangles_;
    std::vector<BvhNode> nodes_;
    std::vector<MeshTexture> textures_;
    Triangle* deviceTriangles_ = nullptr;
    int deviceCount_ = 0;
    bool deviceDirty_ = true;
    BvhNode* deviceNodes_ = nullptr;
    int deviceNodeCount_ = 0;
    bool deviceNodesDirty_ = true;
    bool bvhDirty_ = true;
    bool boundsDirty_ = true;
    bool texturesDirty_ = true;
    bool textureNearest_ = false;
    Vec3 boundsMin_{};
    Vec3 boundsMax_{};
    std::vector<unsigned char*> deviceTexturePixels_;
    TextureDeviceView* deviceTextures_ = nullptr;
    int deviceTextureCount_ = 0;
};
