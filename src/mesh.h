#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <vector_types.h>
#else
// Define uint3 for non-CUDA compilation
struct uint3 {
    unsigned int x, y, z;
};
#endif

#include "bvh_data.h"
#include "material.h"
#include "vec3.h"

// Forward declaration
struct TextureDeviceView;

struct MeshTexture {
    std::vector<unsigned char> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
    ColorSpace colorSpace = ColorSpace::LINEAR;
};

struct TextureDeviceView {
    const unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int srgb = 0;  // 1 if SRGB color space
};

// Device view for indexed mesh with materials
struct MeshDeviceView {
    // Geometry (indexed)
    const Vec3* vertices = nullptr;
    const Vec3* normals = nullptr;          // Per-vertex normals (can be null)
    const Vec2* texcoords = nullptr;        // Per-vertex UVs (can be null)
    const uint3* indices = nullptr;         // Triangle vertex indices
    int numTriangles = 0;
    int numVertices = 0;

    // BVH
    const BvhNode* bvhNodes = nullptr;
    int numBvhNodes = 0;

    // Materials
    const Material* materials = nullptr;
    const uint32_t* materialMap = nullptr;  // Starting triangle index per primitive
    const int* materialIds = nullptr;       // Material ID per primitive
    int numPrimitives = 0;
    int numMaterials = 0;

    // Textures
    const TextureDeviceView* textures = nullptr;
    int numTextures = 0;
    int textureNearest = 0;

    // Flags
    int hasMeshMaterials = 0;  // 1 = use mesh materials, 0 = use global
    int hasNormals = 0;
    int hasTexcoords = 0;
};

class Mesh {
public:
    Mesh() = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;
    ~Mesh();

    // Geometry buffers (indexed)
    std::vector<Vec3> vertices_;
    std::vector<Vec3> normals_;
    std::vector<Vec2> texcoords_;
    std::vector<uint3> indices_;

    // Materials
    std::vector<Material> materials_;
    std::vector<uint32_t> materialMap_;   // Starting triangle index per primitive
    std::vector<int> materialIds_;        // Material ID per primitive (-1 = use global)
    bool hasMeshMaterials_ = false;

    // Textures
    std::vector<MeshTexture> textures_;
    bool textureNearest_ = false;

    // Accessors
    int numTriangles() const { return static_cast<int>(indices_.size()); }
    int numVertices() const { return static_cast<int>(vertices_.size()); }
    bool hasNormals() const { return !normals_.empty(); }
    bool hasTexcoords() const { return !texcoords_.empty(); }
    bool hasMeshMaterials() const { return hasMeshMaterials_; }

    const std::vector<MeshTexture>& textures() const { return textures_; }
    int textureCount() const { return static_cast<int>(textures_.size()); }

    void setTextureNearest(bool nearest) { textureNearest_ = nearest; }
    bool textureNearest() const { return textureNearest_; }

    // BVH
    void buildBvh();
    bool hasBvh() const { return !bvhNodes_.empty(); }
    int nodeCount() const { return static_cast<int>(bvhNodes_.size()); }
    size_t bvhStorageBytes() const { return bvhNodes_.size() * sizeof(BvhNode); }
    Vec3 boundsMin() const { return boundsMin_; }
    Vec3 boundsMax() const { return boundsMax_; }

    // Device management
    bool uploadToDevice();
    MeshDeviceView deviceView() const;
    void releaseDevice();

    // Clear all data
    void clear();

private:
    // BVH
    std::vector<BvhNode> bvhNodes_;
    Vec3 boundsMin_{};
    Vec3 boundsMax_{};
    bool bvhDirty_ = true;
    bool boundsDirty_ = true;

    // Device geometry
    Vec3* deviceVertices_ = nullptr;
    Vec3* deviceNormals_ = nullptr;
    Vec2* deviceTexcoords_ = nullptr;
    uint3* deviceIndices_ = nullptr;
    int deviceNumTriangles_ = 0;
    int deviceNumVertices_ = 0;

    // Device BVH
    BvhNode* deviceBvhNodes_ = nullptr;
    int deviceNumBvhNodes_ = 0;

    // Device materials
    Material* deviceMaterials_ = nullptr;
    uint32_t* deviceMaterialMap_ = nullptr;
    int* deviceMaterialIds_ = nullptr;
    int deviceNumMaterials_ = 0;
    int deviceNumPrimitives_ = 0;

    // Device textures
    std::vector<unsigned char*> deviceTexturePixels_;
    TextureDeviceView* deviceTextures_ = nullptr;
    int deviceNumTextures_ = 0;

    // Dirty flags
    bool geometryDirty_ = true;
    bool materialsDirty_ = true;
    bool texturesDirty_ = true;
    bool bvhNodesDirty_ = true;
};
