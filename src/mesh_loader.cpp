#include "mesh_loader.h"

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// Helper to create uint3 (since makeUint3 is CUDA-only)
inline uint3 makeUint3(unsigned int x, unsigned int y, unsigned int z) {
    uint3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

// TinyGLTF (implementation is provided by ext/tinygltf)
#include "stb_image.h"
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NOEXCEPTION
#include <tiny_gltf.h>

namespace {

Vec3 toVec3(const aiVector3D& v) {
    return Vec3(v.x, v.y, v.z);
}

Vec3 normalizeSafe(const Vec3& v, const Vec3& fallback) {
    float len = length(v);
    if (len > 1e-8f) {
        return v / len;
    }
    return fallback;
}

void expandBounds(const Vec3& v, Vec3* minV, Vec3* maxV) {
    minV->x = fminf(minV->x, v.x);
    minV->y = fminf(minV->y, v.y);
    minV->z = fminf(minV->z, v.z);
    maxV->x = fmaxf(maxV->x, v.x);
    maxV->y = fmaxf(maxV->y, v.y);
    maxV->z = fmaxf(maxV->z, v.z);
}

void normalizeMesh(Mesh* mesh) {
    if (mesh->vertices_.empty()) return;

    Vec3 minV(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 maxV(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const Vec3& v : mesh->vertices_) {
        expandBounds(v, &minV, &maxV);
    }

    Vec3 size = maxV - minV;
    float maxExtent = fmaxf(size.x, fmaxf(size.y, size.z));
    if (maxExtent <= 0.0f) return;

    Vec3 center = (minV + maxV) * 0.5f;
    float scale = 2.0f / maxExtent * 5.0f;

    for (Vec3& v : mesh->vertices_) {
        v = (v - center) * scale;
    }
}

void scaleMesh(Mesh* mesh, float scale) {
    if (scale == 1.0f) return;
    for (Vec3& v : mesh->vertices_) {
        v = v * scale;
    }
}

// Helper to load image data using stb_image
struct ImageData {
    std::vector<unsigned char> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
};

bool loadImageFromMemory(const unsigned char* data, int dataSize, ImageData* out) {
    int w, h, c;
    unsigned char* pixels = stbi_load_from_memory(data, dataSize, &w, &h, &c, 4);
    if (!pixels) return false;

    out->width = w;
    out->height = h;
    out->channels = 4;
    out->pixels.assign(pixels, pixels + w * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool loadImageFromFile(const std::string& path, ImageData* out) {
    int w, h, c;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!pixels) return false;

    out->width = w;
    out->height = h;
    out->channels = 4;
    out->pixels.assign(pixels, pixels + w * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool loadImageFromBytes(const unsigned char* data, int dataSize, ImageData* out) {
    return loadImageFromMemory(data, dataSize, out);
}

bool gltfImageLoader(tinygltf::Image* image,
                     const int image_idx,
                     std::string* err,
                     std::string* warn,
                     int req_width,
                     int req_height,
                     const unsigned char* bytes,
                     int size,
                     void*) {
    (void)warn;
    (void)req_width;
    (void)req_height;

    if (!bytes || size <= 0) {
        if (err) {
            (*err) += "GLTF image " + std::to_string(image_idx) + " has no data.\n";
        }
        return false;
    }

    ImageData decoded;
    if (!loadImageFromBytes(bytes, size, &decoded)) {
        if (err) {
            (*err) += "Failed to decode GLTF image " + std::to_string(image_idx) + "\n";
        }
        return false;
    }

    image->width = decoded.width;
    image->height = decoded.height;
    image->component = decoded.channels;
    image->bits = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    image->image = std::move(decoded.pixels);
    return true;
}

}  // namespace

// =============================================================================
// GLTF Loading with TinyGLTF
// =============================================================================

bool LoadGltfWithMaterials(const std::string& path,
                           Mesh* outMesh,
                           std::string* error,
                           bool normalize,
                           float scale) {
    if (!outMesh) {
        if (error) *error = "Output mesh pointer is null.";
        return false;
    }

    outMesh->clear();

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(gltfImageLoader, nullptr);
    std::string err, warn;

    std::filesystem::path filePath(path);
    bool loadOk = false;

    if (filePath.extension() == ".glb") {
        loadOk = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        loadOk = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }

    if (!warn.empty()) {
        fprintf(stderr, "GLTF Warning: %s\n", warn.c_str());
    }

    if (!loadOk) {
        if (error) *error = err.empty() ? "Failed to load GLTF file" : err;
        return false;
    }

    // Load textures from images
    std::filesystem::path baseDir = filePath.parent_path();
    for (const auto& image : model.images) {
        MeshTexture tex;

        if (!image.image.empty()) {
            // Image data is already loaded by TinyGLTF
            tex.width = image.width;
            tex.height = image.height;
            tex.channels = image.component;
            tex.pixels = image.image;
            tex.colorSpace = ColorSpace::LINEAR;  // Will be set to SRGB for color textures
        } else if (!image.uri.empty()) {
            // Load from external file
            ImageData imgData;
            std::filesystem::path imgPath = baseDir / image.uri;
            if (loadImageFromFile(imgPath.string(), &imgData)) {
                tex.width = imgData.width;
                tex.height = imgData.height;
                tex.channels = imgData.channels;
                tex.pixels = std::move(imgData.pixels);
                tex.colorSpace = ColorSpace::LINEAR;
            }
        }

        outMesh->textures_.push_back(std::move(tex));
    }

    // Load materials
    for (const auto& gltfMat : model.materials) {
        size_t materialIndex = outMesh->materials_.size();
        Material mat = Material::defaultMaterial();

        // Base color
        const auto& pbr = gltfMat.pbrMetallicRoughness;
        mat.base_color.value = Vec3(
            static_cast<float>(pbr.baseColorFactor[0]),
            static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]));
        mat.base_color.textured = false;

        if (pbr.baseColorTexture.index >= 0) {
            int texIdx = model.textures[pbr.baseColorTexture.index].source;
            if (texIdx >= 0 && texIdx < static_cast<int>(outMesh->textures_.size())) {
                mat.base_color.textured = true;
                mat.base_color.textureId = static_cast<uint32_t>(texIdx);
                outMesh->textures_[texIdx].colorSpace = ColorSpace::SRGB;
            }
        }

        if (!mat.base_color.textured) {
            fprintf(stderr,
                    "[GLTF] material %zu constant base color = (%f, %f, %f)\n",
                    materialIndex,
                    mat.base_color.value.x,
                    mat.base_color.value.y,
                    mat.base_color.value.z);
        }

        // Metallic/Roughness
        mat.metallic.value = static_cast<float>(pbr.metallicFactor);
        mat.roughness.value = static_cast<float>(pbr.roughnessFactor);

        if (pbr.metallicRoughnessTexture.index >= 0) {
            int texIdx = model.textures[pbr.metallicRoughnessTexture.index].source;
            if (texIdx >= 0 && texIdx < static_cast<int>(outMesh->textures_.size())) {
                // Metallic in blue channel, roughness in green channel
                mat.metallic.textured = true;
                mat.metallic.textureId = static_cast<uint32_t>(texIdx);
                mat.metallic.channel = 2;  // Blue

                mat.roughness.textured = true;
                mat.roughness.textureId = static_cast<uint32_t>(texIdx);
                mat.roughness.channel = 1;  // Green

                outMesh->textures_[texIdx].colorSpace = ColorSpace::LINEAR;
            }
        }

        if (!mat.metallic.textured) {
            fprintf(stderr,
                    "[GLTF] material %zu constant metallic = %f\n",
                    materialIndex,
                    mat.metallic.value);
        }
        if (!mat.roughness.textured) {
            fprintf(stderr,
                    "[GLTF] material %zu constant roughness = %f\n",
                    materialIndex,
                    mat.roughness.value);
        }
        if (!mat.specular.textured) {
            fprintf(stderr,
                    "[GLTF] material %zu constant specular = %f\n",
                    materialIndex,
                    mat.specular.value);
        }

        mat.metallic.textured = false;
        mat.metallic.value = 0.0f;

        mat.roughness.textured = false;
        mat.roughness.value = 0.0f;

        mat.specular.textured = false;
        mat.specular.value = 0.0f;

        // Normal map
        if (gltfMat.normalTexture.index >= 0) {
            int texIdx = model.textures[gltfMat.normalTexture.index].source;
            if (texIdx >= 0 && texIdx < static_cast<int>(outMesh->textures_.size())) {
                mat.normal.textured = true;
                mat.normal.textureId = static_cast<uint32_t>(texIdx);
                outMesh->textures_[texIdx].colorSpace = ColorSpace::LINEAR;
            }
        }

        // Emission
        if (!gltfMat.emissiveFactor.empty()) {
            mat.base_emission.value = Vec3(
                static_cast<float>(gltfMat.emissiveFactor[0]),
                static_cast<float>(gltfMat.emissiveFactor[1]),
                static_cast<float>(gltfMat.emissiveFactor[2]));

            if (gltfMat.emissiveTexture.index >= 0) {
                int texIdx = model.textures[gltfMat.emissiveTexture.index].source;
                if (texIdx >= 0 && texIdx < static_cast<int>(outMesh->textures_.size())) {
                    mat.base_emission.textured = true;
                    mat.base_emission.textureId = static_cast<uint32_t>(texIdx);
                    outMesh->textures_[texIdx].colorSpace = ColorSpace::SRGB;
                }
            }
        }

        // Extensions: transmission
        auto transIt = gltfMat.extensions.find("KHR_materials_transmission");
        if (transIt != gltfMat.extensions.end()) {
            if (transIt->second.Has("transmissionFactor")) {
                mat.specular_transmission = static_cast<float>(
                    transIt->second.Get("transmissionFactor").GetNumberAsDouble());
            }
        }

        // Extensions: IOR
        auto iorIt = gltfMat.extensions.find("KHR_materials_ior");
        if (iorIt != gltfMat.extensions.end()) {
            if (iorIt->second.Has("ior")) {
                mat.ior = static_cast<float>(iorIt->second.Get("ior").GetNumberAsDouble());
            }
        }

        // Extensions: emissive strength
        auto emStrIt = gltfMat.extensions.find("KHR_materials_emissive_strength");
        if (emStrIt != gltfMat.extensions.end()) {
            if (emStrIt->second.Has("emissiveStrength")) {
                mat.emission_scale = static_cast<float>(
                    emStrIt->second.Get("emissiveStrength").GetNumberAsDouble());
            }
        }

        outMesh->materials_.push_back(mat);
    }

    // Add default material if none exist
    if (outMesh->materials_.empty()) {
        outMesh->materials_.push_back(Material::defaultMaterial());
    }

    // Load mesh geometry
    // Process all scenes and nodes
    std::function<void(int, const std::array<float, 16>&)> processNode;

    auto matrixFromGltf = [](const std::vector<double>& m) -> std::array<float, 16> {
        std::array<float, 16> result;
        for (int i = 0; i < 16; ++i) {
            result[i] = static_cast<float>(m[i]);
        }
        return result;
    };

    auto identityMatrix = []() -> std::array<float, 16> {
        return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    };

    auto multiplyMatrices = [](const std::array<float, 16>& a, const std::array<float, 16>& b) {
        std::array<float, 16> result = {};
        // Both arrays store matrices in column-major (GLTF) layout.
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                result[col * 4 + row] = sum;
            }
        }
        return result;
    };

    auto transformPoint = [](const std::array<float, 16>& m, const Vec3& p) {
        float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
        float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
        float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
        return Vec3(x, y, z);
    };

    auto transformNormal = [](const std::array<float, 16>& m, const Vec3& n) {
        // For normals, use inverse transpose (assuming uniform scale, just use upper 3x3)
        float x = m[0] * n.x + m[4] * n.y + m[8] * n.z;
        float y = m[1] * n.x + m[5] * n.y + m[9] * n.z;
        float z = m[2] * n.x + m[6] * n.y + m[10] * n.z;
        return normalizeSafe(Vec3(x, y, z), Vec3(0, 1, 0));
    };

    processNode = [&](int nodeIdx, const std::array<float, 16>& parentTransform) {
        const tinygltf::Node& node = model.nodes[nodeIdx];

        // Compute node transform
        std::array<float, 16> nodeTransform = identityMatrix();
        if (!node.matrix.empty()) {
            nodeTransform = matrixFromGltf(node.matrix);
        } else {
            // TRS
            std::array<float, 16> T = identityMatrix();
            std::array<float, 16> R = identityMatrix();
            std::array<float, 16> S = identityMatrix();

            if (!node.translation.empty()) {
                T[12] = static_cast<float>(node.translation[0]);
                T[13] = static_cast<float>(node.translation[1]);
                T[14] = static_cast<float>(node.translation[2]);
            }

            if (!node.rotation.empty()) {
                // Quaternion to rotation matrix
                float qx = static_cast<float>(node.rotation[0]);
                float qy = static_cast<float>(node.rotation[1]);
                float qz = static_cast<float>(node.rotation[2]);
                float qw = static_cast<float>(node.rotation[3]);

                R[0] = 1 - 2 * (qy * qy + qz * qz);
                R[1] = 2 * (qx * qy + qz * qw);
                R[2] = 2 * (qx * qz - qy * qw);
                R[4] = 2 * (qx * qy - qz * qw);
                R[5] = 1 - 2 * (qx * qx + qz * qz);
                R[6] = 2 * (qy * qz + qx * qw);
                R[8] = 2 * (qx * qz + qy * qw);
                R[9] = 2 * (qy * qz - qx * qw);
                R[10] = 1 - 2 * (qx * qx + qy * qy);
            }

            if (!node.scale.empty()) {
                S[0] = static_cast<float>(node.scale[0]);
                S[5] = static_cast<float>(node.scale[1]);
                S[10] = static_cast<float>(node.scale[2]);
            }

            nodeTransform = multiplyMatrices(T, multiplyMatrices(R, S));
        }

        std::array<float, 16> worldTransform = multiplyMatrices(parentTransform, nodeTransform);

        // Process mesh if present
        if (node.mesh >= 0) {
            const tinygltf::Mesh& gltfMesh = model.meshes[node.mesh];

            for (const auto& primitive : gltfMesh.primitives) {
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES) continue;

                // Record start of this primitive
                outMesh->materialMap_.push_back(static_cast<uint32_t>(outMesh->indices_.size()));

                uint32_t baseVertex = static_cast<uint32_t>(outMesh->vertices_.size());

                // Get accessors
                auto posIt = primitive.attributes.find("POSITION");
                if (posIt == primitive.attributes.end()) continue;

                const tinygltf::Accessor& posAccessor = model.accessors[posIt->second];
                const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
                const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];
                const float* positions = reinterpret_cast<const float*>(
                    posBuffer.data.data() + posView.byteOffset + posAccessor.byteOffset);

                size_t vertexCount = posAccessor.count;
                size_t posStride = posView.byteStride ? posView.byteStride / sizeof(float) : 3;

                // Load positions
                for (size_t i = 0; i < vertexCount; ++i) {
                    Vec3 pos(positions[i * posStride], positions[i * posStride + 1], positions[i * posStride + 2]);
                    outMesh->vertices_.push_back(transformPoint(worldTransform, pos));
                }

                // Load normals if available, otherwise add defaults per-vertex
                auto normIt = primitive.attributes.find("NORMAL");
                if (normIt != primitive.attributes.end()) {
                    const tinygltf::Accessor& normAccessor = model.accessors[normIt->second];
                    const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
                    const tinygltf::Buffer& normBuffer = model.buffers[normView.buffer];
                    const float* normals = reinterpret_cast<const float*>(
                        normBuffer.data.data() + normView.byteOffset + normAccessor.byteOffset);

                    size_t normStride = normView.byteStride ? normView.byteStride / sizeof(float) : 3;

                    for (size_t i = 0; i < vertexCount; ++i) {
                        Vec3 norm(normals[i * normStride], normals[i * normStride + 1], normals[i * normStride + 2]);
                        outMesh->normals_.push_back(transformNormal(worldTransform, norm));
                    }
                } else {
                    outMesh->normals_.resize(outMesh->normals_.size() + vertexCount, Vec3(0, 1, 0));
                }

                // Load texcoords if available, otherwise add defaults per-vertex
                auto uvIt = primitive.attributes.find("TEXCOORD_0");
                if (uvIt != primitive.attributes.end()) {
                    const tinygltf::Accessor& uvAccessor = model.accessors[uvIt->second];
                    const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
                    const tinygltf::Buffer& uvBuffer = model.buffers[uvView.buffer];
                    const float* uvs = reinterpret_cast<const float*>(
                        uvBuffer.data.data() + uvView.byteOffset + uvAccessor.byteOffset);

                    size_t uvStride = uvView.byteStride ? uvView.byteStride / sizeof(float) : 2;

                    for (size_t i = 0; i < vertexCount; ++i) {
                        outMesh->texcoords_.push_back(Vec2(uvs[i * uvStride], uvs[i * uvStride + 1]));
                    }
                } else {
                    outMesh->texcoords_.resize(outMesh->texcoords_.size() + vertexCount, Vec2(0, 0));
                }

                // Load indices
                if (primitive.indices >= 0) {
                    const tinygltf::Accessor& idxAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& idxView = model.bufferViews[idxAccessor.bufferView];
                    const tinygltf::Buffer& idxBuffer = model.buffers[idxView.buffer];
                    const unsigned char* idxData = idxBuffer.data.data() + idxView.byteOffset + idxAccessor.byteOffset;

                    size_t indexCount = idxAccessor.count;

                    for (size_t i = 0; i < indexCount; i += 3) {
                        uint32_t i0, i1, i2;

                        if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                            const uint16_t* indices16 = reinterpret_cast<const uint16_t*>(idxData);
                            i0 = indices16[i];
                            i1 = indices16[i + 1];
                            i2 = indices16[i + 2];
                        } else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                            const uint32_t* indices32 = reinterpret_cast<const uint32_t*>(idxData);
                            i0 = indices32[i];
                            i1 = indices32[i + 1];
                            i2 = indices32[i + 2];
                        } else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                            i0 = idxData[i];
                            i1 = idxData[i + 1];
                            i2 = idxData[i + 2];
                        } else {
                            continue;
                        }

                        outMesh->indices_.push_back(makeUint3(baseVertex + i0, baseVertex + i1, baseVertex + i2));
                    }
                }

                // Material ID for this primitive
                int matId = primitive.material >= 0 ? primitive.material : 0;
                outMesh->materialIds_.push_back(matId);
            }
        }

        // Process children
        for (int childIdx : node.children) {
            processNode(childIdx, worldTransform);
        }
    };

    // Process all scenes
    for (const auto& scene : model.scenes) {
        for (int nodeIdx : scene.nodes) {
            processNode(nodeIdx, identityMatrix());
        }
    }

    if (outMesh->indices_.empty()) {
        if (error) *error = "No triangles found in GLTF file.";
        return false;
    }

    // Ensure normals array matches vertices if partially filled
    if (!outMesh->normals_.empty() && outMesh->normals_.size() != outMesh->vertices_.size()) {
        // Pad with computed face normals or truncate
        outMesh->normals_.resize(outMesh->vertices_.size(), Vec3(0, 1, 0));
    }

    // Ensure texcoords array matches vertices if partially filled
    if (!outMesh->texcoords_.empty() && outMesh->texcoords_.size() != outMesh->vertices_.size()) {
        outMesh->texcoords_.resize(outMesh->vertices_.size(), Vec2(0, 0));
    }

    outMesh->hasMeshMaterials_ = true;

    if (normalize) {
        normalizeMesh(outMesh);
    }
    if (scale != 1.0f) {
        scaleMesh(outMesh, scale);
    }

    return true;
}

// =============================================================================
// Assimp Loading (OBJ, FBX, etc.) - No textures, no materials
// =============================================================================

bool LoadMeshFromFile(const std::string& path,
                      Mesh* outMesh,
                      std::string* error,
                      bool normalize,
                      float scale) {
    if (!outMesh) {
        if (error) *error = "Output mesh pointer is null.";
        return false;
    }

    outMesh->clear();

    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate |
                         aiProcess_JoinIdenticalVertices |
                         aiProcess_ImproveCacheLocality |
                         aiProcess_PreTransformVertices |
                         aiProcess_GenNormals;

    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->HasMeshes()) {
        if (error) *error = importer.GetErrorString();
        return false;
    }

    // Single primitive for the entire mesh
    outMesh->materialMap_.push_back(0);
    outMesh->materialIds_.push_back(-1);  // Use global material

    uint32_t baseVertex = 0;

    for (unsigned int meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
        const aiMesh* mesh = scene->mMeshes[meshIdx];
        if (!mesh || mesh->mNumFaces == 0 || mesh->mNumVertices == 0) continue;

        // Load vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            outMesh->vertices_.push_back(toVec3(mesh->mVertices[i]));
        }

        // Load normals
        if (mesh->HasNormals()) {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                outMesh->normals_.push_back(normalizeSafe(toVec3(mesh->mNormals[i]), Vec3(0, 1, 0)));
            }
        }

        // Load faces as indices
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            const aiFace& face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue;

            outMesh->indices_.push_back(makeUint3(
                baseVertex + face.mIndices[0],
                baseVertex + face.mIndices[1],
                baseVertex + face.mIndices[2]));
        }

        baseVertex = static_cast<uint32_t>(outMesh->vertices_.size());
    }

    if (outMesh->indices_.empty()) {
        if (error) *error = "No triangles found in mesh.";
        return false;
    }

    // Ensure normals match vertices
    if (!outMesh->normals_.empty() && outMesh->normals_.size() != outMesh->vertices_.size()) {
        outMesh->normals_.resize(outMesh->vertices_.size(), Vec3(0, 1, 0));
    }

    outMesh->hasMeshMaterials_ = false;

    if (normalize) {
        normalizeMesh(outMesh);
    }
    if (scale != 1.0f) {
        scaleMesh(outMesh, scale);
    }

    return true;
}

// =============================================================================
// Auto-detect and load
// =============================================================================

bool LoadMeshAuto(const std::string& path,
                  Mesh* outMesh,
                  std::string* error,
                  bool normalize,
                  float scale) {
    std::filesystem::path filePath(path);
    std::string ext = filePath.extension().string();

    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".gltf" || ext == ".glb") {
        return LoadGltfWithMaterials(path, outMesh, error, normalize, scale);
    } else {
        return LoadMeshFromFile(path, outMesh, error, normalize, scale);
    }
}

// =============================================================================
// UV Sphere generation
// =============================================================================

void GenerateUvSphere(Mesh* outMesh, int stacks, int slices, float radius) {
    if (!outMesh) return;

    outMesh->clear();

    if (stacks < 2 || slices < 3) return;

    const float kPi = 3.14159265358979323846f;

    // Generate vertices and normals
    for (int i = 0; i <= stacks; ++i) {
        float v = static_cast<float>(i) / static_cast<float>(stacks);
        float phi = v * kPi;

        for (int j = 0; j <= slices; ++j) {
            float u = static_cast<float>(j) / static_cast<float>(slices);
            float theta = u * 2.0f * kPi;

            float x = sinf(phi) * cosf(theta);
            float y = cosf(phi);
            float z = sinf(phi) * sinf(theta);

            outMesh->vertices_.push_back(Vec3(x * radius, y * radius, z * radius));
            outMesh->normals_.push_back(Vec3(x, y, z));
            outMesh->texcoords_.push_back(Vec2(u, v));
        }
    }

    // Generate indices
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t p0 = i * (slices + 1) + j;
            uint32_t p1 = p0 + 1;
            uint32_t p2 = p0 + (slices + 1);
            uint32_t p3 = p2 + 1;

            outMesh->indices_.push_back(makeUint3(p0, p2, p1));
            outMesh->indices_.push_back(makeUint3(p1, p2, p3));
        }
    }

    // Single primitive, use global material
    outMesh->materialMap_.push_back(0);
    outMesh->materialIds_.push_back(-1);
    outMesh->hasMeshMaterials_ = false;
}
