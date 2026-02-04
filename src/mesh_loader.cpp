#include "mesh_loader.h"

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "material.h"
#include "stb_image.h"

namespace {

Vec3 toVec3(const aiVector3D& v) {
    return Vec3(v.x, v.y, v.z);
}

Vec3 toVec3(const aiColor4D& c) {
    return Vec3(c.r, c.g, c.b);
}

Vec2 toVec2(const aiVector3D& v) {
    return Vec2(v.x, v.y);
}

Vec3 normalizeSafe(const Vec3& v, const Vec3& fallback) {
    float len = length(v);
    if (len > 1e-8f) {
        return v / len;
    }
    return fallback;
}

Vec3 srgbToLinear(Vec3 c) {
    auto toLinear = [](float v) {
        if (v <= 0.04045f) {
            return v / 12.92f;
        }
        return powf((v + 0.055f) / 1.055f, 2.4f);
    };
    return Vec3(toLinear(c.x), toLinear(c.y), toLinear(c.z));
}

Vec3 mul(Vec3 a, Vec3 b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

struct ImageData {
    int width = 0;
    int height = 0;
    int channels = 0;
    bool srgb = true;
    std::vector<unsigned char> pixels;
};

void expandBounds(const Vec3& v, Vec3* minV, Vec3* maxV) {
    minV->x = fminf(minV->x, v.x);
    minV->y = fminf(minV->y, v.y);
    minV->z = fminf(minV->z, v.z);
    maxV->x = fmaxf(maxV->x, v.x);
    maxV->y = fmaxf(maxV->y, v.y);
    maxV->z = fmaxf(maxV->z, v.z);
}

void normalizeTriangles(std::vector<Triangle>& triangles) {
    if (triangles.empty()) {
        return;
    }

    Vec3 minV(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 maxV(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const Triangle& tri : triangles) {
        expandBounds(tri.v0, &minV, &maxV);
        expandBounds(tri.v1, &minV, &maxV);
        expandBounds(tri.v2, &minV, &maxV);
    }

    Vec3 size = maxV - minV;
    float maxExtent = fmaxf(size.x, fmaxf(size.y, size.z));
    if (maxExtent <= 0.0f) {
        return;
    }

    Vec3 center = (minV + maxV) * 0.5f;
    float scale = 2.0f / maxExtent * 5.0f;
    for (Triangle& tri : triangles) {
        tri.v0 = (tri.v0 - center) * scale;
        tri.v1 = (tri.v1 - center) * scale;
        tri.v2 = (tri.v2 - center) * scale;
        tri.normal = normalizeSafe(cross(tri.v1 - tri.v0, tri.v2 - tri.v0), tri.normal);
        tri.n0 = normalizeSafe(tri.n0, tri.normal);
        tri.n1 = normalizeSafe(tri.n1, tri.normal);
        tri.n2 = normalizeSafe(tri.n2, tri.normal);
    }
}

void scaleTriangles(std::vector<Triangle>& triangles, float scale) {
    if (triangles.empty() || scale == 1.0f) {
        return;
    }

    for (Triangle& tri : triangles) {
        tri.v0 = tri.v0 * scale;
        tri.v1 = tri.v1 * scale;
        tri.v2 = tri.v2 * scale;
    }
}

bool loadImageFromFile(const std::filesystem::path& path, ImageData* outImage, std::string* error) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    if (!data) {
        if (error) {
            *error = stbi_failure_reason() ? stbi_failure_reason() : "Failed to load image file.";
        }
        return false;
    }
    outImage->width = width;
    outImage->height = height;
    outImage->channels = 4;
    outImage->pixels.assign(data, data + static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    stbi_image_free(data);
    return true;
}

bool loadImageFromEmbedded(const aiTexture* texture, ImageData* outImage, std::string* error) {
    if (!texture) {
        if (error) {
            *error = "Embedded texture is null.";
        }
        return false;
    }
    if (texture->mHeight == 0) {
        int width = 0;
        int height = 0;
        int channels = 0;
        const auto* data = reinterpret_cast<const stbi_uc*>(texture->pcData);
        stbi_uc* decoded = stbi_load_from_memory(
            data,
            static_cast<int>(texture->mWidth),
            &width,
            &height,
            &channels,
            4);
        if (!decoded) {
            if (error) {
                *error = stbi_failure_reason() ? stbi_failure_reason() : "Failed to decode embedded texture.";
            }
            return false;
        }
        outImage->width = width;
        outImage->height = height;
        outImage->channels = 4;
        outImage->pixels.assign(decoded, decoded + static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
        stbi_image_free(decoded);
        return true;
    }

    int width = static_cast<int>(texture->mWidth);
    int height = static_cast<int>(texture->mHeight);
    outImage->width = width;
    outImage->height = height;
    outImage->channels = 4;
    outImage->pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const aiTexel& texel = texture->pcData[static_cast<size_t>(y) * width + x];
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            outImage->pixels[idx + 0] = texel.r;
            outImage->pixels[idx + 1] = texel.g;
            outImage->pixels[idx + 2] = texel.b;
            outImage->pixels[idx + 3] = texel.a;
        }
    }
    return true;
}

Vec3 sampleTexture(const ImageData& image, float u, float v) {
    if (image.width <= 0 || image.height <= 0 || image.pixels.empty()) {
        return Vec3(1.0f, 1.0f, 1.0f);
    }

    u = u - floorf(u);
    v = v - floorf(v);
    v = 1.0f - v;

    float x = u * static_cast<float>(image.width - 1);
    float y = v * static_cast<float>(image.height - 1);
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = std::min(x0 + 1, image.width - 1);
    int y1 = std::min(y0 + 1, image.height - 1);
    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);

    auto fetch = [&](int xi, int yi) {
        size_t idx = (static_cast<size_t>(yi) * image.width + xi) * 4;
        Vec3 c(
            image.pixels[idx + 0] / 255.0f,
            image.pixels[idx + 1] / 255.0f,
            image.pixels[idx + 2] / 255.0f);
        return image.srgb ? srgbToLinear(c) : c;
    };

    Vec3 c00 = fetch(x0, y0);
    Vec3 c10 = fetch(x1, y0);
    Vec3 c01 = fetch(x0, y1);
    Vec3 c11 = fetch(x1, y1);
    Vec3 c0 = lerp(c00, c10, tx);
    Vec3 c1 = lerp(c01, c11, tx);
    return lerp(c0, c1, ty);
}

}  // namespace

bool LoadMeshFromFile(const std::string& path,
                      Mesh* outMesh,
                      std::string* error,
                      bool normalize,
                      float scale) {
    if (!outMesh) {
        if (error) {
            *error = "Output mesh pointer is null.";
        }
        return false;
    }

    const Material defaultMaterial;
    const Vec3 defaultColor = defaultMaterial.base_color;
    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_ImproveCacheLocality |
            aiProcess_PreTransformVertices |
            aiProcess_GenNormals;
    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->HasMeshes()) {
        if (error) {
            *error = importer.GetErrorString();
        }
        return false;
    }

    std::vector<Triangle> triangles;
    triangles.reserve(1024);

    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        if (!mesh || mesh->mNumFaces == 0 || mesh->mNumVertices == 0 || !mesh->mVertices || !mesh->mFaces) {
            continue;
        }

        for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
            const aiFace& face = mesh->mFaces[faceIndex];
            if (face.mNumIndices != 3) {
                continue;
            }

            unsigned int i0 = face.mIndices[0];
            unsigned int i1 = face.mIndices[1];
            unsigned int i2 = face.mIndices[2];
            Vec3 v0 = toVec3(mesh->mVertices[i0]);
            Vec3 v1 = toVec3(mesh->mVertices[i1]);
            Vec3 v2 = toVec3(mesh->mVertices[i2]);
            Vec3 faceNormal = normalizeSafe(cross(v1 - v0, v2 - v0), Vec3(0.0f, 1.0f, 0.0f));
            Vec3 n0 = faceNormal;
            Vec3 n1 = faceNormal;
            Vec3 n2 = faceNormal;
            if (mesh->HasNormals()) {
                n0 = normalizeSafe(toVec3(mesh->mNormals[i0]), faceNormal);
                n1 = normalizeSafe(toVec3(mesh->mNormals[i1]), faceNormal);
                n2 = normalizeSafe(toVec3(mesh->mNormals[i2]), faceNormal);
            }

            Vec3 c0 = defaultColor;
            Vec3 c1 = defaultColor;
            Vec3 c2 = defaultColor;
            if (mesh->HasVertexColors(0)) {
                c0 = toVec3(mesh->mColors[0][i0]);
                c1 = toVec3(mesh->mColors[0][i1]);
                c2 = toVec3(mesh->mColors[0][i2]);
            }

            Triangle tri;
            tri.v0 = v0;
            tri.v1 = v1;
            tri.v2 = v2;
            tri.normal = faceNormal;
            tri.n0 = n0;
            tri.n1 = n1;
            tri.n2 = n2;
            tri.c0 = c0;
            tri.c1 = c1;
            tri.c2 = c2;
            tri.uv0 = Vec2();
            tri.uv1 = Vec2();
            tri.uv2 = Vec2();
            tri.texId = -1;
            triangles.push_back(tri);
        }
    }

    if (triangles.empty()) {
        if (error) {
            *error = "No triangles found in mesh.";
        }
        return false;
    }

    if (normalize) {
        normalizeTriangles(triangles);
    }
    if (scale != 1.0f) {
        scaleTriangles(triangles, scale);
    }
    outMesh->setTriangles(std::move(triangles));
    outMesh->setTextures({});
    outMesh->setTextureNearest(false);
    return true;
}

bool LoadTexturedGltfFromFile(const std::string& path,
                              Mesh* outMesh,
                              std::string* error,
                              bool normalize,
                              bool nearestFilter,
                              float scale) {
    if (!outMesh) {
        if (error) {
            *error = "Output mesh pointer is null.";
        }
        return false;
    }

    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_ImproveCacheLocality |
            aiProcess_PreTransformVertices |
            aiProcess_GenNormals;
    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->HasMeshes()) {
        if (error) {
            *error = importer.GetErrorString();
        }
        return false;
    }

    struct MaterialInfo {
        Vec3 baseColor{1.0f, 1.0f, 1.0f};
        int textureIndex = -1;
        int normalTexIndex = -1;
    };

    std::unordered_map<std::string, int> textureCache;
    std::vector<ImageData> textures;
    std::vector<MaterialInfo> materials(scene->mNumMaterials);
    std::filesystem::path baseDir = std::filesystem::path(path).parent_path();

    auto loadTexture = [&](const aiString& texPath, std::string* texError, bool isSRGB = true) -> int {
        if (texPath.length == 0) {
            return -1;
        }
        std::string key = texPath.C_Str();
        auto it = textureCache.find(key);
        if (it != textureCache.end()) {
            return it->second;
        }

        ImageData image;
        bool loaded = false;
        if (!key.empty() && key[0] == '*') {
            int index = std::atoi(key.c_str() + 1);
            if (index >= 0 && static_cast<unsigned int>(index) < scene->mNumTextures) {
                loaded = loadImageFromEmbedded(scene->mTextures[index], &image, texError);
            }
        } else {
            std::filesystem::path texFile = baseDir / key;
            loaded = loadImageFromFile(texFile, &image, texError);
        }

        if (!loaded) {
            return -1;
        }
        image.srgb = isSRGB;
        int id = static_cast<int>(textures.size());
        textures.push_back(std::move(image));
        textureCache[key] = id;
        return id;
    };

    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        const aiMaterial* mat = scene->mMaterials[i];
        if (!mat) {
            continue;
        }
        MaterialInfo info;
        aiColor4D baseColor(1.0f, 1.0f, 1.0f, 1.0f);
        if (aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &baseColor) == AI_SUCCESS) {
            info.baseColor = toVec3(baseColor);
        } else if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &baseColor) == AI_SUCCESS) {
            info.baseColor = toVec3(baseColor);
        }

        aiString texPath;
        if (mat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS) {
            info.textureIndex = loadTexture(texPath, error, true);
        }
        if (info.textureIndex < 0 &&
            mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
            info.textureIndex = loadTexture(texPath, error, true);
        }

        // Load normal map (linear space, not sRGB) - TODO: implement normal mapping
        if (mat->GetTexture(aiTextureType_NORMALS, 0, &texPath) == AI_SUCCESS) {
            info.normalTexIndex = loadTexture(texPath, error, false);
        } else if (mat->GetTexture(aiTextureType_HEIGHT, 0, &texPath) == AI_SUCCESS) {
            info.normalTexIndex = loadTexture(texPath, error, false);
        }

        materials[i] = info;
    }

    std::vector<Triangle> triangles;
    triangles.reserve(1024);

    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        if (!mesh || mesh->mNumFaces == 0 || mesh->mNumVertices == 0 || !mesh->mVertices || !mesh->mFaces) {
            continue;
        }

        MaterialInfo matInfo;
        if (mesh->mMaterialIndex < materials.size()) {
            matInfo = materials[mesh->mMaterialIndex];
        }

        for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
            const aiFace& face = mesh->mFaces[faceIndex];
            if (face.mNumIndices != 3) {
                continue;
            }

            unsigned int i0 = face.mIndices[0];
            unsigned int i1 = face.mIndices[1];
            unsigned int i2 = face.mIndices[2];
            Vec3 v0 = toVec3(mesh->mVertices[i0]);
            Vec3 v1 = toVec3(mesh->mVertices[i1]);
            Vec3 v2 = toVec3(mesh->mVertices[i2]);
            Vec3 faceNormal = normalizeSafe(cross(v1 - v0, v2 - v0), Vec3(0.0f, 1.0f, 0.0f));
            Vec3 n0 = faceNormal;
            Vec3 n1 = faceNormal;
            Vec3 n2 = faceNormal;
            if (mesh->HasNormals()) {
                n0 = normalizeSafe(toVec3(mesh->mNormals[i0]), faceNormal);
                n1 = normalizeSafe(toVec3(mesh->mNormals[i1]), faceNormal);
                n2 = normalizeSafe(toVec3(mesh->mNormals[i2]), faceNormal);
            }

            Vec3 c0 = matInfo.baseColor;
            Vec3 c1 = matInfo.baseColor;
            Vec3 c2 = matInfo.baseColor;
            if (mesh->HasVertexColors(0)) {
                c0 = mul(c0, toVec3(mesh->mColors[0][i0]));
                c1 = mul(c1, toVec3(mesh->mColors[0][i1]));
                c2 = mul(c2, toVec3(mesh->mColors[0][i2]));
            }
            Vec2 uv0;
            Vec2 uv1;
            Vec2 uv2;
            int texId = -1;
            int normalTexId = -1;
            if (mesh->HasTextureCoords(0)) {
                const aiVector3D tuv0 = mesh->mTextureCoords[0][i0];
                const aiVector3D tuv1 = mesh->mTextureCoords[0][i1];
                const aiVector3D tuv2 = mesh->mTextureCoords[0][i2];
                uv0 = Vec2(tuv0.x, tuv0.y);
                uv1 = Vec2(tuv1.x, tuv1.y);
                uv2 = Vec2(tuv2.x, tuv2.y);
                if (matInfo.textureIndex >= 0) {
                    texId = matInfo.textureIndex;
                }
                if (matInfo.normalTexIndex >= 0) {
                    normalTexId = matInfo.normalTexIndex;
                }
            }

            Triangle tri;
            tri.v0 = v0;
            tri.v1 = v1;
            tri.v2 = v2;
            tri.normal = faceNormal;
            tri.n0 = n0;
            tri.n1 = n1;
            tri.n2 = n2;
            tri.c0 = c0;
            tri.c1 = c1;
            tri.c2 = c2;
            tri.uv0 = uv0;
            tri.uv1 = uv1;
            tri.uv2 = uv2;
            tri.texId = texId;
            tri.normalTexId = normalTexId;
            tri._pad1 = 0;
            tri._pad2 = 0;
            triangles.push_back(tri);
        }
    }

    if (triangles.empty()) {
        if (error) {
            *error = "No triangles found in mesh.";
        }
        return false;
    }

    if (normalize) {
        normalizeTriangles(triangles);
    }
    if (scale != 1.0f) {
        scaleTriangles(triangles, scale);
    }
    std::vector<MeshTexture> meshTextures;
    meshTextures.reserve(textures.size());
    for (ImageData& image : textures) {
        MeshTexture tex;
        tex.width = image.width;
        tex.height = image.height;
        tex.channels = image.channels;
        tex.pixels = std::move(image.pixels);
        meshTextures.push_back(std::move(tex));
    }
    outMesh->setTriangles(std::move(triangles));
    outMesh->setTextures(std::move(meshTextures));
    outMesh->setTextureNearest(nearestFilter);
    return true;
}

void GenerateUvSphere(Mesh* outMesh, int stacks, int slices, float radius) {
    if (!outMesh) {
        return;
    }

    if (stacks < 2 || slices < 3) {
        outMesh->setTriangles({});
        return;
    }

    const Material defaultMaterial;
    const Vec3 defaultColor = defaultMaterial.base_color;
    const float kPi = 3.14159265358979323846f;
    std::vector<Triangle> triangles;
    triangles.reserve(static_cast<size_t>(stacks) * static_cast<size_t>(slices) * 2);

    for (int i = 0; i < stacks; ++i) {
        float v0 = static_cast<float>(i) / static_cast<float>(stacks);
        float v1 = static_cast<float>(i + 1) / static_cast<float>(stacks);
        float phi0 = v0 * kPi;
        float phi1 = v1 * kPi;

        for (int j = 0; j < slices; ++j) {
            float u0 = static_cast<float>(j) / static_cast<float>(slices);
            float u1 = static_cast<float>(j + 1) / static_cast<float>(slices);
            float theta0 = u0 * 2.0f * kPi;
            float theta1 = u1 * 2.0f * kPi;

            Vec3 p00(
                    radius * sinf(phi0) * cosf(theta0),
                    radius * cosf(phi0),
                    radius * sinf(phi0) * sinf(theta0));
            Vec3 p01(
                    radius * sinf(phi0) * cosf(theta1),
                    radius * cosf(phi0),
                    radius * sinf(phi0) * sinf(theta1));
            Vec3 p10(
                    radius * sinf(phi1) * cosf(theta0),
                    radius * cosf(phi1),
                    radius * sinf(phi1) * sinf(theta0));
            Vec3 p11(
                    radius * sinf(phi1) * cosf(theta1),
                    radius * cosf(phi1),
                    radius * sinf(phi1) * sinf(theta1));

            Triangle tri0 = makeTriangle(p00, p10, p11);
            tri0.c0 = defaultColor;
            tri0.c1 = defaultColor;
            tri0.c2 = defaultColor;
            Triangle tri1 = makeTriangle(p00, p11, p01);
            tri1.c0 = defaultColor;
            tri1.c1 = defaultColor;
            tri1.c2 = defaultColor;
            triangles.push_back(tri0);
            triangles.push_back(tri1);
        }
    }

    outMesh->setTriangles(std::move(triangles));
}
