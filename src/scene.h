#pragma once

#include <string>
#include <vector>

#include "material.h"
#include "mesh.h"
#include "vec3.h"

struct EnvironmentDeviceView {
    const Vec3* pixels = nullptr;
    int width = 0;
    int height = 0;
    float rotation = 0.0f;  // Rotation in degrees around Y axis
    float strength = 1.0f;  // Environment intensity multiplier
};

class EnvironmentMap {
public:
    EnvironmentMap() = default;
    EnvironmentMap(const EnvironmentMap&) = delete;
    EnvironmentMap& operator=(const EnvironmentMap&) = delete;
    ~EnvironmentMap();

    bool loadFromFile(const std::string& path, std::string* error);
    bool uploadToDevice();
    void releaseDevice();
    bool isValid() const { return width_ > 0 && height_ > 0 && !pixels_.empty(); }
    EnvironmentDeviceView deviceView() const { return EnvironmentDeviceView{devicePixels_, width_, height_, rotation_, strength_}; }
    void setRotation(float rotation) { rotation_ = rotation; }
    void setStrength(float strength) { strength_ = strength; }

private:
    std::vector<Vec3> pixels_;
    int width_ = 0;
    int height_ = 0;
    Vec3* devicePixels_ = nullptr;
    int deviceCount_ = 0;
    bool deviceDirty_ = true;
    float rotation_ = 0.0f;
    float strength_ = 1.0f;
};

class Scene {
public:
    Mesh& originalMesh() { return originalMesh_; }
    const Mesh& originalMesh() const { return originalMesh_; }

    Mesh& innerShell() { return innerShell_; }
    const Mesh& innerShell() const { return innerShell_; }

    Mesh& outerShell() { return outerShell_; }
    const Mesh& outerShell() const { return outerShell_; }

    Mesh& additionalMesh() { return additionalMesh_; }
    const Mesh& additionalMesh() const { return additionalMesh_; }

    EnvironmentMap& environment() { return environment_; }
    const EnvironmentMap& environment() const { return environment_; }

    Material& globalMaterial() { return globalMaterial_; }
    const Material& globalMaterial() const { return globalMaterial_; }

private:
    Mesh originalMesh_;
    Mesh innerShell_;
    Mesh outerShell_;
    Mesh additionalMesh_;
    EnvironmentMap environment_;
    Material globalMaterial_ = Material::defaultMaterial();  // For non-GLTF meshes and neural rendering
};
