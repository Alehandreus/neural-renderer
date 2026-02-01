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
    EnvironmentDeviceView deviceView() const { return EnvironmentDeviceView{devicePixels_, width_, height_}; }
    Vec3 averageColor() const { return averageColor_; }
    float averageLuminance() const { return averageLuminance_; }
    bool isLdr() const { return isLdr_; }
    const std::string& path() const { return path_; }

private:
    std::vector<Vec3> pixels_;
    int width_ = 0;
    int height_ = 0;
    Vec3 averageColor_{};
    float averageLuminance_ = 0.0f;
    bool isLdr_ = false;
    std::string path_;
    Vec3* devicePixels_ = nullptr;
    int deviceCount_ = 0;
    bool deviceDirty_ = true;
};

class Scene {
public:
    Mesh& originalMesh() { return originalMesh_; }
    const Mesh& originalMesh() const { return originalMesh_; }

    Mesh& innerShell() { return innerShell_; }
    const Mesh& innerShell() const { return innerShell_; }

    Mesh& outerShell() { return outerShell_; }
    const Mesh& outerShell() const { return outerShell_; }

    EnvironmentMap& environment() { return environment_; }
    const EnvironmentMap& environment() const { return environment_; }

    Material& material() { return material_; }
    const Material& material() const { return material_; }

private:
    Mesh originalMesh_;
    Mesh innerShell_;
    Mesh outerShell_;
    EnvironmentMap environment_;
    Material material_;
};
