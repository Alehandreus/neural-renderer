#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "cuda_renderer_neural.h"
#include "input_controller.h"
#include "mesh_loader.h"
#include "scene.h"

namespace {

void glfwErrorCallback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

bool loadMesh(const char* path, Mesh* mesh, const char* label, bool normalize, bool nearestTex) {
    if (!path || path[0] == '\0') return false;
    std::string loadError;
    std::filesystem::path meshPath(path);
    std::string ext = meshPath.extension().string();
    for (char& ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    bool loaded = false;
    if (ext == ".gltf" || ext == ".glb") {
        loaded = LoadTexturedGltfFromFile(path, mesh, &loadError, normalize, nearestTex);
    } else {
        loaded = LoadMeshFromFile(path, mesh, &loadError, normalize);
    }
    if (!loaded) {
        std::fprintf(stderr, "Failed to load %s mesh '%s': %s\n", label, path, loadError.c_str());
    }
    return loaded;
}

}  // namespace

int main(int argc, char** argv) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const int startWidth = 1920;
    const int startHeight = 1080;
    GLFWwindow* window = glfwCreateWindow(startWidth, startHeight, "CUDA ImGui Sphere", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    InputController input(window);

    // const char* kOriginalMeshPath = "/home/me/brain/mesh-mapping/models/superdragon_orig.obj";
    // const char* kInnerShellPath = "/home/me/brain/mesh-mapping/models/superdragon_inner_5000.obj";
    // const char* kOuterShellPath = "/home/me/brain/mesh-mapping/models/superdragon_outer_5000.obj";
    const char* kOriginalMeshPath = "/home/me/Downloads/chess_orig.fbx";
    const char* kInnerShellPath = "/home/me/Downloads/chess_outer_10000.fbx";
    const char* kOuterShellPath = "/home/me/Downloads/chess_inner_10000.fbx";

    const char* kCheckpointPath = "/home/me/brain/mesh-mapping/checkpoints/run_1770028802.bin";
    const int kBounceCount = 3;
    const int kSamplesPerPixel = 1;
    const bool kNormalizeMeshes = false;
    const bool kNearestTextureSampling = true;
    const char* kDefaultHdriPath = "/home/me/Downloads/lilienstein_4k.hdr";

    Scene scene;
    Mesh& originalMesh = scene.originalMesh();
    Mesh& innerShell = scene.innerShell();
    Mesh& outerShell = scene.outerShell();

    std::string originalMeshLabel = "procedural sphere";
    std::string innerShellLabel = "(none)";
    std::string outerShellLabel = "(none)";

    if (loadMesh(kOriginalMeshPath, &originalMesh, "original", kNormalizeMeshes, kNearestTextureSampling)) {
        originalMeshLabel = kOriginalMeshPath;
    }
    if (originalMesh.triangleCount() == 0) {
        GenerateUvSphere(&originalMesh, 48, 96, 1.0f);
    }

    if (loadMesh(kInnerShellPath, &innerShell, "inner shell", kNormalizeMeshes, false)) {
        innerShellLabel = kInnerShellPath;
    }

    if (loadMesh(kOuterShellPath, &outerShell, "outer shell", kNormalizeMeshes, false)) {
        outerShellLabel = kOuterShellPath;
    }

    std::string envError;
    if (!scene.environment().loadFromFile(kDefaultHdriPath, &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", kDefaultHdriPath, envError.c_str());
    }

    // Set camera position and speed based on mesh bounds.
    {
        Vec3 bmin = originalMesh.boundsMin();
        Vec3 bmax = originalMesh.boundsMax();
        Vec3 center((bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f, (bmin.z + bmax.z) * 0.5f);
        Vec3 ext(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);
        float diagonal = std::sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);

        if (diagonal > 0.0f) {
            input.setMoveSpeed(diagonal * 0.15f);

            // Place camera at front-right, slightly above, far enough to see the full mesh.
            float fovY = input.camera().fovY;  // radians
            float dist = (diagonal * 0.5f) / std::tan(fovY * 0.5f) * 1.1f;
            // Front-right direction: (+X, +Z), normalized (1,0,1)/sqrt(2), slight upward offset.
            constexpr float kInvSqrt2 = 0.70710678f;
            Vec3 camPos(
                center.x + dist * kInvSqrt2,
                center.y + diagonal * 0.2f,
                center.z + dist * kInvSqrt2);

            input.camera().position = camPos;

            // Compute yaw/pitch to look at mesh center.
            Vec3 forward(center.x - camPos.x, center.y - camPos.y, center.z - camPos.z);
            float hLen = std::sqrt(forward.x * forward.x + forward.z * forward.z);
            constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
            input.camera().yaw = std::atan2(forward.z, forward.x) * kRadToDeg;
            input.camera().pitch = std::atan2(forward.y, hLen) * kRadToDeg;
        }
    }

    RendererNeural renderer(scene);
    bool loaded = false;
    renderer.setUseNeuralQuery(false);
    renderer.setBounceCount(kBounceCount);
    renderer.setSamplesPerPixel(kSamplesPerPixel);
    if (kCheckpointPath && kCheckpointPath[0] != '\0') {
        loaded = renderer.loadWeightsFromFile(kCheckpointPath);
    }
    if (loaded) {
        std::printf("Neural parameters loaded from file.\n");
    } else {
        std::printf("Neural parameters not loaded (using initialization).\n");
    }
    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    renderer.resize(fbWidth, fbHeight);
    std::vector<uchar4> hostPixels(static_cast<size_t>(fbWidth) * static_cast<size_t>(fbHeight));

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            fbWidth,
            fbHeight,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr);

    double lastTime = glfwGetTime();
    bool lambertView = false;
    bool useNeuralQuery = true;
    int bounceCount = kBounceCount;
    int samplesPerPixel = kSamplesPerPixel;
    int classicMeshIndex = 0;
    bool uiWantsMouse = false;

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - lastTime);
        lastTime = now;

        input.update(dt, uiWantsMouse);

        const CameraState& camera = input.camera();
        const CameraBasis& basis = input.basis();

        if (bounceCount < 0) bounceCount = 0;
        if (samplesPerPixel < 1) samplesPerPixel = 1;

        renderer.setLambertView(lambertView);
        renderer.setUseNeuralQuery(useNeuralQuery);
        renderer.setBounceCount(bounceCount);
        renderer.setSamplesPerPixel(samplesPerPixel);
        renderer.setClassicMeshIndex(classicMeshIndex);

        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth != renderer.width() || fbHeight != renderer.height()) {
            if (fbWidth > 0 && fbHeight > 0) {
                renderer.resize(fbWidth, fbHeight);
                hostPixels.resize(static_cast<size_t>(fbWidth) * static_cast<size_t>(fbHeight));
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(
                        GL_TEXTURE_2D,
                        0,
                        GL_RGBA,
                        fbWidth,
                        fbHeight,
                        0,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        nullptr);
            }
        }

        if (fbWidth > 0 && fbHeight > 0) {
            RenderBasis renderBasis;
            renderBasis.forward = basis.forward;
            renderBasis.right = basis.right;
            renderBasis.up = basis.up;
            renderBasis.fovY = camera.fovY;
            renderer.setCameraBasis(renderBasis);
            renderer.render(camera.position, hostPixels);
            glBindTexture(GL_TEXTURE_2D, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    fbWidth,
                    fbHeight,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    hostPixels.data());
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiWindowFlags viewportFlags = ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse |
                ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("Viewport", nullptr, viewportFlags);
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(texture)), io.DisplaySize);
        ImGui::End();
        ImGui::PopStyleVar();

        ImGuiWindowFlags infoFlags = ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav;
        ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
        ImGui::Begin("Info", nullptr, infoFlags);
        ImGui::Text("WASD move, Q/E up/down, mouse look.");
        ImGui::Text("ESC releases mouse, click to recapture.");
        ImGui::Checkbox("Neural query", &useNeuralQuery);
        ImGui::Checkbox("Lambert (no bounces)", &lambertView);
        ImGui::InputInt("Max bounces", &bounceCount);
        ImGui::InputInt("Samples per pixel", &samplesPerPixel);
        const char* meshNames[] = {"Original", "Inner shell", "Outer shell"};
        ImGui::Combo("Classic mesh", &classicMeshIndex, meshNames, 3);
        ImGui::Text("Resolution: %d x %d", fbWidth, fbHeight);
        ImGui::Separator();
        ImGui::Text("Original: %s", originalMeshLabel.c_str());
        ImGui::Text("  triangles: %d, BVH: %d (%.2f MB)",
                    originalMesh.triangleCount(),
                    originalMesh.nodeCount(),
                    static_cast<double>(originalMesh.bvhStorageBytes()) / (1024.0 * 1024.0));
        ImGui::Text("Inner shell: %s", innerShellLabel.c_str());
        if (innerShell.triangleCount() > 0) {
            ImGui::Text("  triangles: %d, BVH: %d (%.2f MB)",
                        innerShell.triangleCount(),
                        innerShell.nodeCount(),
                        static_cast<double>(innerShell.bvhStorageBytes()) / (1024.0 * 1024.0));
        }
        ImGui::Text("Outer shell: %s", outerShellLabel.c_str());
        if (outerShell.triangleCount() > 0) {
            ImGui::Text("  triangles: %d, BVH: %d (%.2f MB)",
                        outerShell.triangleCount(),
                        outerShell.nodeCount(),
                        static_cast<double>(outerShell.bvhStorageBytes()) / (1024.0 * 1024.0));
        }
        ImGui::Text("Network params (fp16): %.2f MB",
                    static_cast<double>(renderer.paramsBytes()) / (1024.0 * 1024.0));
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::End();

        uiWantsMouse = io.WantCaptureMouse;

        ImGui::Render();

        glViewport(0, 0, fbWidth, fbHeight);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &texture);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
