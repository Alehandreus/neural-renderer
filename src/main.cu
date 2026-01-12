#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

    const char* kExactMeshPath = "/home/me/Downloads/chess.fbx";
    const char* kRoughMeshPath = "/home/me/brain/mesh-mapping/models/dragon2_outer_3000.fbx";
    const char* kCheckpointPath = "/home/me/brain/mesh-mapping/inner_params.bin";
    const int kBounceCount = 3;
    const bool kNormalizeMeshes = true;
    const char* kDefaultHdriPath = "/home/me/Downloads/lilienstein.jpg";

    Scene scene;
    Mesh& exactMesh = scene.exactMesh();
    Mesh& roughMesh = scene.roughMesh();
    std::string exactMeshLabel = "procedural sphere";
    std::string roughMeshLabel = "procedural sphere";
    if (kExactMeshPath && kExactMeshPath[0] != '\0') {
        std::string loadError;
        if (LoadMeshFromFile(kExactMeshPath, &exactMesh, &loadError, kNormalizeMeshes)) {
            exactMeshLabel = kExactMeshPath;
        } else {
            std::fprintf(stderr, "Failed to load exact mesh '%s': %s\n", kExactMeshPath, loadError.c_str());
        }
    }
    if (exactMesh.triangleCount() == 0) {
        GenerateUvSphere(&exactMesh, 48, 96, 1.0f);
    }
    if (kRoughMeshPath && kRoughMeshPath[0] != '\0') {
        std::string loadError;
        if (LoadMeshFromFile(kRoughMeshPath, &roughMesh, &loadError, kNormalizeMeshes)) {
            roughMeshLabel = kRoughMeshPath;
        } else {
            std::fprintf(stderr, "Failed to load rough mesh '%s': %s\n", kRoughMeshPath, loadError.c_str());
        }
    }
    if (roughMesh.triangleCount() == 0) {
        GenerateUvSphere(&roughMesh, 48, 96, 1.0f);
    }

    std::string envError;
    if (!scene.environment().loadFromFile(kDefaultHdriPath, &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", kDefaultHdriPath, envError.c_str());
    }

    RendererNeural renderer(scene);
    bool loaded = false;
    renderer.setUseNeuralQuery(false);
    renderer.setBounceCount(kBounceCount);
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
    bool key1WasDown = false;
    bool key2WasDown = false;
    bool key3WasDown = false;
    bool showLossView = false;
    bool lambertView = false;
    bool useNeuralQuery = false;
    int bounceCount = kBounceCount;
    bool uiWantsMouse = false;

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - lastTime);
        lastTime = now;

        input.update(dt, uiWantsMouse);

        const CameraState& camera = input.camera();
        const CameraBasis& basis = input.basis();
        bool key1Down = glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS;
        bool key2Down = glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS;
        bool key3Down = glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS;
        if (key1Down && !key1WasDown) {
            renderer.setGradientMode(RendererNeural::GradientMode::InputOnly);
        }
        if (key2Down && !key2WasDown) {
            renderer.setGradientMode(RendererNeural::GradientMode::WeightsOnly);
        }
        if (key3Down && !key3WasDown) {
            renderer.setGradientMode(RendererNeural::GradientMode::InputAndWeights);
        }
        key1WasDown = key1Down;
        key2WasDown = key2Down;
        key3WasDown = key3Down;
        if (bounceCount < 0) {
            bounceCount = 0;
        }
        renderer.setLossView(showLossView);
        renderer.setLambertView(lambertView);
        renderer.setUseNeuralQuery(useNeuralQuery);
        renderer.setBounceCount(bounceCount);

        double cycle = std::fmod(now, 5.0);
        double ramp = (cycle < 2.5) ? (cycle / 2.5) : ((5.0 - cycle) / 2.5);
        int steps = static_cast<int>(std::round(ramp * 10.0));
        // renderer.setGdSteps(steps);
        renderer.setGdSteps(0);

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
        ImGui::Checkbox("Loss view", &showLossView);
        ImGui::Checkbox("Lambert (no bounces)", &lambertView);
        ImGui::Text("Avg loss: %.6f", renderer.averageLoss());
        ImGui::Text("GD steps: %d", renderer.gdSteps());
        ImGui::InputInt("Max bounces", &bounceCount);
        ImGui::Text("Resolution: %d x %d", fbWidth, fbHeight);
        ImGui::Text("Active mesh: %s", useNeuralQuery ? "rough" : "exact");
        ImGui::Text("Exact mesh: %s", exactMeshLabel.c_str());
        ImGui::Text("Exact triangles: %d", exactMesh.triangleCount());
        ImGui::Text("Exact BVH nodes: %d (%.2f MB)",
                    exactMesh.nodeCount(),
                    static_cast<double>(exactMesh.bvhStorageBytes()) / (1024.0 * 1024.0));
        ImGui::Text("Rough mesh: %s", roughMeshLabel.c_str());
        ImGui::Text("Rough triangles: %d", roughMesh.triangleCount());
        ImGui::Text("Rough BVH nodes: %d (%.2f MB)",
                    roughMesh.nodeCount(),
                    static_cast<double>(roughMesh.bvhStorageBytes()) / (1024.0 * 1024.0));
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
