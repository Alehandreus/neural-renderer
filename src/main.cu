#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <nfd.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "stb_image_write.h"

#include "config_loader.h"
#include "cuda_renderer_neural.h"
#include "input_controller.h"
#include "mesh_loader.h"
#include "scene.h"

namespace {

void glfwErrorCallback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

bool loadMesh(const char* path, Mesh* mesh, const char* label, bool normalize, bool nearestTex, float scale = 1.0f) {
    if (!path || path[0] == '\0') return false;
    std::string loadError;
    std::filesystem::path meshPath(path);
    std::string ext = meshPath.extension().string();
    for (char& ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    bool loaded = false;
    if (ext == ".gltf" || ext == ".glb") {
        loaded = LoadTexturedGltfFromFile(path, mesh, &loadError, normalize, nearestTex, scale);
    } else {
        loaded = LoadMeshFromFile(path, mesh, &loadError, normalize, scale);
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

    if (NFD_Init() != NFD_OKAY) {
        std::fprintf(stderr, "Failed to initialize NFD: %s\n", NFD_GetError());
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    InputController input(window);

    // Load configuration
    const char* kDefaultConfigPath = "configs/chess.json";
    const char* configPath = (argc > 1) ? argv[1] : kDefaultConfigPath;

    RendererConfig config;
    std::string configError;
    if (!LoadConfigFromFile(configPath, &config, &configError)) {
        std::fprintf(stderr, "Failed to load config: %s\n", configError.c_str());
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const int kBounceCount = 3;
    const int kSamplesPerPixel = 1;

    Scene scene;
    Mesh& originalMesh = scene.originalMesh();
    Mesh& innerShell = scene.innerShell();
    Mesh& outerShell = scene.outerShell();
    Mesh& additionalMesh = scene.additionalMesh();

    std::string originalMeshLabel = "procedural sphere";
    std::string innerShellLabel = "(none)";
    std::string outerShellLabel = "(none)";
    std::string additionalMeshLabel = "(none)";

    if (!config.original_mesh.path.empty() && loadMesh(config.original_mesh.path.c_str(), &originalMesh, "original",
                                                        config.rendering.normalize_meshes,
                                                        config.rendering.nearest_texture_sampling,
                                                        config.original_mesh.scale)) {
        originalMeshLabel = config.original_mesh.path;
    }
    if (originalMesh.triangleCount() == 0) {
        GenerateUvSphere(&originalMesh, 48, 96, 1.0f);
    }
    originalMesh.setUseTextureColor(config.original_mesh.use_texture_color);

    if (!config.inner_shell.path.empty() && loadMesh(config.inner_shell.path.c_str(), &innerShell, "inner shell",
                                                      config.rendering.normalize_meshes, false,
                                                      config.inner_shell.scale)) {
        innerShellLabel = config.inner_shell.path;
    }
    innerShell.setUseTextureColor(config.inner_shell.use_texture_color);

    if (!config.outer_shell.path.empty() && loadMesh(config.outer_shell.path.c_str(), &outerShell, "outer shell",
                                                      config.rendering.normalize_meshes, false,
                                                      config.outer_shell.scale)) {
        outerShellLabel = config.outer_shell.path;
    }
    outerShell.setUseTextureColor(config.outer_shell.use_texture_color);

    if (!config.additional_mesh.path.empty() && loadMesh(config.additional_mesh.path.c_str(), &additionalMesh, "additional mesh",
                                                          config.rendering.normalize_meshes,
                                                          config.rendering.nearest_texture_sampling,
                                                          config.additional_mesh.scale)) {
        additionalMeshLabel = config.additional_mesh.path;
    }
    additionalMesh.setUseTextureColor(config.additional_mesh.use_texture_color);

    std::string envError;
    if (!config.environment.hdri_path.empty() && !scene.environment().loadFromFile(config.environment.hdri_path.c_str(), &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", config.environment.hdri_path.c_str(), envError.c_str());
    }
    scene.environment().setRotation(config.environment.rotation);
    scene.environment().setStrength(config.environment.strength);

    // Apply material config to scene
    scene.material().base_color = config.material.base_color;
    scene.material().roughness = config.material.roughness;
    scene.material().metallic = config.material.metallic;
    scene.material().specular = config.material.specular;
    scene.material().specular_tint = config.material.specular_tint;
    scene.material().anisotropy = config.material.anisotropy;
    scene.material().sheen = config.material.sheen;
    scene.material().sheen_tint = config.material.sheen_tint;
    scene.material().clearcoat = config.material.clearcoat;
    scene.material().clearcoat_gloss = config.material.clearcoat_gloss;

    // Set camera from config matrix
    Vec3 configCamPos;
    float configCamYaw, configCamPitch;
    {
        MatrixToCameraState(config.camera.matrix, &configCamPos, &configCamYaw, &configCamPitch);
        input.camera().position = configCamPos;
        input.camera().yaw = configCamYaw;
        input.camera().pitch = configCamPitch;
        input.camera().fovY = config.camera.yfov;

        // Set camera movement speed
        if (config.camera.move_speed > 0.0f) {
            input.setMoveSpeed(config.camera.move_speed);
        } else {
            // Auto-calculate based on mesh bounds if move_speed not specified
            Vec3 bmin = originalMesh.boundsMin();
            Vec3 bmax = originalMesh.boundsMax();
            Vec3 ext(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);
            float diagonal = std::sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);
            if (diagonal > 0.0f) {
                input.setMoveSpeed(diagonal * 0.15f);
            }
        }
    }

    RendererNeural renderer(scene, &config.neural_network);
    bool loaded = false;
    renderer.setUseNeuralQuery(false);
    renderer.setBounceCount(kBounceCount);
    renderer.setSamplesPerPixel(kSamplesPerPixel);
    if (!config.checkpoint_path.empty()) {
        loaded = renderer.loadWeightsFromFile(config.checkpoint_path.c_str());
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
    bool useNeuralQuery = config.neural_network.use_neural_query;
    int bounceCount = kBounceCount;
    int samplesPerPixel = kSamplesPerPixel;
    int classicMeshIndex = 0;
    float envmapRotation = config.environment.rotation;
    float envmapStrength = config.environment.strength;
    float lastEnvmapStrength = envmapStrength;
    bool uiWantsMouse = false;
    Material lastMaterial = scene.material();

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
        renderer.setEnvmapRotation(envmapRotation);
        scene.environment().setStrength(envmapStrength);

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
        ImGui::DragFloat("Envmap rotation", &envmapRotation, 1.0f, 0.0f, 360.0f, "%.1f deg");
        ImGui::InputFloat("Envmap strength", &envmapStrength);
        float fovDeg = camera.fovY * (180.0f / 3.14159265f);
        if (ImGui::SliderFloat("FOV", &fovDeg, 10.0f, 120.0f, "%.1f deg")) {
            input.camera().fovY = fovDeg * (3.14159265f / 180.0f);
        }
        if (ImGui::TreeNode("Material properties")) {
            Material& mat = scene.material();
            ImGui::ColorEdit3("Base color", &mat.base_color.x);
            ImGui::SliderFloat("Roughness", &mat.roughness, 0.0f, 1.0f);
            ImGui::SliderFloat("Metallic", &mat.metallic, 0.0f, 1.0f);
            ImGui::SliderFloat("Specular", &mat.specular, 0.0f, 1.0f);
            ImGui::SliderFloat("Specular tint", &mat.specular_tint, 0.0f, 1.0f);
            ImGui::SliderFloat("Anisotropy", &mat.anisotropy, 0.0f, 1.0f);
            ImGui::SliderFloat("Sheen", &mat.sheen, 0.0f, 1.0f);
            ImGui::SliderFloat("Sheen tint", &mat.sheen_tint, 0.0f, 1.0f);
            ImGui::SliderFloat("Clearcoat", &mat.clearcoat, 0.0f, 1.0f);
            ImGui::SliderFloat("Clearcoat gloss", &mat.clearcoat_gloss, 0.0f, 1.0f);
            ImGui::TreePop();
        }
        // Reset ray accumulation if material properties changed
        {
            Material& mat = scene.material();
            if (mat.base_color.x != lastMaterial.base_color.x ||
                mat.base_color.y != lastMaterial.base_color.y ||
                mat.base_color.z != lastMaterial.base_color.z ||
                mat.roughness != lastMaterial.roughness ||
                mat.metallic != lastMaterial.metallic ||
                mat.specular != lastMaterial.specular ||
                mat.specular_tint != lastMaterial.specular_tint ||
                mat.anisotropy != lastMaterial.anisotropy ||
                mat.sheen != lastMaterial.sheen ||
                mat.sheen_tint != lastMaterial.sheen_tint ||
                mat.clearcoat != lastMaterial.clearcoat ||
                mat.clearcoat_gloss != lastMaterial.clearcoat_gloss) {
                renderer.resetSamples();
                lastMaterial = mat;
            }
        }
        // Reset ray accumulation if envmap strength changed
        if (envmapStrength != lastEnvmapStrength) {
            renderer.resetSamples();
            lastEnvmapStrength = envmapStrength;
        }
        if (ImGui::TreeNode("Camera matrix")) {
            float matrix[16];
            CameraStateToMatrix(camera.position, camera.yaw, camera.pitch, matrix);
            ImGui::Text("pos: %.3f, %.3f, %.3f", camera.position.x, camera.position.y, camera.position.z);
            ImGui::Text("yaw: %.3f, pitch: %.3f", camera.yaw, camera.pitch);
            for (int row = 0; row < 4; row++) {
                ImGui::Text("[%6.3f %6.3f %6.3f %6.3f]",
                    matrix[row], matrix[4 + row], matrix[8 + row], matrix[12 + row]);
            }
            if (ImGui::Button("Reset to config")) {
                input.camera().position = configCamPos;
                input.camera().yaw = configCamYaw;
                input.camera().pitch = configCamPitch;
                renderer.resetSamples();
            }
            ImGui::TreePop();
        }
        if (ImGui::Button("Export camera to JSON")) {
            nfdchar_t* outPath = nullptr;
            nfdfilteritem_t filters[1] = {{"JSON", "json"}};
            nfdresult_t result = NFD_SaveDialog(&outPath, filters, 1, nullptr, "camera.json");
            if (result == NFD_OKAY) {
                FILE* f = std::fopen(outPath, "w");
                if (f) {
                    // Convert camera state to matrix format
                    float matrix[16];
                    CameraStateToMatrix(camera.position, camera.yaw, camera.pitch, matrix);

                    std::fprintf(f, "{\n");
                    std::fprintf(f, "  \"camera\": {\n");
                    std::fprintf(f, "    \"matrix\": [\n");
                    for (int i = 0; i < 4; i++) {
                        std::fprintf(f, "      %.10f,\n", matrix[i * 4 + 0]);
                        std::fprintf(f, "      %.10f,\n", matrix[i * 4 + 1]);
                        std::fprintf(f, "      %.10f,\n", matrix[i * 4 + 2]);
                        std::fprintf(f, "      %.10f%s\n", matrix[i * 4 + 3], (i < 3) ? "," : "");
                    }
                    std::fprintf(f, "    ],\n");
                    std::fprintf(f, "    \"yfov\": %.10f\n", camera.fovY);
                    std::fprintf(f, "  }\n");
                    std::fprintf(f, "}\n");
                    std::fclose(f);
                    std::printf("Camera exported to %s\n", outPath);
                } else {
                    std::fprintf(stderr, "Failed to open %s for writing\n", outPath);
                }
                NFD_FreePath(outPath);
            } else if (result == NFD_CANCEL) {
                std::printf("Camera export cancelled\n");
            } else {
                std::fprintf(stderr, "Error opening save dialog: %s\n", NFD_GetError());
            }
        }
        if (ImGui::Button("Save rendering to image")) {
            nfdchar_t* outPath = nullptr;
            nfdfilteritem_t filters[1] = {{"PNG Image", "png"}};
            nfdresult_t result = NFD_SaveDialog(&outPath, filters, 1, nullptr, "render.png");
            if (result == NFD_OKAY) {
                // Save the current rendering as PNG
                int success = stbi_write_png(outPath, fbWidth, fbHeight, 4, hostPixels.data(), fbWidth * 4);
                if (success) {
                    std::printf("Image saved to %s\n", outPath);
                } else {
                    std::fprintf(stderr, "Failed to save image to %s\n", outPath);
                }
                NFD_FreePath(outPath);
            } else if (result == NFD_CANCEL) {
                std::printf("Image save cancelled\n");
            } else {
                std::fprintf(stderr, "Error opening save dialog: %s\n", NFD_GetError());
            }
        }
        ImGui::Text("Resolution: %d x %d", fbWidth, fbHeight);
        ImGui::Separator();
        if (ImGui::TreeNode("Mesh & BVH Info")) {
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
            ImGui::Text("Additional: %s", additionalMeshLabel.c_str());
            if (additionalMesh.triangleCount() > 0) {
                ImGui::Text("  triangles: %d, BVH: %d (%.2f MB)",
                            additionalMesh.triangleCount(),
                            additionalMesh.nodeCount(),
                            static_cast<double>(additionalMesh.bvhStorageBytes()) / (1024.0 * 1024.0));
            }
            ImGui::Text("Network params (fp16): %.2f MB",
                        static_cast<double>(renderer.paramsBytes()) / (1024.0 * 1024.0));
            ImGui::TreePop();
        }
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

    NFD_Quit();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
