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
#include <cuda_gl_interop.h>
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

std::string filenameFromPath(const std::string& path) {
    if (path.empty()) {
        return "(none)";
    }
    return std::filesystem::path(path).filename().string();
}

}  // namespace

int main(int argc, char** argv) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return 1;
    }

    // Load configuration early to get the window resolution
    const char* kDefaultConfigPath = "configs/chess.json";
    const char* configPath = (argc > 1) ? argv[1] : kDefaultConfigPath;

    RendererConfig config;
    {
        std::string configError;
        if (!LoadConfigFromFile(configPath, &config, &configError)) {
            std::fprintf(stderr, "Failed to load config: %s\n", configError.c_str());
            glfwTerminate();
            return 1;
        }
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(config.rendering.width, config.rendering.height, "Neural Renderer", nullptr, nullptr);
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

    const int kSamplesPerPixel = 1;

    Scene scene;
    Mesh& originalMesh = scene.originalMesh();
    Mesh& innerShell = scene.innerShell();
    Mesh& outerShell = scene.outerShell();
    Mesh& additionalMesh = scene.additionalMesh();

    std::string innerShellLabel = "(none)";
    std::string outerShellLabel = "(none)";

    if (!config.original_mesh.path.empty() &&
        LoadMeshLabeled(config.original_mesh.path.c_str(), &originalMesh, "original",
                        false, true, config.original_mesh.scale)) {
    }
    if (originalMesh.numTriangles() == 0) {
        GenerateUvSphere(&originalMesh, 48, 96, 1.0f);
    }

    if (!config.inner_shell.path.empty() &&
        LoadMeshLabeled(config.inner_shell.path.c_str(), &innerShell, "inner shell",
                        false, false, config.inner_shell.scale)) {
        innerShellLabel = filenameFromPath(config.inner_shell.path);
    }

    if (!config.outer_shell.path.empty() &&
        LoadMeshLabeled(config.outer_shell.path.c_str(), &outerShell, "outer shell",
                        false, false, config.outer_shell.scale)) {
        outerShellLabel = filenameFromPath(config.outer_shell.path);
    }

    if (!config.additional_mesh.path.empty() &&
        LoadMeshLabeled(config.additional_mesh.path.c_str(), &additionalMesh, "additional mesh",
                        false, true, config.additional_mesh.scale)) {
    }

    std::string envError;
    if (!config.environment.hdri_path.empty() && !scene.environment().loadFromFile(config.environment.hdri_path.c_str(), &envError)) {
        std::fprintf(stderr, "Failed to load HDRI '%s': %s\n", config.environment.hdri_path.c_str(), envError.c_str());
    }
    scene.environment().setRotation(config.environment.rotation);
    scene.environment().setStrength(config.environment.strength);

    // Apply material config to scene
    auto applyMaterialConfig = [&](Material& mat) {
        mat.base_color = MaterialParamVec3::constant(config.material.base_color);
        mat.roughness = MaterialParam::constant(config.material.roughness);
        mat.metallic = MaterialParam::constant(config.material.metallic);
        mat.specular = MaterialParam::constant(config.material.specular);
        mat.specular_tint = MaterialParam::constant(config.material.specular_tint);
        mat.anisotropy = MaterialParam::constant(config.material.anisotropy);
        mat.sheen = MaterialParam::constant(config.material.sheen);
        mat.sheen_tint = MaterialParam::constant(config.material.sheen_tint);
        mat.clearcoat = MaterialParam::constant(config.material.clearcoat);
        mat.clearcoat_gloss = MaterialParam::constant(config.material.clearcoat_gloss);
    };
    // Override only non-texture material params on per-mesh materials (preserve base_color textures)
    auto applyMaterialParamsOnly = [&](Material& mat) {
        mat.roughness = MaterialParam::constant(config.material.roughness);
        mat.metallic = MaterialParam::constant(config.material.metallic);
        mat.specular = MaterialParam::constant(config.material.specular);
        mat.specular_tint = MaterialParam::constant(config.material.specular_tint);
        mat.anisotropy = MaterialParam::constant(config.material.anisotropy);
        mat.sheen = MaterialParam::constant(config.material.sheen);
        mat.sheen_tint = MaterialParam::constant(config.material.sheen_tint);
        mat.clearcoat = MaterialParam::constant(config.material.clearcoat);
        mat.clearcoat_gloss = MaterialParam::constant(config.material.clearcoat_gloss);
    };
    applyMaterialConfig(scene.globalMaterial());
    for (auto& mat : originalMesh.materials_) applyMaterialParamsOnly(mat);
    for (auto& mat : innerShell.materials_) applyMaterialParamsOnly(mat);
    for (auto& mat : outerShell.materials_) applyMaterialParamsOnly(mat);
    for (auto& mat : additionalMesh.materials_) applyMaterialParamsOnly(mat);

    // Set camera from config matrix
    Vec3 configCamPos;
    float configCamYaw, configCamPitch;
    {
        MatrixToCameraState(config.camera.matrix, &configCamPos, &configCamYaw, &configCamPitch);
        configCamPos = configCamPos * config.original_mesh.scale;
        input.camera().position = configCamPos;
        input.camera().yaw = configCamYaw;
        input.camera().pitch = configCamPitch;
        input.camera().fovY = config.camera.yfov;

        // Set camera movement speed
        if (config.camera.move_speed > 0.0f) {
            input.setMoveSpeed(config.camera.move_speed * config.original_mesh.scale);
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
    renderer.setConstantNeuralColor(config.material.use_constant_neural_color, config.material.constant_neural_color);
    bool loaded = false;
    renderer.setUseNeuralQuery(false);
    renderer.setBounceCount(config.rendering.bounce_count);
    renderer.setSamplesPerPixel(kSamplesPerPixel);
    if (!config.checkpoint_path.empty()) {
        loaded = renderer.loadWeightsFromFile(config.checkpoint_path.c_str());
    }
    size_t checkpointFileBytes = 0;
    if (loaded && !config.checkpoint_path.empty()) {
        std::error_code ec;
        auto sz = std::filesystem::file_size(config.checkpoint_path, ec);
        if (!ec) checkpointFileBytes = static_cast<size_t>(sz);
    }
    if (loaded) {
        std::printf("Neural parameters loaded from file (%.2f MB).\n",
                    static_cast<double>(checkpointFileBytes) / (1024.0 * 1024.0));
    } else {
        std::printf("Neural parameters not loaded (using initialization).\n");
    }
    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    renderer.resize(fbWidth, fbHeight);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fbWidth, fbHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    cudaGraphicsResource_t cudaTexResource = nullptr;
    if (fbWidth > 0 && fbHeight > 0) {
        cudaGraphicsGLRegisterImage(&cudaTexResource, texture, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsWriteDiscard);
    }

    double lastTime = glfwGetTime();
    bool lambertView = false;
    bool useNeuralQuery = config.neural_network.use_neural_query;
#ifdef USE_OPTIX
    bool useHardwareRT = true;
#endif
    int bounceCount = config.rendering.bounce_count;
    int samplesPerPixel = kSamplesPerPixel;
    int classicMeshIndex = 0;
    float envmapRotation = config.environment.rotation;
    float envmapStrength = config.environment.strength;
    float lastEnvmapStrength = envmapStrength;
    bool useDirectEnvColor = false;
    float directEnvColor[3] = {0.0f, 0.0f, 0.0f};
    bool useAdditionalMesh = additionalMesh.numTriangles() > 0;
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
#ifdef USE_OPTIX
        renderer.setUseHardwareRT(useHardwareRT);
#endif
        renderer.setBounceCount(bounceCount);
        renderer.setSamplesPerPixel(samplesPerPixel);
        renderer.setClassicMeshIndex(classicMeshIndex);
        renderer.setEnvmapRotation(envmapRotation);
        scene.environment().setStrength(envmapStrength);
        renderer.setDirectEnvColor(useDirectEnvColor, Vec3(directEnvColor[0], directEnvColor[1], directEnvColor[2]));
        renderer.setUseAdditionalMesh(useAdditionalMesh);

        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth != renderer.width() || fbHeight != renderer.height()) {
            if (fbWidth > 0 && fbHeight > 0) {
                renderer.resize(fbWidth, fbHeight);
                if (cudaTexResource) {
                    cudaGraphicsUnregisterResource(cudaTexResource);
                    cudaTexResource = nullptr;
                }
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fbWidth, fbHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                cudaGraphicsGLRegisterImage(&cudaTexResource, texture, GL_TEXTURE_2D,
                                            cudaGraphicsRegisterFlagsWriteDiscard);
            }
        }

        if (fbWidth > 0 && fbHeight > 0) {
            RenderBasis renderBasis;
            renderBasis.forward = basis.forward;
            renderBasis.right = basis.right;
            renderBasis.up = basis.up;
            renderBasis.fovY = camera.fovY;
            renderer.setCameraBasis(renderBasis);
            renderer.render(camera.position);

            cudaArray_t texArray;
            cudaGraphicsMapResources(1, &cudaTexResource);
            cudaGraphicsSubResourceGetMappedArray(&texArray, cudaTexResource, 0, 0);
            cudaMemcpy2DToArray(texArray, 0, 0,
                                renderer.devicePixels(),
                                fbWidth * sizeof(uchar4),
                                fbWidth * sizeof(uchar4),
                                fbHeight,
                                cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &cudaTexResource);
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
        ImGui::Image(ImTextureID(texture), io.DisplaySize);
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
        ImGui::Checkbox("Neural Mode", &useNeuralQuery);
#ifdef USE_OPTIX
        ImGui::Checkbox("Hardware RT (OptiX)", &useHardwareRT);
#endif
        ImGui::Checkbox("Lambert shading", &lambertView);
        ImGui::InputInt("Max bounces", &bounceCount);
        ImGui::InputInt("Samples per pixel", &samplesPerPixel);
        const char* meshNames[] = {"Original", "Inner shell", "Outer shell"};
        ImGui::Combo("Classic mesh", &classicMeshIndex, meshNames, 3);
        if (additionalMesh.numTriangles() > 0) {
            ImGui::Checkbox("Extra mesh (uncompressed)", &useAdditionalMesh);
        }
        ImGui::DragFloat("Envmap rotation", &envmapRotation, 1.0f, 0.0f, 360.0f, "%.1f deg");
        ImGui::InputFloat("Envmap strength", &envmapStrength);
        float fovDeg = camera.fovY * (180.0f / 3.14159265f);
        if (ImGui::SliderFloat("FOV", &fovDeg, 10.0f, 120.0f, "%.1f deg")) {
            input.camera().fovY = fovDeg * (3.14159265f / 180.0f);
        }
        // Reset ray accumulation if envmap strength changed
        if (envmapStrength != lastEnvmapStrength) {
            renderer.resetSamples();
            lastEnvmapStrength = envmapStrength;
        }
        if (ImGui::Checkbox("Direct env color", &useDirectEnvColor)) {
            renderer.resetSamples();
        }
        if (useDirectEnvColor) {
            ImGui::SameLine();
            if (ImGui::ColorEdit3("##directEnvColor", directEnvColor,
                                  ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_Float)) {
                renderer.resetSamples();
            }
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
                std::vector<uchar4> savePixels(static_cast<size_t>(fbWidth) * fbHeight);
                cudaMemcpy(savePixels.data(), renderer.devicePixels(),
                           savePixels.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);
                int success = stbi_write_png(outPath, fbWidth, fbHeight, 4, savePixels.data(), fbWidth * 4);
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
            ImGui::Text("Inner shell: %s", innerShellLabel.c_str());
            {
                const size_t vertBytes = innerShell.vertices_.size() * sizeof(Vec3);
                const size_t faceBytes = innerShell.indices_.size() * sizeof(uint3);
                const size_t bvhBytes = innerShell.bvhStorageBytes();
                const size_t totalBytes = vertBytes + faceBytes + bvhBytes;
                ImGui::Text("  vertices: %d, buffer: %zu bytes (%.2f MB)",
                            innerShell.numVertices(),
                            vertBytes,
                            static_cast<double>(vertBytes) / (1024.0 * 1024.0));
                ImGui::Text("  faces: %d, buffer: %zu bytes (%.2f MB)",
                            innerShell.numTriangles(),
                            faceBytes,
                            static_cast<double>(faceBytes) / (1024.0 * 1024.0));
                ImGui::Text("  BVH nodes: %d, buffer: %zu bytes (%.2f MB)",
                            innerShell.nodeCount(),
                            bvhBytes,
                            static_cast<double>(bvhBytes) / (1024.0 * 1024.0));
                ImGui::Text("  total: %zu bytes (%.2f MB)",
                            totalBytes,
                            static_cast<double>(totalBytes) / (1024.0 * 1024.0));
            }
            ImGui::Text("Outer shell: %s", outerShellLabel.c_str());
            {
                const size_t vertBytes = outerShell.vertices_.size() * sizeof(Vec3);
                const size_t faceBytes = outerShell.indices_.size() * sizeof(uint3);
                const size_t bvhBytes = outerShell.bvhStorageBytes();
                const size_t totalBytes = vertBytes + faceBytes + bvhBytes;
                ImGui::Text("  vertices: %d, buffer: %zu bytes (%.2f MB)",
                            outerShell.numVertices(),
                            vertBytes,
                            static_cast<double>(vertBytes) / (1024.0 * 1024.0));
                ImGui::Text("  faces: %d, buffer: %zu bytes (%.2f MB)",
                            outerShell.numTriangles(),
                            faceBytes,
                            static_cast<double>(faceBytes) / (1024.0 * 1024.0));
                ImGui::Text("  BVH nodes: %d, buffer: %zu bytes (%.2f MB)",
                            outerShell.nodeCount(),
                            bvhBytes,
                            static_cast<double>(bvhBytes) / (1024.0 * 1024.0));
                ImGui::Text("  total: %zu bytes (%.2f MB)",
                            totalBytes,
                            static_cast<double>(totalBytes) / (1024.0 * 1024.0));
            }
            ImGui::Text("Network (checkpoint): %.2f MB",
                        static_cast<double>(checkpointFileBytes) / (1024.0 * 1024.0));
            auto meshBytes = [](const Mesh& m) -> size_t {
                return m.vertices_.size() * sizeof(Vec3)
                     + m.indices_.size() * sizeof(uint3)
                     + m.bvhStorageBytes();
            };
            const size_t totalAllBytes = checkpointFileBytes
                + meshBytes(innerShell) + meshBytes(outerShell);
            ImGui::Separator();
            ImGui::Text("Total (network + shells): %.2f MB",
                        static_cast<double>(totalAllBytes) / (1024.0 * 1024.0));
            ImGui::TreePop();
        }
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::End();

#ifdef PROFILE_KERNELS
        {
            const KernelTimings& kt = renderer.lastFrameTimings();
            const double rayCount = static_cast<double>(kt.rayCount);

            ImGuiWindowFlags profFlags =
                ImGuiWindowFlags_NoDecoration        |
                ImGuiWindowFlags_AlwaysAutoResize    |
                ImGuiWindowFlags_NoSavedSettings     |
                ImGuiWindowFlags_NoFocusOnAppearing  |
                ImGuiWindowFlags_NoNav;
            // Anchor top-right corner of this window at (displayWidth - 10, 10)
            ImGui::SetNextWindowPos(
                ImVec2(io.DisplaySize.x - 10.0f, 10.0f),
                ImGuiCond_Always,
                ImVec2(1.0f, 0.0f));
            ImGui::Begin("GPU Kernel Timings", nullptr, profFlags);
            ImGui::Text("GPU Kernel Timings (last frame)");
            ImGui::Separator();

            if (ImGui::BeginTable("kerneltbl", 3,
                    ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Kernel",
                    ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("ms",
                    ImGuiTableColumnFlags_WidthFixed, 55.0f);
                ImGui::TableSetupColumn("ns/ray",
                    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                ImGui::TableHeadersRow();

                for (int i = 0; i < KID_COUNT; ++i) {
                    const double ms = kt.ms[i];
                    if (ms < 0.001) continue;
                    const double nsPerRay = rayCount > 0.0
                        ? ms * 1.0e6 / rayCount : 0.0;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(KernelTimings::name(i));
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2f", ms);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.1f", nsPerRay);
                }

                // Total row
                {
                    const double totalMs = kt.totalMs();
                    const double totalNsPerRay = rayCount > 0.0
                        ? totalMs * 1.0e6 / rayCount : 0.0;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted("Total");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2f", totalMs);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.1f", totalNsPerRay);
                }

                ImGui::EndTable();
            }

            ImGui::Text("Rays: %d  (%.2f Mrays)", kt.rayCount,
                        static_cast<double>(kt.rayCount) / 1.0e6);
            if (kt.neuralRayCalls > 0 && kt.rayCount > 0) {
                ImGui::Text("Avg neural calls/ray: %.2f",
                            static_cast<double>(kt.neuralRayCalls) /
                            static_cast<double>(kt.rayCount));
            }
            ImGui::End();
        }
#endif  // PROFILE_KERNELS

        uiWantsMouse = io.WantCaptureMouse;

        ImGui::Render();

        glViewport(0, 0, fbWidth, fbHeight);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    if (cudaTexResource) {
        cudaGraphicsUnregisterResource(cudaTexResource);
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
