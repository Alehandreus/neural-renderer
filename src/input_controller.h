#pragma once

#include <GLFW/glfw3.h>

#include "vec3.h"

struct CameraState {
    Vec3 position;
    float yaw;
    float pitch;
    float fovY;
};

struct CameraBasis {
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

class InputController {
 public:
    explicit InputController(GLFWwindow* window);

    void update(float dt, bool uiWantsMouse);

    CameraState& camera() { return camera_; }
    const CameraState& camera() const { return camera_; }
    const CameraBasis& basis() const { return basis_; }

    void setMoveSpeed(float speed) { moveSpeed_ = speed; }

 private:
    struct MouseLookState {
        double lastX = 0.0;
        double lastY = 0.0;
        bool hasLast = false;
        int ignoreFrames = 0;
    };

    void resetMouseState(int ignoreFrames);
    void updateMouse();
    void updateKeyboard(float dt);
    void updateBasis();

    GLFWwindow* window_ = nullptr;
    CameraState camera_{};
    CameraBasis basis_{};
    MouseLookState mouseState_{};
    float moveSpeed_ = 30000.0f;
    bool mouseCaptured_ = true;
    bool escWasDown_ = false;
    bool clickWasDown_ = false;
};
