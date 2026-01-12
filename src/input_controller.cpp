#include "input_controller.h"

#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kMouseSensitivity = 0.1f;
constexpr float kMaxDelta = 250.0f;
constexpr float kMaxPitch = 89.0f;

CameraState initialCamera() {
    CameraState camera{};
    camera.position = Vec3(2.0f, 1.2f, 2.0f);
    camera.fovY = 60.0f * kDegToRad;

    Vec3 target(0.0f, 0.0f, 0.0f);
    Vec3 forward = normalize(target - camera.position);
    camera.yaw = atan2f(forward.z, forward.x) / kDegToRad;
    camera.pitch = asinf(forward.y) / kDegToRad;
    return camera;
}

}  // namespace

InputController::InputController(GLFWwindow* window) : window_(window) {
    camera_ = initialCamera();

    if (window_) {
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        resetMouseState(3);
    }

    updateBasis();
}

void InputController::update(float dt, bool uiWantsMouse) {
    if (!window_) {
        return;
    }

    glfwPollEvents();

    bool escDown = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if (escDown && !escWasDown_) {
        mouseCaptured_ = false;
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    escWasDown_ = escDown;

    bool clickDown = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (clickDown && !clickWasDown_ && !mouseCaptured_ && !uiWantsMouse) {
        mouseCaptured_ = true;
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        resetMouseState(2);
    }
    clickWasDown_ = clickDown;

    updateMouse();
    updateBasis();
    updateKeyboard(dt);
}

void InputController::resetMouseState(int ignoreFrames) {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window_, &width, &height);
    if (width > 0 && height > 0) {
        glfwSetCursorPos(window_, static_cast<double>(width) * 0.5, static_cast<double>(height) * 0.5);
    }
    mouseState_.hasLast = false;
    mouseState_.ignoreFrames = ignoreFrames;
}

void InputController::updateMouse() {
    if (!mouseCaptured_) {
        mouseState_.hasLast = false;
        mouseState_.ignoreFrames = 0;
        return;
    }

    double xpos = 0.0;
    double ypos = 0.0;
    glfwGetCursorPos(window_, &xpos, &ypos);

    if (!mouseState_.hasLast) {
        mouseState_.lastX = xpos;
        mouseState_.lastY = ypos;
        mouseState_.hasLast = true;
        return;
    }

    float xoffset = static_cast<float>(xpos - mouseState_.lastX);
    float yoffset = static_cast<float>(mouseState_.lastY - ypos);
    mouseState_.lastX = xpos;
    mouseState_.lastY = ypos;

    if (mouseState_.ignoreFrames > 0) {
        if (xoffset != 0.0f || yoffset != 0.0f) {
            --mouseState_.ignoreFrames;
        }
        return;
    }

    if (fabsf(xoffset) > kMaxDelta || fabsf(yoffset) > kMaxDelta) {
        return;
    }

    camera_.yaw += xoffset * kMouseSensitivity;
    camera_.pitch += yoffset * kMouseSensitivity;

    if (camera_.pitch > kMaxPitch) {
        camera_.pitch = kMaxPitch;
    } else if (camera_.pitch < -kMaxPitch) {
        camera_.pitch = -kMaxPitch;
    }
}

void InputController::updateKeyboard(float dt) {
    float speed = 3.5f;
    if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        speed *= 2.0f;
    }

    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    float delta = speed * dt;
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) {
        camera_.position += basis_.forward * delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
        camera_.position -= basis_.forward * delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
        camera_.position -= basis_.right * delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
        camera_.position += basis_.right * delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera_.position += worldUp * delta;
    }
    if (glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
        camera_.position -= worldUp * delta;
    }
}

void InputController::updateBasis() {
    float yawRad = camera_.yaw * kDegToRad;
    float pitchRad = camera_.pitch * kDegToRad;
    Vec3 forward(
            cosf(pitchRad) * cosf(yawRad),
            sinf(pitchRad),
            cosf(pitchRad) * sinf(yawRad));
    forward = normalize(forward);

    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    Vec3 right = cross(forward, worldUp);
    float rLen = length(right);
    if (rLen < 1e-4f) {
        right = Vec3(1.0f, 0.0f, 0.0f);
    } else {
        right = right / rLen;
    }
    Vec3 up = normalize(cross(right, forward));

    basis_.forward = forward;
    basis_.right = right;
    basis_.up = up;
}
