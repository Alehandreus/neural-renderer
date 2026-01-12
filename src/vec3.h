#pragma once

#include <cmath>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Vec3 operator-(Vec3 a) {
    return Vec3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline Vec3 operator*(Vec3 a, float b) {
    return Vec3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline Vec3 operator*(float b, Vec3 a) {
    return a * b;
}

__host__ __device__ inline Vec3 operator/(Vec3 a, float b) {
    return Vec3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ inline Vec3& operator+=(Vec3& a, Vec3 b) {
    a = a + b;
    return a;
}

__host__ __device__ inline Vec3& operator-=(Vec3& a, Vec3 b) {
    a = a - b;
    return a;
}

__host__ __device__ inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float length(Vec3 v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline Vec3 normalize(Vec3 v) {
    float len = length(v);
    if (len > 0.0f) {
        return v / len;
    }
    return Vec3();
}

__host__ __device__ inline Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}
