#pragma once

#include "mesh.h"
#include "vec3.h"

__device__ inline float length2(Vec3 v) {
    return dot(v, v);
}

__device__ inline float distance2PointAabb(const Vec3& p, const Vec3& boundsMin, const Vec3& boundsMax) {
    float dx = 0.0f;
    if (p.x < boundsMin.x) {
        dx = boundsMin.x - p.x;
    } else if (p.x > boundsMax.x) {
        dx = p.x - boundsMax.x;
    }
    float dy = 0.0f;
    if (p.y < boundsMin.y) {
        dy = boundsMin.y - p.y;
    } else if (p.y > boundsMax.y) {
        dy = p.y - boundsMax.y;
    }
    float dz = 0.0f;
    if (p.z < boundsMin.z) {
        dz = boundsMin.z - p.z;
    } else if (p.z > boundsMax.z) {
        dz = p.z - boundsMax.z;
    }
    return dx * dx + dy * dy + dz * dz;
}

__device__ inline Vec3 closestPointOnTriangle(const Vec3& p, const Triangle& tri) {
    const Vec3 a = tri.v0;
    const Vec3 b = tri.v1;
    const Vec3 c = tri.v2;
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return a;
    }

    const Vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        return b;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    const Vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        return c;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + ac * w;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

__device__ inline Vec3 closestPointOnMesh(const Vec3& p, const MeshDeviceView& mesh) {
    if (mesh.nodeCount <= 0 || mesh.triangleCount <= 0) {
        return p;
    }

    Vec3 bestPoint = p;
    float bestDist2 = 1e30f;

    constexpr int kMaxStack = 128;
    int stack[kMaxStack];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        if (nodeIndex < 0 || nodeIndex >= mesh.nodeCount) {
            continue;
        }

        const BvhNode node = mesh.nodes[nodeIndex];
        float nodeDist2 = distance2PointAabb(p, node.boundsMin, node.boundsMax);
        if (nodeDist2 > bestDist2) {
            continue;
        }

        if (node.isLeaf) {
            int start = node.first;
            int end = start + node.count;
            for (int i = start; i < end; ++i) {
                Vec3 candidate = closestPointOnTriangle(p, mesh.triangles[i]);
                float d2 = length2(candidate - p);
                if (d2 < bestDist2) {
                    bestDist2 = d2;
                    bestPoint = candidate;
                }
            }
        } else {
            int left = node.left;
            int right = node.right;
            float leftDist2 = 1e30f;
            float rightDist2 = 1e30f;
            if (left >= 0 && left < mesh.nodeCount) {
                const BvhNode leftNode = mesh.nodes[left];
                leftDist2 = distance2PointAabb(p, leftNode.boundsMin, leftNode.boundsMax);
            }
            if (right >= 0 && right < mesh.nodeCount) {
                const BvhNode rightNode = mesh.nodes[right];
                rightDist2 = distance2PointAabb(p, rightNode.boundsMin, rightNode.boundsMax);
            }

            int first = left;
            int second = right;
            float firstDist2 = leftDist2;
            float secondDist2 = rightDist2;
            if (rightDist2 < leftDist2) {
                first = right;
                second = left;
                firstDist2 = rightDist2;
                secondDist2 = leftDist2;
            }
            if (secondDist2 <= bestDist2 && stackSize < kMaxStack) {
                stack[stackSize++] = second;
            }
            if (firstDist2 <= bestDist2 && stackSize < kMaxStack) {
                stack[stackSize++] = first;
            }
        }
    }

    return bestPoint;
}
