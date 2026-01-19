#pragma once

__global__ void debugLowResRayKernel(float* inputs,
                                     float* hitPositions,
                                     float* hitNormals,
                                     int* hitFlags,
                                     RenderParams params,
                                     MeshDeviceView mesh,
                                     int stride,
                                     int useTwoHits,
                                     float* altInputs,
                                     float* altPositions,
                                     float* altNormals,
                                     int* altFlags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    int strideValue = stride > 0 ? stride : 1;
    bool activeSample = (x % strideValue == 0) && (y % strideValue == 0);

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        int sampleIdx = pixelIdx + s * params.pixelCount;
        int base = sampleIdx * 3;
        if (!activeSample || s != 0) {
            inputs[base + 0] = 0.0f;
            inputs[base + 1] = 0.0f;
            inputs[base + 2] = 0.0f;
            hitPositions[base + 0] = 0.0f;
            hitPositions[base + 1] = 0.0f;
            hitPositions[base + 2] = 0.0f;
            hitNormals[base + 0] = 0.0f;
            hitNormals[base + 1] = 0.0f;
            hitNormals[base + 2] = 0.0f;
            hitFlags[sampleIdx] = 0;
            if (altInputs && altPositions && altNormals && altFlags) {
                altInputs[base + 0] = 0.0f;
                altInputs[base + 1] = 0.0f;
                altInputs[base + 2] = 0.0f;
                altPositions[base + 0] = 0.0f;
                altPositions[base + 1] = 0.0f;
                altPositions[base + 2] = 0.0f;
                altNormals[base + 0] = 0.0f;
                altNormals[base + 1] = 0.0f;
                altNormals[base + 2] = 0.0f;
                altFlags[sampleIdx] = 0;
            }
            continue;
        }

        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray ray = generatePrimaryRay(x, y, params, rng);
        if (useTwoHits) {
            HitInfo hit1{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
            HitInfo hit2{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
            int hitCount = traceMeshTwoHits(ray, mesh, &hit1, &hit2);
            if (hitCount > 0) {
                Vec3 hitPos = ray.at(hit1.distance);
                Vec3 local = hitPos - params.meshMin;
                Vec3 normalized(
                        local.x * params.meshInvExtent.x,
                        local.y * params.meshInvExtent.y,
                        local.z * params.meshInvExtent.z);
                inputs[base + 0] = normalized.x;
                inputs[base + 1] = normalized.y;
                inputs[base + 2] = normalized.z;
                hitPositions[base + 0] = hitPos.x;
                hitPositions[base + 1] = hitPos.y;
                hitPositions[base + 2] = hitPos.z;
                hitNormals[base + 0] = hit1.normal.x;
                hitNormals[base + 1] = hit1.normal.y;
                hitNormals[base + 2] = hit1.normal.z;
                hitFlags[sampleIdx] = 1;
            } else {
                inputs[base + 0] = 0.0f;
                inputs[base + 1] = 0.0f;
                inputs[base + 2] = 0.0f;
                hitPositions[base + 0] = 0.0f;
                hitPositions[base + 1] = 0.0f;
                hitPositions[base + 2] = 0.0f;
                hitNormals[base + 0] = 0.0f;
                hitNormals[base + 1] = 0.0f;
                hitNormals[base + 2] = 0.0f;
                hitFlags[sampleIdx] = 0;
            }

            if (altInputs && altPositions && altNormals && altFlags) {
                if (hitCount > 1) {
                    const float kEps = 1e-3f;
                    Vec3 ray2Origin = ray.origin + ray.direction * (hit1.distance + kEps);
                    Vec3 hitPos2 = ray2Origin + ray.direction * hit2.distance;
                    Vec3 local2 = hitPos2 - params.meshMin;
                    Vec3 normalized2(
                            local2.x * params.meshInvExtent.x,
                            local2.y * params.meshInvExtent.y,
                            local2.z * params.meshInvExtent.z);
                    altInputs[base + 0] = normalized2.x;
                    altInputs[base + 1] = normalized2.y;
                    altInputs[base + 2] = normalized2.z;
                    altPositions[base + 0] = hitPos2.x;
                    altPositions[base + 1] = hitPos2.y;
                    altPositions[base + 2] = hitPos2.z;
                    altNormals[base + 0] = hit2.normal.x;
                    altNormals[base + 1] = hit2.normal.y;
                    altNormals[base + 2] = hit2.normal.z;
                    altFlags[sampleIdx] = 1;
                } else {
                    altInputs[base + 0] = 0.0f;
                    altInputs[base + 1] = 0.0f;
                    altInputs[base + 2] = 0.0f;
                    altPositions[base + 0] = 0.0f;
                    altPositions[base + 1] = 0.0f;
                    altPositions[base + 2] = 0.0f;
                    altNormals[base + 0] = 0.0f;
                    altNormals[base + 1] = 0.0f;
                    altNormals[base + 2] = 0.0f;
                    altFlags[sampleIdx] = 0;
                }
            }
        } else {
            HitInfo hitInfo{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
            bool hit = traceMesh(ray, mesh, &hitInfo);
            if (hit) {
                Vec3 hitPos = ray.at(hitInfo.distance);
                Vec3 local = hitPos - params.meshMin;
                Vec3 normalized(
                        local.x * params.meshInvExtent.x,
                        local.y * params.meshInvExtent.y,
                        local.z * params.meshInvExtent.z);
                inputs[base + 0] = normalized.x;
                inputs[base + 1] = normalized.y;
                inputs[base + 2] = normalized.z;
                hitPositions[base + 0] = hitPos.x;
                hitPositions[base + 1] = hitPos.y;
                hitPositions[base + 2] = hitPos.z;
                hitNormals[base + 0] = hitInfo.normal.x;
                hitNormals[base + 1] = hitInfo.normal.y;
                hitNormals[base + 2] = hitInfo.normal.z;
                hitFlags[sampleIdx] = 1;
            } else {
                inputs[base + 0] = 0.0f;
                inputs[base + 1] = 0.0f;
                inputs[base + 2] = 0.0f;
                hitPositions[base + 0] = 0.0f;
                hitPositions[base + 1] = 0.0f;
                hitPositions[base + 2] = 0.0f;
                hitNormals[base + 0] = 0.0f;
                hitNormals[base + 1] = 0.0f;
                hitNormals[base + 2] = 0.0f;
                hitFlags[sampleIdx] = 0;
            }
            if (altInputs && altPositions && altNormals && altFlags) {
                altInputs[base + 0] = 0.0f;
                altInputs[base + 1] = 0.0f;
                altInputs[base + 2] = 0.0f;
                altPositions[base + 0] = 0.0f;
                altPositions[base + 1] = 0.0f;
                altPositions[base + 2] = 0.0f;
                altNormals[base + 0] = 0.0f;
                altNormals[base + 1] = 0.0f;
                altNormals[base + 2] = 0.0f;
                altFlags[sampleIdx] = 0;
            }
        }
    }
}

__global__ void debugExactNormalTransformKernel(float* inputs,
                                                float* hitPositions,
                                                float* hitNormals,
                                                int* hitFlags,
                                                RenderParams params,
                                                MeshDeviceView exactMesh,
                                                MeshDeviceView roughMesh,
                                                int stride,
                                                bool transformInputs,
                                                bool dropNonExact) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    int pixelIdx = y * params.width + x;
    int strideValue = stride > 0 ? stride : 1;
    bool activeSample = (x % strideValue == 0) && (y % strideValue == 0);
    if (!activeSample) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        if (s != 0) {
            continue;
        }
        int sampleIdx = pixelIdx + s * params.pixelCount;
        if (!hitFlags[sampleIdx]) {
            continue;
        }
        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray primaryRay = generatePrimaryRay(x, y, params, rng);
        HitInfo exactHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
        if (!traceMesh(primaryRay, exactMesh, &exactHit, true)) {
            if (dropNonExact) {
                int base = sampleIdx * 3;
                inputs[base + 0] = 0.0f;
                inputs[base + 1] = 0.0f;
                inputs[base + 2] = 0.0f;
                hitPositions[base + 0] = 0.0f;
                hitPositions[base + 1] = 0.0f;
                hitPositions[base + 2] = 0.0f;
                hitNormals[base + 0] = 0.0f;
                hitNormals[base + 1] = 0.0f;
                hitNormals[base + 2] = 0.0f;
                hitFlags[sampleIdx] = 0;
            }
            continue;
        }
        if (!transformInputs) {
            continue;
        }

        Vec3 exactPos = primaryRay.at(exactHit.distance);
        Vec3 normal = exactHit.normal;
        float nlen = length(normal);
        if (nlen > 0.0f) {
            normal = normal / nlen;
        } else {
            normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        if (dot(normal, primaryRay.direction) > 0.0f) {
            normal = normal * -1.0f;
        }

        Ray secondaryRay(exactPos + normal * 1e-3f, normal);
        HitInfo roughHit{false, 0.0f, Vec3(), Vec3(), Vec2(), -1};
        if (!traceMesh(secondaryRay, roughMesh, &roughHit, false)) {
            int base = sampleIdx * 3;
            inputs[base + 0] = 0.0f;
            inputs[base + 1] = 0.0f;
            inputs[base + 2] = 0.0f;
            hitPositions[base + 0] = 0.0f;
            hitPositions[base + 1] = 0.0f;
            hitPositions[base + 2] = 0.0f;
            hitNormals[base + 0] = 0.0f;
            hitNormals[base + 1] = 0.0f;
            hitNormals[base + 2] = 0.0f;
            hitFlags[sampleIdx] = 0;
            continue;
        }

        Vec3 roughPos = secondaryRay.at(roughHit.distance);
        Vec3 local = roughPos - params.meshMin;
        Vec3 normalized(
                local.x * params.meshInvExtent.x,
                local.y * params.meshInvExtent.y,
                local.z * params.meshInvExtent.z);
        int base = sampleIdx * 3;
        inputs[base + 0] = normalized.x;
        inputs[base + 1] = normalized.y;
        inputs[base + 2] = normalized.z;
        hitPositions[base + 0] = roughPos.x;
        hitPositions[base + 1] = roughPos.y;
        hitPositions[base + 2] = roughPos.z;
        hitNormals[base + 0] = roughHit.normal.x;
        hitNormals[base + 1] = roughHit.normal.y;
        hitNormals[base + 2] = roughHit.normal.z;
        hitFlags[sampleIdx] = 1;
    }
}

__global__ void debugExactBigPointKernel(const float* inputs,
                                         float* hitPositions,
                                         float* hitNormals,
                                         const int* hitFlags,
                                         RenderParams params,
                                         Vec3 meshMin,
                                         Vec3 meshExtent,
                                         MeshDeviceView exactMesh,
                                         int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    if (params.samplesPerPixel <= 0 || params.pixelCount <= 0) {
        return;
    }

    int pixelIdx = y * params.width + x;
    int strideValue = stride > 0 ? stride : 1;
    bool activeSample = (x % strideValue == 0) && (y % strideValue == 0);
    if (!activeSample) {
        return;
    }

    for (int s = 0; s < params.samplesPerPixel; ++s) {
        if (s != 0) {
            continue;
        }
        int sampleIdx = pixelIdx + s * params.pixelCount;
        if (!hitFlags[sampleIdx]) {
            continue;
        }

        int base = sampleIdx * 3;
        Vec3 inputNorm(
                inputs[base + 0],
                inputs[base + 1],
                inputs[base + 2]);
        Vec3 smallPos(
                inputNorm.x * meshExtent.x + meshMin.x,
                inputNorm.y * meshExtent.y + meshMin.y,
                inputNorm.z * meshExtent.z + meshMin.z);
        Vec3 exactPos = closestPointOnMesh(smallPos, exactMesh);
        hitPositions[base + 0] = exactPos.x;
        hitPositions[base + 1] = exactPos.y;
        hitPositions[base + 2] = exactPos.z;
    }
}

__global__ void debugPointBackgroundKernel(uchar4* output,
                                           RenderParams params,
                                           EnvironmentDeviceView env) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    int pixelIdx = y * params.width + x;
    uint32_t rng = initRng(pixelIdx, params.sampleOffset, 0);
    Ray primaryRay = generatePrimaryRay(x, y, params, rng);
    Vec3 color = sampleEnvironment(env, primaryRay.direction);
    color = clampRadiance(color, params.maxRadiance);

    color = encodeSrgb(color);
    output[pixelIdx] = make_uchar4(
            static_cast<unsigned char>(color.x * 255.0f),
            static_cast<unsigned char>(color.y * 255.0f),
            static_cast<unsigned char>(color.z * 255.0f),
            255);
}

__device__ inline bool projectToScreen(const Vec3& point, const RenderParams& params, int* outX, int* outY) {
    Vec3 toPoint = point - params.camPos;
    float z = dot(toPoint, params.camForward);
    if (z <= 1e-6f) {
        return false;
    }
    float x = dot(toPoint, params.camRight);
    float y = dot(toPoint, params.camUp);
    float tanHalfFovY = tanf(params.fovY * 0.5f);
    if (tanHalfFovY <= 0.0f) {
        return false;
    }
    float aspect = params.height > 0 ? static_cast<float>(params.width) / static_cast<float>(params.height) : 1.0f;
    float ndcX = x / (z * tanHalfFovY * aspect);
    float ndcY = y / (z * tanHalfFovY);
    int px = static_cast<int>((ndcX * 0.5f + 0.5f) * static_cast<float>(params.width));
    int py = static_cast<int>((0.5f - ndcY * 0.5f) * static_cast<float>(params.height));
    if (px < 0 || px >= params.width || py < 0 || py >= params.height) {
        return false;
    }
    *outX = px;
    *outY = py;
    return true;
}

__device__ inline void drawCircle(uchar4* output,
                                  const RenderParams& params,
                                  int cx,
                                  int cy,
                                  int radius,
                                  uchar4 color) {
    int r = radius > 0 ? radius : 1;
    int r2 = r * r;
    int minX = cx - r;
    int maxX = cx + r;
    int minY = cy - r;
    int maxY = cy + r;
    if (minX < 0) {
        minX = 0;
    }
    if (minY < 0) {
        minY = 0;
    }
    if (maxX >= params.width) {
        maxX = params.width - 1;
    }
    if (maxY >= params.height) {
        maxY = params.height - 1;
    }

    for (int yy = minY; yy <= maxY; ++yy) {
        int dy = yy - cy;
        for (int xx = minX; xx <= maxX; ++xx) {
            int dx = xx - cx;
            if ((dx * dx + dy * dy) <= r2) {
                output[yy * params.width + xx] = color;
            }
        }
    }
}

__device__ inline void drawLine(uchar4* output,
                                const RenderParams& params,
                                int x0,
                                int y0,
                                int x1,
                                int y1,
                                uchar4 color) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    if (steps == 0) {
        if (x0 >= 0 && x0 < params.width && y0 >= 0 && y0 < params.height) {
            output[y0 * params.width + x0] = color;
        }
        return;
    }
    float invSteps = 1.0f / static_cast<float>(steps);
    float fx = static_cast<float>(x0);
    float fy = static_cast<float>(y0);
    float stepX = static_cast<float>(dx) * invSteps;
    float stepY = static_cast<float>(dy) * invSteps;
    for (int i = 0; i <= steps; ++i) {
        int px = static_cast<int>(fx + 0.5f);
        int py = static_cast<int>(fy + 0.5f);
        if (px >= 0 && px < params.width && py >= 0 && py < params.height) {
            output[py * params.width + px] = color;
        }
        fx += stepX;
        fy += stepY;
    }
}

__global__ void debugPointDrawKernel(uchar4* output,
                                     const float* compactedInputs,
                                     const int* hitIndices,
                                     const int* hitFlags,
                                     int hitCount,
                                     const float* hitPositions,
                                     const float* lossValues,
                                     const float* lossMax,
                                     RenderParams params,
                                     Vec3 meshMin,
                                     Vec3 meshExtent,
                                     int circleRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hitCount) {
        return;
    }

    int sampleIdx = hitIndices[idx];
    if (!hitFlags[sampleIdx]) {
        return;
    }

    Vec3 inputNorm(
            compactedInputs[idx * 3 + 0],
            compactedInputs[idx * 3 + 1],
            compactedInputs[idx * 3 + 2]);
    Vec3 inputPos(
            inputNorm.x * meshExtent.x + meshMin.x,
            inputNorm.y * meshExtent.y + meshMin.y,
            inputNorm.z * meshExtent.z + meshMin.z);
    Vec3 hitPos(
            hitPositions[sampleIdx * 3 + 0],
            hitPositions[sampleIdx * 3 + 1],
            hitPositions[sampleIdx * 3 + 2]);

    int inputX = 0;
    int inputY = 0;
    int hitX = 0;
    int hitY = 0;
    bool inputVisible = projectToScreen(inputPos, params, &inputX, &inputY);
    bool hitVisible = projectToScreen(hitPos, params, &hitX, &hitY);
    if (!inputVisible && !hitVisible) {
        return;
    }

    float maxLoss = lossMax ? *lossMax : 0.0f;
    float normalized = 0.0f;
    if (maxLoss > 1e-6f) {
        normalized = fminf(lossValues[sampleIdx] / maxLoss, 1.0f);
    }
    Vec3 baseColor = Vec3(normalized, normalized, normalized);
    Vec3 lineColor = baseColor * 0.6f;
    baseColor = encodeSrgb(baseColor);
    lineColor = encodeSrgb(lineColor);
    uchar4 pointPacked = make_uchar4(
            static_cast<unsigned char>(baseColor.x * 255.0f),
            static_cast<unsigned char>(baseColor.y * 255.0f),
            static_cast<unsigned char>(baseColor.z * 255.0f),
            255);
    uchar4 linePacked = make_uchar4(
            static_cast<unsigned char>(lineColor.x * 255.0f),
            static_cast<unsigned char>(lineColor.y * 255.0f),
            static_cast<unsigned char>(lineColor.z * 255.0f),
            255);

    int radius = circleRadius > 0 ? circleRadius : 1;
    int inputRadius = radius > 1 ? radius / 2 : 1;

    if (inputVisible && hitVisible) {
        drawLine(output, params, inputX, inputY, hitX, hitY, linePacked);
    }
    if (inputVisible) {
        drawCircle(output, params, inputX, inputY, inputRadius, pointPacked);
    }
    if (hitVisible) {
        drawCircle(output, params, hitX, hitY, radius, pointPacked);
    }
}

void renderDebugPointCloud(uchar4* output,
                           float* inputs,
                           float* hitPositions,
                           float* hitNormals,
                           int* hitFlags,
                           float* lossValues,
                           float* altLossValues,
                           float* lossMax,
                           float* lossSum,
                           int* lossHitCount,
                           float* compactedInputs,
                           float* compactedDLDInput,
                           float* adamM,
                           float* adamV,
                           void* outputs,
                           void* dL_doutput,
                           int* hitIndices,
                           int* hitCount,
                           tcnn::cpp::Module* network,
                           void* networkParams,
                           uint32_t outputDims,
                           size_t outputElemSize,
                           int elementCount,
                           int gdSteps,
                           int gdSteps2,
                           float gdLearningRate,
                           float gdLearningRate2,
                           float lossThreshold,
                           int stride,
                           bool twoHitSelect,
                           float* altInputs,
                           float* altPositions,
                           float* altNormals,
                           int* altFlags,
                           bool exactNormalTransform,
                           bool dropNonExactHits,
                           bool exactBigPoints,
                           bool exactBigPointsOnly,
                           RenderParams params,
                           Vec3 meshMin,
                           Vec3 meshExtent,
                           Vec3 meshInvExtent,
                           MeshDeviceView meshView,
                           MeshDeviceView exactMeshView,
                           MeshDeviceView roughMeshView,
                           EnvironmentDeviceView envView) {
    if (!output || !inputs || !hitPositions || !hitNormals || !hitFlags || !lossValues ||
        !lossMax || !lossSum || !lossHitCount || !compactedInputs || !compactedDLDInput ||
        !adamM || !adamV || !outputs || !dL_doutput || !hitIndices || !hitCount || !network) {
        return;
    }
    if (twoHitSelect &&
        (!altInputs || !altPositions || !altNormals || !altFlags || !altLossValues)) {
        return;
    }

    if (stride < 1) {
        stride = 1;
    }
    int circleRadius = stride / 4;
    if (circleRadius < 1) {
        circleRadius = 1;
    }

    size_t inputBytes = static_cast<size_t>(elementCount) * 3 * sizeof(float);
    checkCuda(cudaMemset(hitFlags, 0, static_cast<size_t>(elementCount) * sizeof(int)),
              "cudaMemset debug hit flags");
    checkCuda(cudaMemset(inputs, 0, inputBytes), "cudaMemset debug inputs");
    checkCuda(cudaMemset(hitPositions, 0, inputBytes), "cudaMemset debug hit positions");
    checkCuda(cudaMemset(hitNormals, 0, inputBytes), "cudaMemset debug hit normals");
    checkCuda(cudaMemset(lossValues, 0, static_cast<size_t>(elementCount) * sizeof(float)),
              "cudaMemset debug loss values");

    dim3 block(8, 8);
    dim3 grid((params.width + block.x - 1) / block.x, (params.height + block.y - 1) / block.y);
    int useTwoHits = twoHitSelect ? 1 : 0;
    debugLowResRayKernel<<<grid, block>>>(
            inputs,
            hitPositions,
            hitNormals,
            hitFlags,
            params,
            meshView,
            stride,
            useTwoHits,
            altInputs,
            altPositions,
            altNormals,
            altFlags);
    checkCuda(cudaGetLastError(), "debugLowResRayKernel launch");

    if (twoHitSelect && outputDims >= 3) {
        const int blockSize = 256;
        int compactBlock = blockSize;
        int compactGrid = (elementCount + compactBlock - 1) / compactBlock;

        checkCuda(cudaMemset(lossValues, 0, static_cast<size_t>(elementCount) * sizeof(float)),
                  "cudaMemset debug two-hit loss values");
        checkCuda(cudaMemset(altLossValues, 0, static_cast<size_t>(elementCount) * sizeof(float)),
                  "cudaMemset debug two-hit loss values 2");

        checkCuda(cudaMemset(hitCount, 0, sizeof(int)), "cudaMemset debug two-hit hitCount");
        compactInputsKernel<<<compactGrid, compactBlock>>>(
                inputs,
                hitFlags,
                elementCount,
                compactedInputs,
                hitIndices,
                hitCount);
        checkCuda(cudaGetLastError(), "compactInputsKernel debug two-hit primary launch");

        int hitCountHost = 0;
        checkCuda(cudaMemcpy(&hitCountHost, hitCount, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy debug two-hit hitCount");
        if (hitCountHost > 0) {
            size_t hitCountSize = static_cast<size_t>(hitCountHost);
            size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
            size_t paddedCount = roundUp(hitCountSize, granularity);
            if (paddedCount > hitCountSize) {
                size_t tail = paddedCount - hitCountSize;
                checkCuda(cudaMemset(
                        compactedInputs + hitCountSize * 3,
                        0,
                        tail * 3 * sizeof(float)),
                        "cudaMemset debug two-hit compacted inputs tail");
            }

            {
                tcnn::cpp::Context ctx = network->forward(
                    0,
                    static_cast<uint32_t>(paddedCount),
                    compactedInputs,
                    outputs,
                    networkParams,
                    false
                );
                (void)ctx;
            }

            int lossGrid = (hitCountHost + blockSize - 1) / blockSize;
            computeLossValuesKernel<<<lossGrid, blockSize>>>(
                    compactedInputs,
                    static_cast<const __half*>(outputs),
                    hitIndices,
                    hitCountHost,
                    params,
                    meshMin,
                    meshExtent,
                    static_cast<int>(outputDims),
                    lossValues);
            checkCuda(cudaGetLastError(), "computeLossValuesKernel debug primary launch");
        }

        checkCuda(cudaMemset(hitCount, 0, sizeof(int)), "cudaMemset debug two-hit hitCount 2");
        compactInputsKernel<<<compactGrid, compactBlock>>>(
                altInputs,
                altFlags,
                elementCount,
                compactedInputs,
                hitIndices,
                hitCount);
        checkCuda(cudaGetLastError(), "compactInputsKernel debug two-hit alt launch");

        hitCountHost = 0;
        checkCuda(cudaMemcpy(&hitCountHost, hitCount, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy debug two-hit hitCount 2");
        if (hitCountHost > 0) {
            size_t hitCountSize = static_cast<size_t>(hitCountHost);
            size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
            size_t paddedCount = roundUp(hitCountSize, granularity);
            if (paddedCount > hitCountSize) {
                size_t tail = paddedCount - hitCountSize;
                checkCuda(cudaMemset(
                        compactedInputs + hitCountSize * 3,
                        0,
                        tail * 3 * sizeof(float)),
                        "cudaMemset debug two-hit compacted inputs 2 tail");
            }

            {
                tcnn::cpp::Context ctx = network->forward(
                    0,
                    static_cast<uint32_t>(paddedCount),
                    compactedInputs,
                    outputs,
                    networkParams,
                    false
                );
                (void)ctx;
            }

            int lossGrid = (hitCountHost + blockSize - 1) / blockSize;
            computeLossValuesKernel<<<lossGrid, blockSize>>>(
                    compactedInputs,
                    static_cast<const __half*>(outputs),
                    hitIndices,
                    hitCountHost,
                    params,
                    meshMin,
                    meshExtent,
                    static_cast<int>(outputDims),
                    altLossValues);
            checkCuda(cudaGetLastError(), "computeLossValuesKernel debug alt launch");
        }

        int selectGrid = (elementCount + blockSize - 1) / blockSize;
        selectLowerLossHitKernel<<<selectGrid, blockSize>>>(
                altInputs,
                altPositions,
                altNormals,
                nullptr,
                altFlags,
                altLossValues,
                inputs,
                hitPositions,
                hitNormals,
                nullptr,
                hitFlags,
                lossValues,
                elementCount);
        checkCuda(cudaGetLastError(), "selectLowerLossHitKernel debug launch");
    }

    bool transformInputs = exactNormalTransform || exactBigPoints;
    if (transformInputs || dropNonExactHits) {
        debugExactNormalTransformKernel<<<grid, block>>>(
                inputs,
                hitPositions,
                hitNormals,
                hitFlags,
                params,
                exactMeshView,
                roughMeshView,
                stride,
                transformInputs,
                dropNonExactHits);
        checkCuda(cudaGetLastError(), "debugExactNormalTransformKernel launch");
    }

    int compactBlock = 256;
    int compactGrid = (elementCount + compactBlock - 1) / compactBlock;
    checkCuda(cudaMemset(hitCount, 0, sizeof(int)), "cudaMemset debug hitCount");
    compactInputsKernel<<<compactGrid, compactBlock>>>(
            inputs,
            hitFlags,
            elementCount,
            compactedInputs,
            hitIndices,
            hitCount);
    checkCuda(cudaGetLastError(), "compactInputsKernel debug launch");

    int hitCountHost = 0;
    checkCuda(cudaMemcpy(&hitCountHost, hitCount, sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy debug hitCount");
    if (hitCountHost > 0 && outputDims >= 3) {
        size_t hitCountSize = static_cast<size_t>(hitCountHost);
        size_t granularity = static_cast<size_t>(tcnn::cpp::batch_size_granularity());
        size_t paddedCount = roundUp(hitCountSize, granularity);
        if (paddedCount > hitCountSize) {
            size_t tail = paddedCount - hitCountSize;
            checkCuda(cudaMemset(
                    compactedInputs + hitCountSize * 3,
                    0,
                    tail * 3 * sizeof(float)),
                    "cudaMemset debug compacted inputs tail");
        }

        size_t outputCount = paddedCount * static_cast<size_t>(outputDims);
        size_t outputBytes = outputCount * outputElemSize;
        const int blockSize = 256;
        int gradGrid = (hitCountHost + blockSize - 1) / blockSize;
        int inputGrid = ((hitCountHost * 3) + blockSize - 1) / blockSize;
        int steps1 = gdSteps;
        if (steps1 < 0) {
            steps1 = 0;
        }
        int steps2 = gdSteps2;
        if (steps2 < 0) {
            steps2 = 0;
        }
        int steps = steps1 + steps2;
        float learningRate = gdLearningRate;
        if (learningRate < 0.0f) {
            learningRate = 0.0f;
        }
        float learningRate2 = gdLearningRate2;
        if (learningRate2 < 0.0f) {
            learningRate2 = 0.0f;
        }
        int firstStageSteps = steps1;
        float invHitCount = 1.0f / static_cast<float>(hitCountHost);

        checkCuda(cudaMemset(adamM, 0, hitCountHost * 3 * sizeof(float)), "cudaMemset debug adam m");
        checkCuda(cudaMemset(adamV, 0, hitCountHost * 3 * sizeof(float)), "cudaMemset debug adam v");

        for (int step = 0; step < steps; ++step) {
            float stepLearningRate = (step < firstStageSteps) ? learningRate : learningRate2;
            tcnn::cpp::Context ctx = network->forward(
                0,
                static_cast<uint32_t>(paddedCount),
                compactedInputs,
                outputs,
                networkParams,
                true
            );
            (void)ctx;

            checkCuda(cudaMemset(dL_doutput, 0, outputBytes), "cudaMemset debug dL_doutput");
            computeLossGradKernel<<<gradGrid, blockSize>>>(
                    compactedInputs,
                    static_cast<const __half*>(outputs),
                    hitIndices,
                    hitCountHost,
                    invHitCount,
                    params,
                    meshMin,
                    meshExtent,
                    static_cast<int>(outputDims),
                    static_cast<__half*>(dL_doutput));
            checkCuda(cudaGetLastError(), "computeLossGradKernel debug launch");

            network->backward(
                    0,
                    ctx,
                    static_cast<uint32_t>(paddedCount),
                    compactedDLDInput,
                    dL_doutput,
                    nullptr,
                    compactedInputs,
                    outputs,
                    networkParams);

            addDirectGradKernel<<<gradGrid, blockSize>>>(
                    compactedInputs,
                    static_cast<const __half*>(outputs),
                    hitIndices,
                    hitCountHost,
                    invHitCount,
                    params,
                    meshMin,
                    meshExtent,
                    static_cast<int>(outputDims),
                    compactedDLDInput);
            checkCuda(cudaGetLastError(), "addDirectGradKernel debug launch");

            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float eps = 1e-8f;
            float t = static_cast<float>(step + 1);
            float alpha = stepLearningRate * sqrtf(1.0f - powf(beta2, t)) /
                    (1.0f - powf(beta1, t));
            adamInputsKernel<<<inputGrid, blockSize>>>(
                    compactedInputs,
                    compactedDLDInput,
                    adamM,
                    adamV,
                    hitCountHost,
                    alpha,
                    beta1,
                    beta2,
                    eps);
            checkCuda(cudaGetLastError(), "adamInputsKernel debug launch");

            projectInputsToMeshKernel<<<gradGrid, blockSize>>>(
                    compactedInputs,
                    hitCountHost,
                    meshMin,
                    meshExtent,
                    meshInvExtent,
                    meshView);
            checkCuda(cudaGetLastError(), "projectInputsToMeshKernel debug launch");
        }

        {
            tcnn::cpp::Context ctx = network->forward(
                0,
                static_cast<uint32_t>(paddedCount),
                compactedInputs,
                outputs,
                networkParams,
                false
            );
            (void)ctx;
        }

        int applyGrid = (hitCountHost + blockSize - 1) / blockSize;
        applyNetworkDeltaKernel<<<applyGrid, blockSize>>>(
                compactedInputs,
                static_cast<const __half*>(outputs),
                hitIndices,
                hitCountHost,
                params,
                meshMin,
                meshExtent,
                static_cast<int>(outputDims),
                lossThreshold,
                hitFlags,
                hitPositions,
                hitNormals);
        checkCuda(cudaGetLastError(), "applyNetworkDeltaKernel debug launch");
    }

    if (exactBigPoints || exactBigPointsOnly) {
        debugExactBigPointKernel<<<grid, block>>>(
                inputs,
                hitPositions,
                hitNormals,
                hitFlags,
                params,
                meshMin,
                meshExtent,
                exactMeshView,
                stride);
        checkCuda(cudaGetLastError(), "debugExactBigPointKernel launch");
    }

    checkCuda(cudaMemset(lossMax, 0, sizeof(float)), "cudaMemset debug lossMax");
    checkCuda(cudaMemset(lossSum, 0, sizeof(float)), "cudaMemset debug lossSum");
    checkCuda(cudaMemset(lossHitCount, 0, sizeof(int)), "cudaMemset debug lossHitCount");
    lossNeuralKernel<<<grid, block>>>(lossValues, hitPositions, hitFlags, params, lossMax, lossSum, lossHitCount);
    checkCuda(cudaGetLastError(), "lossNeuralKernel debug launch");

    debugPointBackgroundKernel<<<grid, block>>>(
            output,
            params,
            envView);
    checkCuda(cudaGetLastError(), "debugPointBackgroundKernel launch");

    if (hitCountHost <= 0) {
        return;
    }

    int drawBlock = 256;
    int drawGrid = (hitCountHost + drawBlock - 1) / drawBlock;
    debugPointDrawKernel<<<drawGrid, drawBlock>>>(
            output,
            compactedInputs,
            hitIndices,
            hitFlags,
            hitCountHost,
            hitPositions,
            lossValues,
            lossMax,
            params,
            meshMin,
            meshExtent,
            circleRadius);
    checkCuda(cudaGetLastError(), "debugPointDrawKernel launch");
}
