#pragma once

__global__ void debugLowResRayKernel(float* inputs,
                                     float* hitPositions,
                                     float* hitNormals,
                                     int* hitFlags,
                                     RenderParams params,
                                     MeshDeviceView mesh,
                                     int stride) {
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
            continue;
        }

        uint32_t rng = initRng(pixelIdx, params.sampleOffset, s);
        Ray ray = generatePrimaryRay(x, y, params, rng);
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
                           float* lossMax,
                           float* lossSum,
                           int* lossHitCount,
                           float* compactedInputs,
                           float* compactedDLDInput,
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
                           float gdLearningRate,
                           float lossThreshold,
                           int stride,
                           RenderParams params,
                           Vec3 meshMin,
                           Vec3 meshExtent,
                           Vec3 meshInvExtent,
                           MeshDeviceView meshView,
                           EnvironmentDeviceView envView) {
    if (!output || !inputs || !hitPositions || !hitNormals || !hitFlags || !lossValues ||
        !lossMax || !lossSum || !lossHitCount || !compactedInputs || !compactedDLDInput ||
        !outputs || !dL_doutput || !hitIndices || !hitCount || !network) {
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
    debugLowResRayKernel<<<grid, block>>>(
            inputs,
            hitPositions,
            hitNormals,
            hitFlags,
            params,
            meshView,
            stride);
    checkCuda(cudaGetLastError(), "debugLowResRayKernel launch");

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
        int steps = gdSteps;
        if (steps < 0) {
            steps = 0;
        }
        float learningRate = gdLearningRate;
        if (learningRate < 0.0f) {
            learningRate = 0.0f;
        }
        float invHitCount = 1.0f / static_cast<float>(hitCountHost);

        for (int step = 0; step < steps; ++step) {
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

            sgdInputsKernel<<<inputGrid, blockSize>>>(
                    compactedInputs,
                    compactedDLDInput,
                    hitCountHost,
                    learningRate);
            checkCuda(cudaGetLastError(), "sgdInputsKernel debug launch");

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
