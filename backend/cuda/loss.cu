// tensor/cuda/loss.cu
#include <cuda_runtime.h>
#include <math.h>
#include "loss.h"

__global__ void categoricalCrossEntropyKernel(const float* pred, const float* target, float* loss, float* grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float p = pred[i];
        float t = target[i];
        grad[i] = p - t;
        atomicAdd(loss, -t * logf(p + 1e-8f));
    }
}

__global__ void mseLossKernel(const float* pred, const float* target, float* loss, float* grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];
        grad[i] = 2.0f * diff;
        atomicAdd(loss, diff * diff);
    }
}

__global__ void maeLossKernel(const float* pred, const float* target, float* loss, float* grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];
        grad[i] = (diff > 0) ? 1.0f : -1.0f;
        atomicAdd(loss, fabsf(diff));
    }
}

__global__ void binaryCrossEntropyKernel(const float* pred, const float* target, float* loss, float* grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float p = pred[i];
        float t = target[i];
        grad[i] = (p - t) / (p * (1.0f - p) + 1e-8f);
        atomicAdd(loss, -(t * logf(p + 1e-8f) + (1 - t) * logf(1 - p + 1e-8f)));
    }
}

__global__ void smoothL1Kernel(const float* pred, const float* target, float* loss, float* grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            grad[i] = diff;
            atomicAdd(loss, 0.5f * diff * diff);
        } else {
            grad[i] = (diff > 0) ? 1.0f : -1.0f;
            atomicAdd(loss, abs_diff - 0.5f);
        }
    }
}

void launchCategoricalCrossEntropy(const float* pred, const float* target, float* loss, float* grad, int size) {
    categoricalCrossEntropyKernel<<<(size + 255) / 256, 256>>>(pred, target, loss, grad, size);
}

void launchMSELoss(const float* pred, const float* target, float* loss, float* grad, int size) {
    mseLossKernel<<<(size + 255) / 256, 256>>>(pred, target, loss, grad, size);
}

void launchMAELoss(const float* pred, const float* target, float* loss, float* grad, int size) {
    maeLossKernel<<<(size + 255) / 256, 256>>>(pred, target, loss, grad, size);
}

void launchBinaryCrossEntropy(const float* pred, const float* target, float* loss, float* grad, int size) {
    binaryCrossEntropyKernel<<<(size + 255) / 256, 256>>>(pred, target, loss, grad, size);
}

void launchSmoothL1Loss(const float* pred, const float* target, float* loss, float* grad, int size) {
    smoothL1Kernel<<<(size + 255) / 256, 256>>>(pred, target, loss, grad, size);
}
