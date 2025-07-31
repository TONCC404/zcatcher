// tensor_ops.cu
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void transpose2D(const float* input, float* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

__global__ void sumAxis0(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            sum += input[i * cols + col];
        }
        output[col] = sum;
    }
}

__global__ void sumAxis1(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += input[row * cols + j];
        }
        output[row] = sum;
    }
}

__global__ void matmul2D(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// mat += bias (1 x n) => each row adds bias[j]
__global__ void addBias2D(const float* mat, const float* bias, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = mat[idx] + bias[col];
    }
}

__global__ void zeroPadKernel(const float* input, float* output,
                              int B, int C, int H, int W, int padding) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h = threadIdx.y + blockIdx.x / ((W + 2*padding + 15) / 16) * 16;
    int w = threadIdx.x + (blockIdx.x % ((W + 2*padding + 15) / 16)) * 16;

    int outH = H + 2 * padding;
    int outW = W + 2 * padding;

    if (h < outH && w < outW) {
        int outIdx = b*C*outH*outW + c*outH*outW + h*outW + w;
        // check if in input range
        int inH = h - padding;
        int inW = w - padding;
        if (inH >=0 && inH < H && inW >= 0 && inW < W) {
            int inIdx = b*C*H*W + c*H*W + inH*W + inW;
            output[outIdx] = input[inIdx];
        } else {
            output[outIdx] = 0.0f;
        }
    }
}

__global__ void sliceKernel(const float* input, float* output,
                            int dims,
                            const int* inputShape, const int* start, const int* outShape,
                            int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    // Compute multi-dimensional index in output
    int outIdx = idx;
    int outIndices[8]; // 支持最多8维，可扩展
#pragma unroll
    for (int i = dims-1; i >= 0; i--) {
        outIndices[i] = outIdx % outShape[i];
        outIdx /= outShape[i];
    }

    // 计算input索引 = start + outIndices
    int inOffset = 0;
    int stride = 1;
    for (int i = dims-1; i >= 0; i--) {
        int inIdx = start[i] + outIndices[i];
        inOffset += inIdx * stride;
        stride *= inputShape[i];
    }

    output[idx] = input[inOffset];
}

void launchMatMul2D(const float* A, const float* B, float* C, int m, int k, int n) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (m + 15) / 16);
    matmul2D<<<numBlocks, threadsPerBlock>>>(A, B, C, m, k, n);
}

void launchAddBias2D(const float* mat, const float* bias, float* out, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);
    addBias2D<<<numBlocks, threadsPerBlock>>>(mat, bias, out, rows, cols);
}

void launchTranspose2D(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);
    transpose2D<<<numBlocks, threadsPerBlock>>>(input, output, rows, cols);
}

void launchSumAxis0(const float* input, float* output, int rows, int cols) {
    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    sumAxis0<<<blocks, threads>>>(input, output, rows, cols);
}

void launchSumAxis1(const float* input, float* output, int rows, int cols) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    sumAxis1<<<blocks, threads>>>(input, output, rows, cols);
}

void launchZeroPad(const float* input, float* output,
                   int B, int C, int H, int W, int padding) {
    dim3 threadsPerBlock(16, 16);
    int outH = H + 2 * padding;
    int outW = W + 2 * padding;
    int blocksX = (outW + 15) / 16;
    int blocksY = (outH + 15) / 16;

    // 采用 (blocksX * blocksY) 作为 blockIdx.x，blockIdx.y=c, blockIdx.z=b
    dim3 numBlocks(blocksX * blocksY, C, B);

    zeroPadKernel<<<numBlocks, threadsPerBlock>>>(input, output, B, C, H, W, padding);
}

void launchSlice(const float* input, float* output,
                 int dims,
                 const int* inputShape, const int* start, const int* outShape,
                 int totalElements) {
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;
    sliceKernel<<<blocks, threads>>>(input, output, dims, inputShape, start, outShape, totalElements);
}
}// extern "C"
