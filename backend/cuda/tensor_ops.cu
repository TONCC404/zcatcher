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
}// extern "C"
