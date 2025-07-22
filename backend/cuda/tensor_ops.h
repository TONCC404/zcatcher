// tensor_ops.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launchTranspose2D(const float* input, float* output, int rows, int cols);
void launchSumAxis0(const float* input, float* output, int rows, int cols);
void launchSumAxis1(const float* input, float* output, int rows, int cols);
void launchMatMul2D(const float* A, const float* B, float* C, int m, int k, int n);
void launchAddBias2D(const float* mat, const float* bias, float* out, int rows, int cols);

#ifdef __cplusplus
}
#endif
