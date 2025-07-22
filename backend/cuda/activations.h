#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

// CUDA 前向传播函数声明
void launchSigmoid(const float* input, float* output, int n, int inplace);
void launchTanh(const float* input, float* output, int n, int inplace);
void launchLeakyReLU(const float* input, float* output, int n, float alpha, int inplace);
void launchELU(const float* input, float* output, int n, float alpha, int inplace);
void launchReLU(const float* input, float* output, int n, int inplace);

#ifdef __cplusplus
}
#endif

#endif // ACTIVATIONS_H