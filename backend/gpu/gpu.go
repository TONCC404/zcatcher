package gpu

// #cgo LDFLAGS: -L./cuda -ltensor_ops -lcublas -lcudart
// #cgo CFLAGS: -I/usr/local/cuda/include
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// void launchTranspose2D(const float* input, float* output, int rows, int cols);
// void launchSumAxis0(const float* input, float* output, int rows, int cols);
// void launchSumAxis1(const float* input, float* output, int rows, int cols);
// void launchMatMul2D(const float* A, const float* B, float* C, int m, int k, int n);
// void launchAddBias2D(const float* mat, const float* bias, float* out, int rows, int cols);
import "C"

type GPUBackend struct{}

func (GPUBackend) Device() string { return "gpu" }

func NewGPUBackend() *GPUBackend {
	return &GPUBackend{}
}
