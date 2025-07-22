package gpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../cuda -lactivations -lcublas -lcudart
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../cuda/activations.h"

void launchSigmoid(const float* input, float* output, int n, int inplace);
void launchTanh(const float* input, float* output, int n, int inplace);
void launchLeakyReLU(const float* input, float* output, int n, float alpha, int inplace);
void launchELU(const float* input, float* output, int n, float alpha, int inplace);
void launchReLU(const float* input, float* output, int n, int inplace);
*/
import "C"
import (
	"unsafe"
	"zcatcher/tensor"
)

func (GPUBackend) Sigmoid(t *tensor.Tensor) *tensor.Tensor {
	out := make([]float32, len(t.Data))
	inPtr := (*C.float)(unsafe.Pointer(&t.Data[0]))
	outPtr := (*C.float)(unsafe.Pointer(&out[0]))
	C.launchSigmoid(inPtr, outPtr, C.int(len(t.Data)), C.int(0))
	return &tensor.Tensor{Data: out, Shape: t.Shape, Device: "gpu"}
}

func (GPUBackend) Tanh(t *tensor.Tensor) *tensor.Tensor {
	out := make([]float32, len(t.Data))
	inPtr := (*C.float)(unsafe.Pointer(&t.Data[0]))
	outPtr := (*C.float)(unsafe.Pointer(&out[0]))
	C.launchTanh(inPtr, outPtr, C.int(len(t.Data)), C.int(0))
	return &tensor.Tensor{Data: out, Shape: t.Shape, Device: "gpu"}
}

func (GPUBackend) LeakyReLU(t *tensor.Tensor, alpha float32) *tensor.Tensor {
	out := make([]float32, len(t.Data))
	inPtr := (*C.float)(unsafe.Pointer(&t.Data[0]))
	outPtr := (*C.float)(unsafe.Pointer(&out[0]))
	C.launchLeakyReLU(inPtr, outPtr, C.int(len(t.Data)), C.float(alpha), C.int(0))
	return &tensor.Tensor{Data: out, Shape: t.Shape, Device: "gpu"}
}

func (GPUBackend) ELU(t *tensor.Tensor, alpha float32) *tensor.Tensor {
	out := make([]float32, len(t.Data))
	inPtr := (*C.float)(unsafe.Pointer(&t.Data[0]))
	outPtr := (*C.float)(unsafe.Pointer(&out[0]))
	C.launchELU(inPtr, outPtr, C.int(len(t.Data)), C.float(alpha), C.int(0))
	return &tensor.Tensor{Data: out, Shape: t.Shape, Device: "gpu"}
}

func (GPUBackend) ReLU(t *tensor.Tensor) *tensor.Tensor {
	out := make([]float32, len(t.Data))
	inPtr := (*C.float)(unsafe.Pointer(&t.Data[0]))
	outPtr := (*C.float)(unsafe.Pointer(&out[0]))
	C.launchReLU(inPtr, outPtr, C.int(len(t.Data)), C.int(0))
	return &tensor.Tensor{Data: out, Shape: t.Shape, Device: "gpu"}
}
