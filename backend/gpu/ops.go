package gpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../cuda -ltensor_ops -lcublas -lcudart
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
#include <cublas_v2.h>
void launchTranspose2D(const float* input, float* output, int rows, int cols);
void launchSumAxis0(const float* input, float* output, int rows, int cols);
void launchSumAxis1(const float* input, float* output, int rows, int cols);
void launchMatMul2D(const float* A, const float* B, float* C, int m, int k, int n);
void launchAddBias2D(const float* mat, const float* bias, float* out, int rows, int cols);
*/
import "C"
import (
	"unsafe"
	"zcatcher/tensor"
)

func (GPUBackend) MatMul(a, b *tensor.Tensor) *tensor.Tensor {
	m, k := a.Shape[0], a.Shape[1]
	n := b.Shape[1]
	out := make([]float32, m*n)

	aPtr := (*C.float)(unsafe.Pointer(&a.Data[0]))
	bPtr := (*C.float)(unsafe.Pointer(&b.Data[0]))
	cPtr := (*C.float)(unsafe.Pointer(&out[0]))

	C.launchMatMul2D(aPtr, bPtr, cPtr, C.int(m), C.int(k), C.int(n))

	return &tensor.Tensor{Data: out, Shape: []int{m, n}, Device: "gpu"}
}

func (GPUBackend) AddBias(mat, bias *tensor.Tensor) *tensor.Tensor {
	rows, cols := mat.Shape[0], mat.Shape[1]
	if bias.Shape[1] != cols {
		panic("Bias shape mismatch")
	}

	out := make([]float32, len(mat.Data))
	matPtr := (*C.float)(unsafe.Pointer(&mat.Data[0]))
	biasPtr := (*C.float)(unsafe.Pointer(&bias.Data[0]))
	outPtr := (*C.float)(unsafe.Pointer(&out[0]))

	C.launchAddBias2D(matPtr, biasPtr, outPtr, C.int(rows), C.int(cols))

	return &tensor.Tensor{Data: out, Shape: mat.Shape, Device: "gpu"}
}

func (GPUBackend) Transpose(t *tensor.Tensor) *tensor.Tensor {
	rows, cols := t.Shape[0], t.Shape[1]
	out := make([]float32, len(t.Data))

	inputPtr := (*C.float)(unsafe.Pointer(&t.Data[0]))
	outputPtr := (*C.float)(unsafe.Pointer(&out[0]))
	C.launchTranspose2D(inputPtr, outputPtr, C.int(rows), C.int(cols))

	return &tensor.Tensor{Data: out, Shape: []int{cols, rows}, Device: "gpu"}
}

func (GPUBackend) Sum(t *tensor.Tensor, axis int) *tensor.Tensor {
	rows, cols := t.Shape[0], t.Shape[1]
	var out []float32
	if axis == 0 {
		out = make([]float32, cols)
		C.launchSumAxis0((*C.float)(unsafe.Pointer(&t.Data[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(rows), C.int(cols))
		return &tensor.Tensor{Data: out, Shape: []int{1, cols}, Device: "gpu"}
	} else if axis == 1 {
		out = make([]float32, rows)
		C.launchSumAxis1((*C.float)(unsafe.Pointer(&t.Data[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(rows), C.int(cols))
		return &tensor.Tensor{Data: out, Shape: []int{rows, 1}, Device: "gpu"}
	} else {
		panic("Invalid axis")
	}
}
