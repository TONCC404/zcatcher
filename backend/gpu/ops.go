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
void launchZeroPad(const float* input, float* output, int B, int C, int H, int W, int padding);
void launchSlice(const float* input, float* output, int dims, const int* inputShape, const int* start, const int* outShape, int totalElements);
*/
import "C"
import (
	"fmt"
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

func (GPUBackend) ZeroPad(input *tensor.Tensor, padding int) *tensor.Tensor {
	B, C, H, W := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	outH, outW := H+2*padding, W+2*padding
	outShape := []int{B, C, outH, outW}
	output := tensor.NewTensor(nil, outShape, "gpu") // 分配GPU内存，Data指针是GPU地址

	inputPtr := unsafe.Pointer(&input.Data[0])
	if output.gpuPtr == nil {
		panic("Failed to allocate GPU memory")
	}
	outputPtr := output.gpuPtr
	C.launchZeroPad(
		(*C.float)(inputPtr),
		(*C.float)(outputPtr),
		C.int(B), C.int(C), C.int(H), C.int(W), C.int(padding),
	)
	// 这里可以cudaDeviceSynchronize或用stream同步
	return output
}

func (GPUBackend) Offset(shape []int, indices []int) int {
	if len(shape) != len(indices) {
		panic("Offset: shape and indices length mismatch")
	}
	offset := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= shape[i] {
			panic(fmt.Sprintf("Offset: index %d out of bounds for axis %d", indices[i], i))
		}
		offset += indices[i] * stride
		stride *= shape[i]
	}
	return offset
}

func (GPUBackend) Get(t *tensor.Tensor, indices ...int) float32 {
	if len(indices) != len(t.Shape) {
		panic("Get: dimension mismatch")
	}
	offset := GPUBackend{}.Offset(t.Shape, indices)
	var val float32
	// 只拷贝单元素
	err := cudaMemcpyHostToDevice(&val, t.Data, offset, 1)
	if err != nil {
		panic(err)
	}
	return val
}

func (GPUBackend) Set(t *tensor.Tensor, value float32, indices ...int) {
	if len(indices) != len(t.Shape) {
		panic("Set: dimension mismatch")
	}
	offset := GPUBackend{}.Offset(t.Shape, indices)
	// 单元素从host拷贝到device
	err := cudaMemcpyDeviceToHost(t.Data, &value, offset, 1)
	if err != nil {
		panic(err)
	}
}

func (GPUBackend) Slice(t *tensor.Tensor, start []int, end []int) *tensor.Tensor {
	if len(start) != len(t.Shape) || len(end) != len(t.Shape) {
		panic("Slice: dimension mismatch")
	}
	outShape := make([]int, len(t.Shape))
	for i := range start {
		if start[i] < 0 || end[i] > t.Shape[i] || start[i] >= end[i] {
			panic(fmt.Sprintf("Slice: invalid slice range on axis %d", i))
		}
		outShape[i] = end[i] - start[i]
	}
	output := tensor.NewEmptyGPU(outShape)

	dims := C.int(len(t.Shape))
	totalElements := 1
	for _, v := range outShape {
		totalElements *= v
	}

	cInputShape := (*C.int)(unsafe.Pointer(&t.Shape[0]))
	cStart := (*C.int)(unsafe.Pointer(&start[0]))
	cOutShape := (*C.int)(unsafe.Pointer(&outShape[0]))

	C.launchSlice(
		(*C.float)(unsafe.Pointer(t.Data)),
		(*C.float)(unsafe.Pointer(output.Data)),
		dims,
		cInputShape,
		cStart,
		cOutShape,
		C.int(totalElements),
	)

	return output
}

func (GPUBackend) Reshape(t *tensor.Tensor, newShape []int) *tensor.Tensor {
	oldSize := 1
	for _, dim := range t.Shape {
		oldSize *= dim
	}
	newSize := 1
	for _, dim := range newShape {
		if dim <= 0 {
			panic("Reshape: invalid new shape dimension")
		}
		newSize *= dim
	}
	if oldSize != newSize {
		panic("Reshape: total size mismatch")
	}
	return &tensor.Tensor{
		Data:   t.Data,
		Shape:  append([]int(nil), newShape...),
		Device: t.Device,
	}
}
