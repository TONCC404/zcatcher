package backend

// #cgo LDFLAGS: -lcublas -lcudart
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
import "C"

import (
	"unsafe"
	"zcatcher/tensor"
)

type GPUBackend struct{}

func (GPUBackend) Device() string { return "gpu" }

func (GPUBackend) MatMul(a, b *tensor.Tensor) *tensor.Tensor {
	// 创建 cuBLAS handle
	var handle C.cublasHandle_t
	C.cublasCreate_v2(&handle)

	// 分配 GPU memory 并复制数据（略去错误处理）
	var dA, dB, dC unsafe.Pointer
	sizeA := len(a.Data) * 4
	sizeB := len(b.Data) * 4
	sizeC := a.Shape[0] * b.Shape[1] * 4

	C.cudaMalloc(&dA, C.size_t(sizeA))
	C.cudaMemcpy(dA, unsafe.Pointer(&a.Data[0]), C.size_t(sizeA), C.cudaMemcpyHostToDevice)
	C.cudaMalloc(&dB, C.size_t(sizeB))
	C.cudaMemcpy(dB, unsafe.Pointer(&b.Data[0]), C.size_t(sizeB), C.cudaMemcpyHostToDevice)
	C.cudaMalloc(&dC, C.size_t(sizeC))

	// 调用 cuBLAS：C = A * B
	m := C.int(a.Shape[0])
	n := C.int(b.Shape[1])
	k := C.int(a.Shape[1])
	alpha := C.float(1.0)
	beta := C.float(0.0)

	C.cublasSgemm(handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		n, m, k,
		&alpha,
		(*C.float)(dB), n,
		(*C.float)(dA), k,
		&beta,
		(*C.float)(dC), n)

	// 从 GPU 拷贝回 CPU
	out := make([]float32, int(m)*int(n))
	C.cudaMemcpy(unsafe.Pointer(&out[0]), dC, C.size_t(sizeC), C.cudaMemcpyDeviceToHost)

	// 清理资源
	C.cudaFree(dA)
	C.cudaFree(dB)
	C.cudaFree(dC)
	C.cublasDestroy_v2(handle)

	return &tensor.Tensor{Data: out, Shape: []int{int(m), int(n)}, Device: "cpu"}
}
