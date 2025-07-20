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

func NewGPUBackend() *GPUBackend {
	return &GPUBackend{}
}

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

func (GPUBackend) AddBias(matrix, bias *tensor.Tensor) *tensor.Tensor {
	// 检查输入形状是否兼容
	if len(matrix.Shape) != 2 || len(bias.Shape) != 1 || matrix.Shape[1] != bias.Shape[0] {
		panic("incompatible shapes for AddBias operation")
	}

	// 创建 cuBLAS handle
	var handle C.cublasHandle_t
	C.cublasCreate_v2(&handle)
	defer C.cublasDestroy_v2(handle)

	// 分配 GPU memory
	var dMatrix, dBias unsafe.Pointer
	matrixSize := C.size_t(len(matrix.Data) * 4)
	biasSize := C.size_t(len(bias.Data) * 4)

	C.cudaMalloc(&dMatrix, C.size_t(matrixSize))
	defer C.cudaFree(dMatrix)
	C.cudaMemcpy(dMatrix, unsafe.Pointer(&matrix.Data[0]), C.size_t(matrixSize), C.cudaMemcpyHostToDevice)

	C.cudaMalloc(&dBias, C.size_t(biasSize))
	defer C.cudaFree(dBias)
	C.cudaMemcpy(dBias, unsafe.Pointer(&bias.Data[0]), C.size_t(biasSize), C.cudaMemcpyHostToDevice)

	// 执行偏置加法: matrix += bias (broadcasted to each row)
	m := C.int(matrix.Shape[0])
	n := C.int(matrix.Shape[1])
	alpha := C.float(1.0)

	// 对每一行执行 y = alpha * x + y
	// 这里我们使用 cublasSaxpy 的批处理版本
	for i := 0; i < int(m); i++ {
		// 计算当前行的指针偏移
		rowOffset := C.size_t(i) * C.size_t(n) * 4
		rowPtr := unsafe.Pointer(uintptr(dMatrix) + uintptr(rowOffset))

		C.cublasSaxpy_v2(handle,
			n,
			&alpha,
			(*C.float)(dBias), 1,
			(*C.float)(rowPtr), 1)
	}

	// 从 GPU 拷贝回 CPU
	out := make([]float32, len(matrix.Data))
	C.cudaMemcpy(unsafe.Pointer(&out[0]), dMatrix, C.size_t(matrixSize), C.cudaMemcpyDeviceToHost)

	return &tensor.Tensor{Data: out, Shape: matrix.Shape, Device: "cpu"}
}
