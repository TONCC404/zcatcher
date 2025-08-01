package tensor

/*
#cgo LDFLAGS: -lcuda
#include <cuda_runtime.h>
*/
import "C"
import "unsafe"

type Tensor struct {
	Data  []float32
	Shape []int
	// 可选设备标签：cpu / gpu
	Device string
	gpuPtr unsafe.Pointer
	Mask   []byte
}

func NewTensor(data []float32, shape []int, device ...string) *Tensor {
	dev := "cpu"
	if len(device) > 0 {
		dev = device[0]
	}

	// 如果是 GPU，分配显存并拷贝数据
	if dev == "gpu" {
		// 1. 在 GPU 上分配内存（通过 CUDA）
		var gpuPtr unsafe.Pointer
		size := len(data) * 4 // float32 占 4 字节
		C.cudaMalloc(&gpuPtr, C.size_t(size))

		// 2. 如果提供了数据，拷贝到 GPU
		if len(data) > 0 {
			C.cudaMemcpy(
				gpuPtr,
				unsafe.Pointer(&data[0]),
				C.size_t(size),
				C.cudaMemcpyHostToDevice,
			)
		}

		// 3. 返回 GPU 张量
		return &Tensor{
			Data:   nil, // GPU 数据不保存在 Go 的切片中
			Shape:  shape,
			Device: "gpu",
			gpuPtr: gpuPtr,
		}
	}
	return &Tensor{
		Data:   data,
		Shape:  shape,
		Device: dev,
		gpuPtr: nil,
	}
}

func NewZeros(shape []int, device ...string) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	dev := "cpu"
	if len(device) > 0 {
		dev = device[0]
	}

	return NewTensor(data, shape, dev)
}
