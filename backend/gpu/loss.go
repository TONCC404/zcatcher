package gpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../cuda -ltensor_ops -lcublas -lcudart
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
#include <stdlib.h>
void launchCategoricalCrossEntropy(const float*, const float*, float*, float*, int);
void launchMSELoss(const float*, const float*, float*, float*, int);
void launchMAELoss(const float*, const float*, float*, float*, int);
void launchBinaryCrossEntropy(const float*, const float*, float*, float*, int);
void launchSmoothL1Loss(const float*, const float*, float*, float*, int);
*/
import "C"
import (
	"unsafe"
	"zcatcher/tensor"
)

func runLossKernel(pred, target *tensor.Tensor, size int, launchFunc func(predPtr, targetPtr, lossPtr, gradPtr unsafe.Pointer, size C.int)) (float32, *tensor.Tensor) {
	loss := C.malloc(C.size_t(4))        // 分配4字节存放float32 loss
	grad := C.malloc(C.size_t(size * 4)) // 分配梯度缓冲区
	defer C.free(loss)
	defer C.free(grad)

	// 调用具体的launch函数
	launchFunc(
		unsafe.Pointer(&pred.Data[0]),
		unsafe.Pointer(&target.Data[0]),
		loss,
		grad,
		C.int(size),
	)

	// 拷贝loss到Go
	goLoss := *(*float32)(loss)
	// 拷贝梯度从设备到主机
	goGrad := make([]float32, size)
	C.cudaMemcpy(
		unsafe.Pointer(&goGrad[0]),
		grad,
		C.size_t(size*4),
		C.cudaMemcpyDeviceToHost,
	)

	return goLoss, &tensor.Tensor{
		Data:   goGrad,
		Shape:  pred.Shape,
		Device: "gpu",
	}
}

func CategoricalCrossEntropy(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(
		pred,
		target,
		len(pred.Data),
		func(predPtr, targetPtr, lossPtr, gradPtr unsafe.Pointer, size C.int) {
			C.launchCategoricalCrossEntropy(
				(*C.float)(predPtr),
				(*C.float)(targetPtr),
				(*C.float)(lossPtr),
				(*C.float)(gradPtr),
				size,
			)
		})
}

func MSELoss(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), func(predPtr, targetPtr, lossPtr, gradPtr unsafe.Pointer, size C.int) {
		C.launchMSELoss(
			(*C.float)(predPtr),
			(*C.float)(targetPtr),
			(*C.float)(lossPtr),
			(*C.float)(gradPtr),
			size,
		)
	})
}

func MAELoss(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), func(predPtr, targetPtr, lossPtr, gradPtr unsafe.Pointer, size C.int) {
		C.launchMAELoss(
			(*C.float)(predPtr),
			(*C.float)(targetPtr),
			(*C.float)(lossPtr),
			(*C.float)(gradPtr),
			size,
		)
	})
}

func BinaryCrossEntropy(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), func(predPtr, targetPtr, lossPtr, gradPtr unsafe.Pointer, size C.int) {
		C.launchBinaryCrossEntropy(
			(*C.float)(predPtr),
			(*C.float)(targetPtr),
			(*C.float)(lossPtr),
			(*C.float)(gradPtr),
			size,
		)
	})
}

func SmoothL1Loss(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), func(predPtr, targetPtr, lossPtr, gradPtr unsafe.Pointer, size C.int) {
		C.launchSmoothL1Loss(
			(*C.float)(predPtr),
			(*C.float)(targetPtr),
			(*C.float)(lossPtr),
			(*C.float)(gradPtr),
			size,
		)
	})
}
