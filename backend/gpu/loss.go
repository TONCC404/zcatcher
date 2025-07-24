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

func runLossKernel(pred, target *tensor.Tensor, size int, lossFunc func(*C.float, *C.float, *C.float, *C.float, C.int)) (float32, *tensor.Tensor) {
	loss := C.malloc(C.size_t(4))
	grad := C.malloc(C.size_t(size * 4))
	defer C.free(loss)
	defer C.free(grad)

	lossFunc(
		(*C.float)(unsafe.Pointer(&pred.Data[0])),
		(*C.float)(unsafe.Pointer(&target.Data[0])),
		(*C.float)(loss),
		(*C.float)(grad),
		C.int(size),
	)

	// 拷贝结果
	goLoss := *(*float32)(loss)
	goGrad := make([]float32, size)
	C.cudaMemcpy(unsafe.Pointer(&goGrad[0]), grad, C.size_t(size*4), C.cudaMemcpyDeviceToHost)

	return goLoss, &tensor.Tensor{
		Data:   goGrad,
		Shape:  pred.Shape,
		Device: "gpu",
	}
}

func CategoricalCrossEntropy(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), C.launchCategoricalCrossEntropy)
}

func MSELoss(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), C.launchMSELoss)
}

func MAELoss(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), C.launchMAELoss)
}

func BinaryCrossEntropy(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), C.launchBinaryCrossEntropy)
}

func SmoothL1Loss(pred, target *tensor.Tensor) (float32, *tensor.Tensor) {
	return runLossKernel(pred, target, len(pred.Data), C.launchSmoothL1Loss)
}
