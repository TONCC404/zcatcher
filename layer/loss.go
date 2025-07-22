package layer

import (
	"errors"
	"math"
	"zcatcher/backend/gpu"
	"zcatcher/tensor"
)

// func CrossEntropy(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
// 	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
// 		return 0, nil, errors.New("shape mismatch between pred and target")
// 	}
// 	if backend.Device() != "cpu" {
// 		return 0, nil, errors.New("CrossEntropy: only CPU implementation available")
// 	}

//		// Softmax 输出与 one-hot target
//		loss := float32(0.0)
//		dout := make([]float32, len(pred.Data))
//		for i := 0; i < len(pred.Data); i++ {
//			p := pred.Data[i]
//			t := target.Data[i]
//			loss += -t * float32(math.Log(float64(p)+1e-8)) // 防止 log(0)
//			dout[i] = p - t
//		}
//		return loss, &tensor.Tensor{
//			Data:   dout,
//			Shape:  pred.Shape,
//			Device: "cpu", // or backend.Device()
//		}, nil
//	}
func CrossEntropy(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
		return 0, nil, errors.New("shape mismatch between pred and target")
	}

	// 根据设备选择实现
	switch backend.Device() {
	case "cpu":
		// CPU 实现
		loss := float32(0.0)
		dout := make([]float32, len(pred.Data))
		for i := 0; i < len(pred.Data); i++ {
			p := pred.Data[i]
			t := target.Data[i]
			loss += -t * float32(math.Log(float64(p)+1e-8))
			dout[i] = p - t
		}
		return loss, &tensor.Tensor{
			Data:   dout,
			Shape:  pred.Shape,
			Device: "cpu",
		}, nil

	case "gpu":
		// GPU 实现
		loss, grad := gpu.CategoricalCrossEntropy(pred, target)
		return loss, grad, nil

	default:
		return 0, nil, errors.New("unsupported device type")
	}
}
