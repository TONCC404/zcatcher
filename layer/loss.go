package layer

import (
	"errors"
	"math"
	"zcatcher/backend/gpu"
	"zcatcher/tensor"
)

func CrossEntropy(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
		return 0, nil, errors.New("shape mismatch between pred and target")
	}
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

func MSE(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
		return 0, nil, errors.New("shape mismatch between pred and target")
	}
	switch backend.Device() {
	case "cpu":
		loss := float32(0.0)
		dout := make([]float32, len(pred.Data))
		for i := 0; i < len(pred.Data); i++ {
			diff := pred.Data[i] - target.Data[i]
			loss += diff * diff
			dout[i] = 2 * diff
		}
		return loss, &tensor.Tensor{
			Data:   dout,
			Shape:  pred.Shape,
			Device: "cpu",
		}, nil

	case "gpu":
		loss, grad := gpu.MSELoss(pred, target)
		return loss, grad, nil

	default:
		return 0, nil, errors.New("unsupported device type")
	}
}

func MAE(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
		return 0, nil, errors.New("shape mismatch between pred and target")
	}
	switch backend.Device() {
	case "cpu":
		loss := float32(0.0)
		dout := make([]float32, len(pred.Data))
		for i := 0; i < len(pred.Data); i++ {
			diff := pred.Data[i] - target.Data[i]
			loss += float32(math.Abs(float64(diff)))
			if math.Signbit(float64(diff)) {
				dout[i] = -1.0
			} else {
				dout[i] = 1.0
			}
		}
		return loss, &tensor.Tensor{
			Data:   dout,
			Shape:  pred.Shape,
			Device: "cpu",
		}, nil

	case "gpu":
		loss, grad := gpu.MAELoss(pred, target)
		return loss, grad, nil

	default:
		return 0, nil, errors.New("unsupported device type")
	}
}

func BinaryCrossEntropy(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
		return 0, nil, errors.New("shape mismatch between pred and target")
	}
	switch backend.Device() {
	case "cpu":
		loss := float32(0.0)
		dout := make([]float32, len(pred.Data))
		for i := 0; i < len(pred.Data); i++ {
			p := pred.Data[i]
			t := target.Data[i]
			loss += -t*float32(math.Log(float64(p)+1e-8)) - (1-t)*float32(math.Log(float64(1-p)+1e-8))
			dout[i] = p - t
		}
		return loss, &tensor.Tensor{
			Data:   dout,
			Shape:  pred.Shape,
			Device: "cpu",
		}, nil

	case "gpu":
		loss, grad := gpu.BinaryCrossEntropy(pred, target)
		return loss, grad, nil

	default:
		return 0, nil, errors.New("unsupported device type")
	}
}

func SmoothL1(pred, target *tensor.Tensor, backend tensor.Backend) (float32, *tensor.Tensor, error) {
	if pred.Shape[0] != target.Shape[0] || pred.Shape[1] != target.Shape[1] {
		return 0, nil, errors.New("shape mismatch between pred and target")
	}
	switch backend.Device() {
	case "cpu":
		loss := float32(0.0)
		dout := make([]float32, len(pred.Data))
		for i := 0; i < len(pred.Data); i++ {
			diff := math.Abs(float64(pred.Data[i] - target.Data[i]))
			if diff < 1.0 {
				loss += float32(0.5 * diff * diff)
				dout[i] = float32(pred.Data[i] - target.Data[i])
			} else {
				loss += float32(diff - 0.5)
				if math.Signbit(float64(pred.Data[i] - target.Data[i])) {
					dout[i] = -1.0
				} else {
					dout[i] = 1.0
				}
			}
		}
		return loss, &tensor.Tensor{
			Data:   dout,
			Shape:  pred.Shape,
			Device: "cpu",
		}, nil

	case "gpu":
		loss, grad := gpu.SmoothL1Loss(pred, target)
		return loss, grad, nil

	default:
		return 0, nil, errors.New("unsupported device type")
	}
}
