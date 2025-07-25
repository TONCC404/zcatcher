package layer

import (
	"zcatcher/tensor"
)

type Activation interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Backward(dout *tensor.Tensor) *tensor.Tensor
	Params() []*tensor.Tensor
	Grads() []*tensor.Tensor
}

func NewActivationFunction(activationType string, backend tensor.Backend, inplace ...bool) Activation {
	switch activationType {
	case "ReLU":
		return NewReLU(backend, inplace...)
	case "Sigmoid":
		return NewSigmoid(backend, inplace...)
	case "LeakyReLU":
		return NewLeakyReLU(backend, 0.01, inplace...)
	case "Tanh":
		return NewTanh(backend, inplace...)
	// 你可以在这里继续添加更多的激活函数
	default:
		return nil
	}
}
