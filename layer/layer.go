package layer

import "zcatcher/tensor"

type Layer interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Backward(dout *tensor.Tensor) *tensor.Tensor
	Params() []*tensor.Tensor
	Grads() []*tensor.Tensor
}
