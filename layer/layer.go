package layer

import (
	"zcatcher/tensor/CPU"
)

type Layer interface {
	Forward(x *CPU.Tensor) *CPU.Tensor
	Backward(dout *CPU.Tensor) *CPU.Tensor
	Params() []*CPU.Tensor
	Grads() []*CPU.Tensor
}
