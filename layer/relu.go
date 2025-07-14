package layer

import (
	"zcatcher/tensor/CPU"
)

type ReLU struct {
	mask []bool
	x    *CPU.Tensor
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *CPU.Tensor) *CPU.Tensor {
	r.x = x
	out := make([]float32, len(x.Data))
	r.mask = make([]bool, len(x.Data))
	for i, v := range x.Data {
		if v > 0 {
			out[i] = v
			r.mask[i] = true
		}
	}
	return &CPU.Tensor{Data: out, Shape: x.Shape}
}

func (r *ReLU) Backward(dout *CPU.Tensor) *CPU.Tensor {
	dx := make([]float32, len(dout.Data))
	for i, v := range dout.Data {
		if r.mask[i] {
			dx[i] = v
		}
	}
	return &CPU.Tensor{Data: dx, Shape: dout.Shape}
}

func (r *ReLU) Params() []*CPU.Tensor {
	return nil
}

func (r *ReLU) Grads() []*CPU.Tensor {
	return nil
}
