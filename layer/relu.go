package layer

import "zcatcher/tensor"

type ReLU struct {
	mask []bool
	x    *tensor.Tensor
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	r.x = x
	out := make([]float32, len(x.Data))
	r.mask = make([]bool, len(x.Data))
	for i, v := range x.Data {
		if v > 0 {
			out[i] = v
			r.mask[i] = true
		}
	}
	return &tensor.Tensor{Data: out, Shape: x.Shape}
}

func (r *ReLU) Backward(dout *tensor.Tensor) *tensor.Tensor {
	dx := make([]float32, len(dout.Data))
	for i, v := range dout.Data {
		if r.mask[i] {
			dx[i] = v
		}
	}
	return &tensor.Tensor{Data: dx, Shape: dout.Shape}
}

func (r *ReLU) Params() []*tensor.Tensor {
	return nil
}

func (r *ReLU) Grads() []*tensor.Tensor {
	return nil
}
