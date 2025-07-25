package layer

import (
	"zcatcher/tensor"
)

type LeakyReLU struct {
	mask    []bool
	x       *tensor.Tensor
	alpha   float32
	inplace bool
	backend tensor.Backend
}

func NewLeakyReLU(backend tensor.Backend, alpha float32, inplace ...bool) *LeakyReLU {
	useInplace := false
	if len(inplace) > 0 {
		useInplace = inplace[0]
	}
	return &LeakyReLU{
		backend: backend,
		alpha:   alpha,
		inplace: useInplace,
	}
}

func (l *LeakyReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	l.x = x
	out := make([]float32, len(x.Data))
	l.mask = make([]bool, len(x.Data))
	for i, v := range x.Data {
		if v > 0 {
			out[i] = v
			l.mask[i] = true
		} else {
			out[i] = l.alpha * v
			l.mask[i] = false
		}
	}
	return &tensor.Tensor{Data: out, Shape: x.Shape, Device: x.Device}
}

func (l *LeakyReLU) Backward(dout *tensor.Tensor) *tensor.Tensor {
	dx := make([]float32, len(dout.Data))
	for i, v := range dout.Data {
		if l.mask[i] {
			dx[i] = v
		} else {
			dx[i] = v * l.alpha
		}
	}
	return &tensor.Tensor{Data: dx, Shape: dout.Shape}
}

func (l *LeakyReLU) Params() []*tensor.Tensor {
	return nil
}

func (l *LeakyReLU) Grads() []*tensor.Tensor {
	return nil
}
