package activations

import (
	"math"
	"zcatcher/tensor"
)

type Sigmoid struct {
	x       *tensor.Tensor
	inplace bool
	backend tensor.Backend
}

func NewSigmoid(backend tensor.Backend, inplace ...bool) *Sigmoid {
	useInplace := false
	if len(inplace) > 0 {
		useInplace = inplace[0]
	}
	return &Sigmoid{
		backend: backend,
		inplace: useInplace,
	}
}

func (s *Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor {
	s.x = x
	out := make([]float32, len(x.Data))
	for i, v := range x.Data {
		out[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	return &tensor.Tensor{Data: out, Shape: x.Shape, Device: x.Device}
}

func (s *Sigmoid) Backward(dout *tensor.Tensor) *tensor.Tensor {
	dx := make([]float32, len(dout.Data))
	for i, v := range dout.Data {
		sigmoidVal := 1 / (1 + math.Exp(-float64(s.x.Data[i])))
		dx[i] = float32(float64(v) * sigmoidVal * (1 - sigmoidVal))
	}
	return &tensor.Tensor{Data: dx, Shape: dout.Shape}
}

func (s *Sigmoid) Params() []*tensor.Tensor {
	return nil
}

func (s *Sigmoid) Grads() []*tensor.Tensor {
	return nil
}
