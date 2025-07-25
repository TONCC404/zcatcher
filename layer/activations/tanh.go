package activations

import (
	"math"
	"zcatcher/tensor"
)

type Tanh struct {
	x       *tensor.Tensor
	inplace bool
	backend tensor.Backend
}

func NewTanh(backend tensor.Backend, inplace ...bool) *Tanh {
	useInplace := false
	if len(inplace) > 0 {
		useInplace = inplace[0]
	}
	return &Tanh{
		backend: backend,
		inplace: useInplace,
	}
}

func (t *Tanh) Forward(x *tensor.Tensor) *tensor.Tensor {
	t.x = x
	out := make([]float32, len(x.Data))
	for i, v := range x.Data {
		out[i] = float32(math.Tanh(float64(v)))
	}
	return &tensor.Tensor{Data: out, Shape: x.Shape, Device: x.Device}
}

func (t *Tanh) Backward(dout *tensor.Tensor) *tensor.Tensor {
	dx := make([]float32, len(dout.Data))
	for i, v := range dout.Data {
		tanhVal := math.Tanh(float64(t.x.Data[i]))
		dx[i] = v * float32(1-tanhVal*tanhVal)
	}
	return &tensor.Tensor{Data: dx, Shape: dout.Shape}
}

func (t *Tanh) Params() []*tensor.Tensor {
	return nil
}

func (t *Tanh) Grads() []*tensor.Tensor {
	return nil
}
