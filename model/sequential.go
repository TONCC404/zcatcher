package model

import (
	"zcatcher/layer"
	"zcatcher/tensor"
)

type Sequential struct {
	Layers []layer.Layer
}

func NewSequential(layers ...layer.Layer) *Sequential {
	return &Sequential{Layers: layers}
}

func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

func (s *Sequential) Backward(dout *tensor.Tensor) {
	for i := len(s.Layers) - 1; i >= 0; i-- {
		dout = s.Layers[i].Backward(dout)
	}
}
