package model

import (
	"zcatcher/layer"
	"zcatcher/tensor/CPU"
)

type Sequential struct {
	Layers []layer.Layer
}

func NewSequential(layers ...layer.Layer) *Sequential {
	return &Sequential{Layers: layers}
}

func (s *Sequential) Forward(x *CPU.Tensor) *CPU.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

func (s *Sequential) Backward(dout *CPU.Tensor) {
	for i := len(s.Layers) - 1; i >= 0; i-- {
		dout = s.Layers[i].Backward(dout)
	}
}
