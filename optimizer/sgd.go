package optimizer

import "zcatcher/layer"

type SGD struct {
	LR float32
}

func NewSGD(lr float32) *SGD {
	return &SGD{LR: lr}
}

func (o *SGD) Step(layers []layer.Layer) {
	for _, l := range layers {
		params := l.Params()
		grads := l.Grads()
		for i := range params {
			for j := range params[i].Data {
				params[i].Data[j] -= o.LR * grads[i].Data[j]
			}
		}
	}
}
