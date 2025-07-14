package layer

import (
	"zcatcher/tensor/CPU"
)

type Linear struct {
	W, B   *CPU.Tensor
	dW, dB *CPU.Tensor
	X      *CPU.Tensor
}

func NewLinear(in, out int) *Linear {
	w := make([]float32, in*out)
	b := make([]float32, out)
	return &Linear{
		W: &CPU.Tensor{Data: w, Shape: []int{in, out}},
		B: &CPU.Tensor{Data: b, Shape: []int{1, out}},
	}
}

func (l *Linear) Forward(x *CPU.Tensor) *CPU.Tensor {
	l.X = x
	out := CPU.MatMul(x, l.W)
	out = CPU.AddBias(out, l.B)
	return out
}

func (l *Linear) Backward(dout *CPU.Tensor) *CPU.Tensor {
	l.dW = CPU.MatMul(CPU.Transpose(l.X), dout)
	l.dB = CPU.Sum(dout, 0)
	dx := CPU.MatMul(dout, CPU.Transpose(l.W))
	return dx
}

func (l *Linear) Params() []*CPU.Tensor {
	return []*CPU.Tensor{l.W, l.B}
}

func (l *Linear) Grads() []*CPU.Tensor {
	return []*CPU.Tensor{l.dW, l.dB}
}
