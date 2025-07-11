package layer

import "zcatcher/tensor"

type Linear struct {
	W, B   *tensor.Tensor
	dW, dB *tensor.Tensor
	X      *tensor.Tensor
}

func NewLinear(in, out int) *Linear {
	w := make([]float32, in*out)
	b := make([]float32, out)
	return &Linear{
		W: &tensor.Tensor{Data: w, Shape: []int{in, out}},
		B: &tensor.Tensor{Data: b, Shape: []int{1, out}},
	}
}

func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	l.X = x
	out := tensor.MatMul(x, l.W)
	// 简略：没有实现 Add Bias 的广播
	return out
}

func (l *Linear) Backward(dout *tensor.Tensor) *tensor.Tensor {
	l.dW = tensor.MatMul(tensor.Transpose(l.X), dout)
	l.dB = tensor.Sum(dout, 0)
	dx := tensor.MatMul(dout, tensor.Transpose(l.W))
	return dx
}

func (l *Linear) Params() []*tensor.Tensor {
	return []*tensor.Tensor{l.W, l.B}
}

func (l *Linear) Grads() []*tensor.Tensor {
	return []*tensor.Tensor{l.dW, l.dB}
}
