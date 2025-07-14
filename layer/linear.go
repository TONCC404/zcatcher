package layer

import (
	"errors"
	"zcatcher/tensor/CPU"
)

// ----------------------- Layer -----------------------
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

// ----------------------- Bilinear Layer -----------------------

// Bilinear
// Input: x1: (N, in1), x2: (N, in2)
// Output: (N, out)
// Calculation： y = x1^T * W * x2 + b ，
// 其中 W 是三维张量 (in1, in2, out)，或者存成 (in1*out, in2) 重塑。

type Bilinear struct {
	W, B          *CPU.Tensor
	dW, dB        *CPU.Tensor
	X1, X2        *CPU.Tensor
	in1, in2, out int
}

func NewBilinear(in1, in2, out int) *Bilinear {
	// W的存储方式是 (in1, in2, out)，flatten成一维
	w := make([]float32, in1*in2*out)
	b := make([]float32, out)
	return &Bilinear{
		W:   &CPU.Tensor{Data: w, Shape: []int{in1, in2, out}},
		B:   &CPU.Tensor{Data: b, Shape: []int{1, out}},
		in1: in1,
		in2: in2,
		out: out,
	}
}

// Forward 计算 y_i = x1^T * W[:,:,i] * x2 + b_i
func (b *Bilinear) Forward(x1, x2 *CPU.Tensor) (*CPU.Tensor, error) {
	if len(x1.Shape) != 2 || len(x2.Shape) != 2 {
		return nil, errors.New("Bilinear Forward requires 2D tensors")
	}
	if x1.Shape[0] != x2.Shape[0] {
		return nil, errors.New("Batch size mismatch")
	}
	if x1.Shape[1] != b.in1 || x2.Shape[1] != b.in2 {
		return nil, errors.New("Input feature size mismatch")
	}
	b.X1 = x1
	b.X2 = x2

	N := x1.Shape[0]
	outData := make([]float32, N*b.out)

	// 对每个batch样本i，计算output向量
	for n := 0; n < N; n++ {
		for o := 0; o < b.out; o++ {
			sum := float32(0)
			for i1 := 0; i1 < b.in1; i1++ {
				for i2 := 0; i2 < b.in2; i2++ {
					wIdx := i1*b.in2*b.out + i2*b.out + o
					sum += x1.Data[n*b.in1+i1] * b.W.Data[wIdx] * x2.Data[n*b.in2+i2]
				}
			}
			sum += b.B.Data[o]
			outData[n*b.out+o] = sum
		}
	}
	return &CPU.Tensor{Data: outData, Shape: []int{N, b.out}}, nil
}

// Backward 计算梯度，暂时只计算 dW, dB，返回对 x1 和 x2 的梯度
func (b *Bilinear) Backward(dout *CPU.Tensor) (*CPU.Tensor, *CPU.Tensor, error) {
	if len(dout.Shape) != 2 || dout.Shape[1] != b.out {
		return nil, nil, errors.New("Bilinear Backward dout shape mismatch")
	}
	N := dout.Shape[0]

	// 初始化梯度
	if b.dW == nil {
		b.dW = &CPU.Tensor{Data: make([]float32, b.in1*b.in2*b.out), Shape: []int{b.in1, b.in2, b.out}}
	} else {
		for i := range b.dW.Data {
			b.dW.Data[i] = 0
		}
	}
	if b.dB == nil {
		b.dB = &CPU.Tensor{Data: make([]float32, b.out), Shape: []int{1, b.out}}
	} else {
		for i := range b.dB.Data {
			b.dB.Data[i] = 0
		}
	}

	dx1Data := make([]float32, N*b.in1)
	dx2Data := make([]float32, N*b.in2)

	// 计算dB
	for n := 0; n < N; n++ {
		for o := 0; o < b.out; o++ {
			b.dB.Data[o] += dout.Data[n*b.out+o]
		}
	}

	// 计算dW 和 dx1, dx2
	for n := 0; n < N; n++ {
		for i1 := 0; i1 < b.in1; i1++ {
			for i2 := 0; i2 < b.in2; i2++ {
				for o := 0; o < b.out; o++ {
					idxW := i1*b.in2*b.out + i2*b.out + o
					dwVal := b.X1.Data[n*b.in1+i1] * b.X2.Data[n*b.in2+i2] * dout.Data[n*b.out+o]
					b.dW.Data[idxW] += dwVal

					dx1Data[n*b.in1+i1] += b.W.Data[idxW] * b.X2.Data[n*b.in2+i2] * dout.Data[n*b.out+o]
					dx2Data[n*b.in2+i2] += b.W.Data[idxW] * b.X1.Data[n*b.in1+i1] * dout.Data[n*b.out+o]
				}
			}
		}
	}

	dx1 := &CPU.Tensor{Data: dx1Data, Shape: []int{N, b.in1}}
	dx2 := &CPU.Tensor{Data: dx2Data, Shape: []int{N, b.in2}}

	return dx1, dx2, nil
}

func (b *Bilinear) Params() []*CPU.Tensor {
	return []*CPU.Tensor{b.W, b.B}
}

func (b *Bilinear) Grads() []*CPU.Tensor {
	return []*CPU.Tensor{b.dW, b.dB}
}

// ----------------------- LazyLinear Layer -----------------------
// LazyLinear 类似 Linear，但是初始化时不知道输入维度，直到第一次Forward时才确定W和B的Shape

type LazyLinear struct {
	W, B   *CPU.Tensor
	dW, dB *CPU.Tensor
	X      *CPU.Tensor
	out    int
	in     int
	init   bool
}

func NewLazyLinear(out int) *LazyLinear {
	return &LazyLinear{
		out:  out,
		init: false,
	}
}

func (l *LazyLinear) Forward(x *CPU.Tensor) *CPU.Tensor {
	if !l.init {
		// 第一次调用，根据输入x的Shape初始化参数
		l.in = x.Shape[1]
		wData := make([]float32, l.in*l.out)
		bData := make([]float32, l.out)
		l.W = &CPU.Tensor{Data: wData, Shape: []int{l.in, l.out}}
		l.B = &CPU.Tensor{Data: bData, Shape: []int{1, l.out}}
		l.init = true
	}
	l.X = x
	out := CPU.MatMul(x, l.W)
	out = CPU.AddBias(out, l.B)
	return out
}

func (l *LazyLinear) Backward(dout *CPU.Tensor) *CPU.Tensor {
	if !l.init {
		panic("LazyLinear: backward called before forward")
	}
	l.dW = CPU.MatMul(CPU.Transpose(l.X), dout)
	l.dB = CPU.Sum(dout, 0)
	dx := CPU.MatMul(dout, CPU.Transpose(l.W))
	return dx
}

func (l *LazyLinear) Params() []*CPU.Tensor {
	if !l.init {
		return nil
	}
	return []*CPU.Tensor{l.W, l.B}
}

func (l *LazyLinear) Grads() []*CPU.Tensor {
	if !l.init {
		return nil
	}
	return []*CPU.Tensor{l.dW, l.dB}
}
