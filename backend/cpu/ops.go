package cpu

import (
	"zcatcher/tensor"
)

func (CPUBackend) MatMul(a, b *tensor.Tensor) *tensor.Tensor {
	m, k := a.Shape[0], a.Shape[1]
	n := b.Shape[1]
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			for t := 0; t < k; t++ {
				out[i*n+j] += a.Data[i*k+t] * b.Data[t*n+j]
			}
		}
	}
	return &tensor.Tensor{Data: out, Shape: []int{m, n}, Device: "cpu"}
}

func (CPUBackend) Transpose(t *tensor.Tensor) *tensor.Tensor {
	if len(t.Shape) != 2 {
		panic("Transpose only supports 2D tensors")
	}
	rows, cols := t.Shape[0], t.Shape[1]
	out := make([]float32, len(t.Data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[j*rows+i] = t.Data[i*cols+j]
		}
	}
	return &tensor.Tensor{Data: out, Shape: []int{cols, rows}, Device: "cpu"}
}

func (CPUBackend) Sum(t *tensor.Tensor, axis int) *tensor.Tensor {
	if len(t.Shape) != 2 {
		panic("Sum only supports 2D tensors")
	}
	rows, cols := t.Shape[0], t.Shape[1]
	var out []float32
	if axis == 0 {
		out = make([]float32, cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[j] += t.Data[i*cols+j]
			}
		}
		return &tensor.Tensor{Data: out, Shape: []int{1, cols}, Device: "cpu"}
	} else if axis == 1 {
		out = make([]float32, rows)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[i] += t.Data[i*cols+j]
			}
		}
		return &tensor.Tensor{Data: out, Shape: []int{rows, 1}, Device: "cpu"}
	} else {
		panic("Invalid axis")
	}
}

func (CPUBackend) AddBias(mat, bias *tensor.Tensor) *tensor.Tensor {
	if len(mat.Shape) != 2 || len(bias.Shape) != 2 {
		panic("AddBias only supports 2D tensors")
	}
	rows, cols := mat.Shape[0], mat.Shape[1]
	if bias.Shape[1] != cols {
		panic("Bias shape mismatch")
	}
	out := make([]float32, len(mat.Data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i*cols+j] = mat.Data[i*cols+j] + bias.Data[j]
		}
	}
	return &tensor.Tensor{Data: out, Shape: mat.Shape, Device: "cpu"}
}
