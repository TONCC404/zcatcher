package cpu

import (
	"math"
	"zcatcher/tensor"
)

func maybeAlloc(input *tensor.Tensor, inplace bool) *tensor.Tensor {
	if inplace {
		return input
	}
	return input.Clone()
}

func (b CPUBackend) ReLU(input *tensor.Tensor, inplace bool) *tensor.Tensor {
	out := maybeAlloc(input, inplace)
	mask := make([]byte, len(input.Data))

	for i, v := range input.Data {
		if v > 0 {
			out.Data[i] = v
			mask[i] = 1
		} else {
			out.Data[i] = 0
			mask[i] = 0
		}
	}
	out.Mask = mask
	return out
}

func (b CPUBackend) ReLUBackward(gradOutput *tensor.Tensor, mask *Mask) *tensor.Tensor {
	gradInput := gradOutput.Clone()
	for i := range gradOutput.Data {
		if mask.Data[i] == 0 {
			gradInput.Data[i] = 0
		}
	}
	return gradInput
}

func (b CPUBackend) Sigmoid(input *tensor.Tensor, inplace bool) *tensor.Tensor {
	out := maybeAlloc(input, inplace)
	for i, v := range input.Data {
		out.Data[i] = 1.0 / (1.0 + float32(math.Exp(-float64(v))))
	}
	return out
}

func (b CPUBackend) SigmoidBackward(output, gradOutput *tensor.Tensor) *tensor.Tensor {
	gradInput := NewTensorLike(output)
	for i := range output.Data {
		y := output.Data[i]
		gradInput.Data[i] = gradOutput.Data[i] * y * (1 - y)
	}
	return gradInput
}

// Tanh
func (b CPUBackend) Tanh(input *tensor.Tensor, inplace bool) *tensor.Tensor {
	out := maybeAlloc(input, inplace)
	for i, v := range input.Data {
		out.Data[i] = float32(math.Tanh(float64(v)))
	}
	return out
}

func (b CPUBackend) TanhBackward(output, gradOutput *tensor.Tensor) *tensor.Tensor {
	gradInput := NewTensorLike(output)
	for i, y := range output.Data {
		gradInput.Data[i] = gradOutput.Data[i] * (1 - y*y)
	}
	return gradInput
}

// LeakyReLU
func (b CPUBackend) LeakyReLU(input *tensor.Tensor, alpha float32, inplace bool) *tensor.Tensor {
	out := maybeAlloc(input, inplace)
	for i, v := range input.Data {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = alpha * v
		}
	}
	return out
}

func (b CPUBackend) LeakyReLUBackward(input, gradOutput *tensor.Tensor, alpha float32) *tensor.Tensor {
	gradInput := NewTensorLike(input)
	for i, v := range input.Data {
		if v > 0 {
			gradInput.Data[i] = gradOutput.Data[i]
		} else {
			gradInput.Data[i] = gradOutput.Data[i] * alpha
		}
	}
	return gradInput
}

// ELU
func (b CPUBackend) ELU(input *tensor.Tensor, alpha float32, inplace bool) *tensor.Tensor {
	out := maybeAlloc(input, inplace)
	for i, v := range input.Data {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = alpha * (float32(math.Exp(float64(v))) - 1)
		}
	}
	return out
}

func (b CPUBackend) ELUBackward(input, gradOutput *tensor.Tensor, alpha float32) *tensor.Tensor {
	gradInput := NewTensorLike(input)
	for i, v := range input.Data {
		if v > 0 {
			gradInput.Data[i] = gradOutput.Data[i]
		} else {
			gradInput.Data[i] = gradOutput.Data[i] * alpha * float32(math.Exp(float64(v)))
		}
	}
	return gradInput
}
