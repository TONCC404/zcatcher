package cpu

import (
	"fmt"
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

func (CPUBackend) ZeroPad(input *tensor.Tensor, padding int) *tensor.Tensor {
	batchSize := input.Shape[0]
	channels := input.Shape[1]
	height := input.Shape[2]
	width := input.Shape[3]
	outHeight := height + 2*padding
	outWidth := width + 2*padding
	output := tensor.NewZeros([]int{batchSize, channels, outHeight, outWidth}, input.Device)
	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					inIdx := b*channels*height*width + c*height*width + h*width + w
					outH := h + padding
					outW := w + padding
					outIdx := b*channels*outHeight*outWidth + c*outHeight*outWidth + outH*outWidth + outW
					output.Data[outIdx] = input.Data[inIdx]
				}
			}
		}
	}

	return output
}

func (CPUBackend) Set(dst *tensor.Tensor, start []int, src *tensor.Tensor) {
	if len(start) != len(dst.Shape) || len(src.Shape) != len(dst.Shape) {
		panic("Set: dimension mismatch")
	}

	dstIndices := make([]int, len(start))
	copy(dstIndices, start)
	srcIndices := make([]int, len(start))

	for {
		backend := CPUBackend{}
		dstOffset := backend.Offset(dst.Shape, dstIndices)
		srcOffset := backend.Offset(src.Shape, srcIndices)
		dst.Data[dstOffset] = src.Data[srcOffset]
		for i := len(src.Shape) - 1; i >= 0; i-- {
			srcIndices[i]++
			dstIndices[i]++
			if srcIndices[i] >= src.Shape[i] {
				if i == 0 {
					return
				}
				srcIndices[i] = 0
				dstIndices[i] = start[i]
			} else {
				break
			}
		}
	}
}

func (CPUBackend) Offset(shape []int, indices []int) int {
	if len(shape) != len(indices) {
		panic("Offset: shape and indices length mismatch")
	}
	offset := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= shape[i] {
			panic(fmt.Sprintf("Offset: index %d out of bounds for axis %d", indices[i], i))
		}
		offset += indices[i] * stride
		stride *= shape[i]
	}
	return offset
}

func (CPUBackend) Slice(t *tensor.Tensor, start []int, end []int) *tensor.Tensor {
	if len(start) != len(t.Shape) || len(end) != len(t.Shape) {
		panic("Slice: dimension mismatch")
	}
	outShape := make([]int, len(t.Shape))
	for i := range start {
		if start[i] < 0 || end[i] > t.Shape[i] || start[i] >= end[i] {
			panic(fmt.Sprintf("Slice: invalid slice range on axis %d", i))
		}
		outShape[i] = end[i] - start[i]
	}

	out := tensor.NewZeros(outShape)

	dstIndices := make([]int, len(outShape))
	srcIndices := make([]int, len(start))
	copy(srcIndices, start)

	for {
		backend := CPUBackend{}
		dstOffset := backend.Offset(out.Shape, dstIndices)
		srcOffset := backend.Offset(t.Shape, srcIndices)
		out.Data[dstOffset] = t.Data[srcOffset]
		for i := len(outShape) - 1; i >= 0; i-- {
			dstIndices[i]++
			srcIndices[i]++
			if dstIndices[i] >= outShape[i] {
				if i == 0 {
					return out
				}
				dstIndices[i] = 0
				srcIndices[i] = start[i]
			} else {
				break
			}
		}
	}
}

func (CPUBackend) Reshape(t *tensor.Tensor, newShape []int) *tensor.Tensor {
	oldSize := 1
	for _, dim := range t.Shape {
		oldSize *= dim
	}

	newSize := 1
	for _, dim := range newShape {
		if dim <= 0 {
			panic("Reshape: invalid new shape dimension")
		}
		newSize *= dim
	}

	if oldSize != newSize {
		panic("Reshape: total size mismatch")
	}

	return &tensor.Tensor{
		Data:  t.Data,
		Shape: append([]int(nil), newShape...), // copy newShape
	}
}
