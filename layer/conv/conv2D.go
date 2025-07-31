package layer

import (
	"errors"
	"zcatcher/tensor"
)

type Conv2D struct {
	Filters *tensor.Tensor
	Bias    *tensor.Tensor
	Stride  int
	Padding int
}

func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int, backend tensor.Backend) *Conv2D {
	// Initialize filters and biases
	filters := tensor.NewTensor(nil, []int{outChannels, inChannels, kernelSize, kernelSize}, backend.Device())
	bias := tensor.NewTensor(nil, []int{outChannels}, backend.Device())

	return &Conv2D{
		Filters: filters,
		Bias:    bias,
		Stride:  stride,
		Padding: padding,
	}
}

func (c *Conv2D) Forward(input *tensor.Tensor, backend tensor.Backend) (*tensor.Tensor, error) {
	if len(input.Shape) != 4 {
		return nil, errors.New("Conv2D input must be 4D [batch, channels, height, width]")
	}

	batchSize := input.Shape[0]
	inChannels := input.Shape[1]
	inHeight := input.Shape[2]
	inWidth := input.Shape[3]
	kernelSize := c.Filters.Shape[2]
	outChannels := c.Filters.Shape[0]

	// Output dimensions
	outHeight := (inHeight+2*c.Padding-kernelSize)/c.Stride + 1
	outWidth := (inWidth+2*c.Padding-kernelSize)/c.Stride + 1

	// Pad the input
	paddedInput := backend.ZeroPad(input, c.Padding)

	// Output tensor
	output := tensor.NewZeros([]int{batchSize, outChannels, outHeight, outWidth})

	// For each batch
	for b := 0; b < batchSize; b++ {
		// For each output position
		for i := 0; i < outHeight; i++ {
			for j := 0; j < outWidth; j++ {
				// Extract region from input
				region := backend.Slice(
					paddedInput,
					[]int{b, 0, i * c.Stride, j * c.Stride},
					[]int{b + 1, inChannels, i*c.Stride + kernelSize, j*c.Stride + kernelSize},
				) // shape: [1, inChannels, kernelSize, kernelSize]

				// Reshape region to [inChannels * kernelSize * kernelSize, 1]
				regionFlat := backend.Reshape(region, []int{inChannels * kernelSize * kernelSize, 1})

				// Reshape filters to [outChannels, inChannels * kernelSize * kernelSize]
				filtersFlat := backend.Reshape(c.Filters, []int{outChannels, inChannels * kernelSize * kernelSize})

				// MatMul: [outChannels, 1]
				outPatch := backend.MatMul(filtersFlat, regionFlat)

				// Add bias
				outPatch = backend.AddBias(outPatch, c.Bias)

				// Set output
				for oc := 0; oc < outChannels; oc++ {
					output.Data[backend.Offset([]int{b, oc, i, j}, output.Shape)] = outPatch.Data[oc]
				}
			}
		}
	}

	return output, nil
}

func (c *Conv2D) Backward(input *tensor.Tensor, gradOutput *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	batchSize := input.Shape[0]
	inChannels := input.Shape[1]
	inHeight := input.Shape[2]
	inWidth := input.Shape[3]

	kernelSize := c.KernelSize
	stride := c.Stride
	padding := c.Padding
	outChannels := c.OutChannels

	// Padding input and prepare gradInput
	paddedInput := c.Backend.ZeroPad(input, padding)
	gradPaddedInput := tensor.NewZeros(paddedInput.Shape)

	gradFilters := tensor.NewZeros(c.Filters.Shape) // [outChannels, inChannels, kH, kW]
	gradBias := tensor.NewZeros([]int{outChannels})

	outHeight := gradOutput.Shape[2]
	outWidth := gradOutput.Shape[3]

	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outHeight; oh++ {
				for ow := 0; ow < outWidth; ow++ {
					dout := gradOutput.Get(b, oc, oh, ow)
					gradBias.Set(gradBias.Get(oc)+dout, oc)

					for ic := 0; ic < inChannels; ic++ {
						for kh := 0; kh < kernelSize; kh++ {
							for kw := 0; kw < kernelSize; kw++ {
								inH := oh*stride + kh
								inW := ow*stride + kw
								inVal := paddedInput.Get(b, ic, inH, inW)

								// grad wrt filter
								old := gradFilters.Get(oc, ic, kh, kw)
								gradFilters.Set(old+inVal*dout, oc, ic, kh, kw)

								// grad wrt input
								weight := c.Filters.Get(oc, ic, kh, kw)
								oldGrad := gradPaddedInput.Get(b, ic, inH, inW)
								gradPaddedInput.Set(oldGrad+weight*dout, b, ic, inH, inW)
							}
						}
					}
				}
			}
		}
	}

	// Unpad gradInput to match input shape
	gradInput := c.Backend.Slice(gradPaddedInput,
		[]int{0, 0, padding, padding},
		[]int{batchSize, inChannels, padding + inHeight, padding + inWidth},
	)

	return gradInput, gradFilters, gradBias
}
