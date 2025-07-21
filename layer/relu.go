package layer

import (
	"zcatcher/backend/gpu"
	"zcatcher/tensor"
)

type ReLU struct {
	mask    []bool
	x       *tensor.Tensor
	inplace bool
	backend tensor.Backend
}

func NewReLU(backend tensor.Backend, inplace ...bool) *ReLU {
	useInplace := false
	if len(inplace) > 0 {
		useInplace = inplace[0]
	}
	return &ReLU{
		backend: backend,
		inplace: useInplace,
	}
}

func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	r.x = x
	if r.backend.Device() == "cpu" {
		if r.inplace {
			for i := range x.Data {
				if x.Data[i] < 0 {
					x.Data[i] = 0
				}
			}
			return x
		} else {
			out := make([]float32, len(x.Data))
			r.mask = make([]bool, len(x.Data))
			for i, v := range x.Data {
				if v > 0 {
					out[i] = v
					r.mask[i] = true
				}
			}
			return &tensor.Tensor{Data: out, Shape: x.Shape, Device: "cpu"}
		}
	} else {
		// GPU 版本：调用后端
		backend := gpu.GPUBackend{}
		return backend.ReLU(x)
	}
}

func (r *ReLU) Backward(dout *tensor.Tensor) *tensor.Tensor {
	dx := make([]float32, len(dout.Data))
	for i, v := range dout.Data {
		if r.mask[i] {
			dx[i] = v
		}
	}
	return &tensor.Tensor{Data: dx, Shape: dout.Shape}
}

func (r *ReLU) Params() []*tensor.Tensor {
	return nil
}

func (r *ReLU) Grads() []*tensor.Tensor {
	return nil
}
