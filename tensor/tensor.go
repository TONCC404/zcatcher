package tensor

type Tensor struct {
	Data  []float32
	Shape []int
	// 可选设备标签：cpu / gpu
	Device string
}

func NewTensor(data []float32, shape []int, device ...string) *Tensor {
	dev := "cpu"
	if len(device) > 0 {
		dev = device[0]
	}
	return &Tensor{
		Data:   data,
		Shape:  shape,
		Device: dev,
	}
}

func NewZeros(shape []int, device ...string) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	dev := "cpu"
	if len(device) > 0 {
		dev = device[0]
	}

	return NewTensor(data, shape, dev)
}
