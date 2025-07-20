package tensor

type Tensor struct {
	Data  []float32
	Shape []int
	// 可选设备标签：cpu / gpu
	Device string
}

func NewTensor(data []float32, shape []int) *Tensor {
	return &Tensor{
		Data:   data,
		Shape:  shape,
		Device: "cpu", // 默认设为cpu
	}
}
