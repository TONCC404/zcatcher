package tensor

type Tensor struct {
	Data  []float32
	Shape []int
	// 可选设备标签：cpu / gpu
	Device string
}