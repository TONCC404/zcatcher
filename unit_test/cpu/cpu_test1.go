package cpu

import (
	"fmt"
	"zcatcher/backend/cpu"
	"zcatcher/layer"
	"zcatcher/model"
	"zcatcher/optimizer"
	"zcatcher/tensor"
)

func main() {
	x := tensor.NewTensor([]float32{
		1, 2,
		3, 4,
	}, []int{2, 2})
	y := tensor.NewTensor([]float32{
		0, 1,
		1, 0,
	}, []int{2, 2})
	backend := cpu.NewCPUBackend()

	net := model.NewSequential(
		layer.NewLinear(2, 3, backend),
		layer.NewReLU(backend),
		layer.NewLinear(3, 2, backend),
	)

	op := optimizer.NewSGD(0.01)

	for epoch := 0; epoch < 10; epoch++ {
		out := net.Forward(x)
		loss, dout, error := layer.CrossEntropy(out, y, backend)
		if error != nil {
			fmt.Printf("Epoch %d error: %v\n", epoch, error)
			continue
		}
		net.Backward(dout)
		op.Step(net.Layers)
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, loss)
	}
}
