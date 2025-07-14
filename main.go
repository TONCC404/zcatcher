package main

import (
	"fmt"
	"zcatcher/layer"
	"zcatcher/model"
	"zcatcher/optimizer"
	"zcatcher/tensor/CPU"
)

func main() {
	x := CPU.New([]float32{
		1, 2,
		3, 4,
	}, []int{2, 2})
	y := CPU.New([]float32{
		0, 1,
		1, 0,
	}, []int{2, 2})

	net := model.NewSequential(
		layer.NewLinear(2, 3),
		layer.NewReLU(),
		layer.NewLinear(3, 2),
	)

	op := optimizer.NewSGD(0.01)

	for epoch := 0; epoch < 10; epoch++ {
		out := net.Forward(x)
		loss, dout := layer.CrossEntropy(out, y)
		net.Backward(dout)
		op.Step(net.Layers)
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, loss)
	}
}
