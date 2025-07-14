package layer

import (
	"math"
	"zcatcher/tensor/CPU"
)

func CrossEntropy(pred, target *CPU.Tensor) (float32, *CPU.Tensor) {
	// 假设 softmax 和 target 都为 one-hot 格式
	loss := float32(0.0)
	dout := make([]float32, len(pred.Data))
	for i := 0; i < len(pred.Data); i++ {
		p := pred.Data[i]
		t := target.Data[i]
		loss += -t * float32(math.Log(float64(p)+1e-8))
		dout[i] = p - t
	}
	return loss, &CPU.Tensor{Data: dout, Shape: pred.Shape}
}
