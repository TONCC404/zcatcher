package CPU

//import "github.com/matrixorigin/simdcpu"

type Tensor struct {
	Data  []float32
	Shape []int
}

func NewTensor(data []float32, shape []int) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}

func MatMul(a, b *Tensor) *Tensor {
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
	return &Tensor{Data: out, Shape: []int{m, n}}
}

func Transpose(t *Tensor) *Tensor {
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
	return &Tensor{Data: out, Shape: []int{cols, rows}}
}

func Sum(t *Tensor, axis int) *Tensor {
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
		return &Tensor{Data: out, Shape: []int{1, cols}}
	} else if axis == 1 {
		out = make([]float32, rows)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[i] += t.Data[i*cols+j]
			}
		}
		return &Tensor{Data: out, Shape: []int{rows, 1}}
	} else {
		panic("Invalid axis")
	}
}

func AddBias(mat *Tensor, bias *Tensor) *Tensor {
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
	return &Tensor{Data: out, Shape: mat.Shape}
}

// todo: SIMDCPU for Data Level Parallelism Optimization
// for SIMD optimize, CPU based
//func AddBiasWithSIMD(mat *Tensor, bias *Tensor) *Tensor {
//	if len(mat.Shape) != 2 || len(bias.Shape) != 2 {
//		panic("AddBias only supports 2D tensors")
//	}
//	rows, cols := mat.Shape[0], mat.Shape[1]
//	if bias.Shape[1] != cols {
//		panic("Bias shape mismatch")
//	}
//	out := make([]float32, len(mat.Data))
//	for i := 0; i < rows; i++ {
//		slice := mat.Data[i*cols : (i+1)*cols]
//		outSlice := simdcpu.AddFloat32(slice, bias.Data) // SIMD 加速逐行加法
//		copy(out[i*cols:(i+1)*cols], outSlice)
//	}
//	return &Tensor{Data: out, Shape: mat.Shape}
//}
