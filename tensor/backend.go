package tensor

type Backend interface {
	MatMul(a, b *Tensor) *Tensor
	Sum(t *Tensor, axis int) *Tensor
	AddBias(mat, bias *Tensor) *Tensor
	Transpose(t *Tensor) *Tensor
	ZeroPad(input *Tensor, padding int) *Tensor
	Set(t *Tensor, value float32, indices ...int) *Tensor
	Get(t *Tensor, indices ...int) float32
	Slice(t *Tensor, start []int, end []int) *Tensor
	Offset(shape []int, indices []int) int
	Reshape(t *Tensor, newShape []int) *Tensor
	Device() string
}
