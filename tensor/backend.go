package tensor

type Backend interface {
	MatMul(a, b *Tensor) *Tensor
	Sum(t *Tensor, axis int) *Tensor
	AddBias(mat, bias *Tensor) *Tensor
	Transpose(t *Tensor) *Tensor
	ZeroPad(input *Tensor, padding int) *Tensor
	Set(index []int, value float32) *Tensor
	Slice(t *Tensor, start []int, end []int) *Tensor
	Offset(shape []int, indices []int) int
	Reshape(t *Tensor, newShape []int) *Tensor
	Device() string
}
