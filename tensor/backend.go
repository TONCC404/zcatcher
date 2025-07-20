package tensor

type Backend interface {
	MatMul(a, b *Tensor) *Tensor
	Sum(t *Tensor, axis int) *Tensor
	AddBias(mat, bias *Tensor) *Tensor
	Transpose(t *Tensor) *Tensor
	Device() string
}
