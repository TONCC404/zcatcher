// backend/cpu.go
package cpu

type CPUBackend struct{}

func NewCPUBackend() *CPUBackend {
	return &CPUBackend{}
}

func (CPUBackend) Device() string {
	return "cpu"
}
