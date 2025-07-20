package tensor

var currentBackend Backend = nil

func SetBackend(b Backend) {
	currentBackend = b
}

func GetBackend() Backend {
	if currentBackend == nil {
		panic("No backend set")
	}
	return currentBackend
}
