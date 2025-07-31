# zcatcher

zcatcher is a deep learning framework implemented in Go, supporting both CPU and GPU (CUDA) backends. It is suitable for research and engineering applications.

## Features

- Tensor operations with automatic CPU/GPU backend selection
- Common neural network layers (e.g., Linear, Conv2D) and activation functions (ReLU, Sigmoid, Tanh, etc.)
- Supports forward and backward propagation, automatic differentiation
- Custom loss functions and optimizers (e.g., SGD)
- Clean and extensible architecture

## Directory Structure

```
backend/      # Backend implementations (CPU/GPU/CUDA)
layer/        # Neural network layers and activations
conv/         # Convolution-related implementations
model/        # Network structures (e.g., Sequential)
optimizer/    # Optimizer implementations
tensor/       # Tensor and dispatcher
unit_test/    # Unit tests
main.go       # Example entry point
go.mod        # Go module dependencies
```

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/TONCC404/zcatcher.git
   cd zcatcher
   ```
2. Build and run:
   ```bash
   go run main.go
   ```
3. Run tests:
   ```bash
   go test ./unit_test/...
   ```

> **Note**: To use the GPU backend, please ensure the CUDA toolkit is properly installed and the relevant `.cu` files are compiled.

## Contribution

Issues, pull requests, and suggestions are welcome!

## License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 TONCC404

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
