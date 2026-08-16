# go-transformer-engine

[![Go Version](https://img.shields.io/badge/Go-1.26.5-00ADD8?style=flat&logo=go)](https://golang.org)
[![Build & Test](https://github.com/Joanoni/go-transformer-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Joanoni/go-transformer-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`go-transformer-engine` is a high-performance Machine Learning and Transformer Acceleration library built from scratch in **Go (Golang)**, Cgo, and CUDA.

Developed from **First Principles**, this project eliminates external heavy machine learning frameworks (such as PyTorch or TensorFlow) to build a production-grade, modular AI engine.

---

## Project Architecture

<!-- START_PROJECT_TREE -->
```text
go-transformer-engine/
├── LICENSE
├── README.md
├── examples/
│   ├── quickstart/
│   │   └── main.go
│   └── xor/
│       └── main.go
├── go.mod
├── pkg/
│   ├── autograd/
│   │   ├── tape.go
│   │   └── tape_test.go
│   ├── nn/
│   │   ├── activation.go
│   │   ├── activation_test.go
│   │   ├── loss.go
│   │   └── loss_test.go
│   └── tensor/
│       ├── ops.go
│       ├── ops_test.go
│       ├── tensor.go
│       └── tensor_test.go
└── scripts/
    ├── hooks/
    │   └── pre-commit
    └── sync_readme.go
```
<!-- END_PROJECT_TREE -->

---

## Current Engine Capabilities

### 1. N-Dimensional CPU Tensor Engine (`pkg/tensor/`)
- **Flat Memory Layout:** Row-Major (C-contiguous) contiguous 1D slice (`[]float64`) for maximum L1/L2/L3 cache locality and SIMD vectorization.
- **Zero-Copy Memory Views:** High-performance metadata-driven `Transpose` and `Reshape` operations sharing underlying data slices without redundant memory allocations.
- **Strided Index Mapping:** General $N$-dimensional coordinate mapping supporting arbitrary strided views and offsets.
- **Linear Algebra Primitives:** `Add`, `Sub`, `Mul` (Hadamard product), `MatMul` ($M \times K \cdot K \times N$), and higher-order `Apply` mapping.

### 2. Automatic Differentiation Engine (`pkg/autograd/`)
- **GradientTape Execution Context:** Dynamic tape recording utilizing Go's native `context.Context` (`autograd.WithTape(ctx)`).
- **Zero Package Circular Dependencies:** Completely decouples mathematical tensor primitives from graph orchestration via internal `tensor.Recorder` interfaces.
- **Reverse Topological Sort:** $O(N)$ reverse tape traversal naturally evaluating the Chain Rule without runtime Depth-First Search (DFS) overhead.
- **Dual-Path Gradient Accumulation (`AccGrad`):** Fast path direct 1D slice accumulation for contiguous memory layouts alongside coordinate mapping for transposed gradient views.
- **Deterministic Memory Cleanup:** `tape.Clear()` explicitly releases graph closures and intermediate activation references.

### 3. Neural Network Primitives (`pkg/nn/`)
- **Activation Functions:** `Sigmoid` and `ReLU` with automatic analytical backward closures registered on the tape.
- **Loss Metrics:** Mean Squared Error (`MSE`) returning a 1x1 scalar tensor with full automatic differentiation support.

---

## Quickstart & Usage Example

<!-- START_QUICKSTART_CODE -->
```go
package main

import (
	"context"
	"fmt"

	"github.com/Joanoni/go-transformer-engine/pkg/autograd"
	"github.com/Joanoni/go-transformer-engine/pkg/nn"
	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

func main() {
	// 1. Create Tape Context for Forward Pass
	ctx, tape := autograd.WithTape(context.Background())

	// 2. Initialize Input Tensor X [1, 2] and Trainable Weight W [2, 1]
	x, _ := tensor.New(1, 2)
	_ = x.Set(2.0, 0, 0)
	_ = x.Set(3.0, 0, 1)

	w, _ := tensor.New(2, 1)
	w.SetRequiresGrad(true) // Enable gradient tracking
	_ = w.Set(0.5, 0, 0)
	_ = w.Set(-1.2, 1, 0)

	// 3. Forward Pass: Output = Sigmoid(X * W)
	h, _ := tensor.MatMul(ctx, x, w)
	yHat, _ := nn.Sigmoid(ctx, h)

	// Target Tensor Y [1, 1]
	y, _ := tensor.New(1, 1)
	_ = y.Set(1.0, 0, 0)

	// 4. Compute Loss
	loss, _ := nn.MSE(ctx, yHat, y)
	fmt.Printf("Forward Loss: %.6f\n", loss.Data()[0])

	// 5. Automatic Backward Pass via Autograd Tape
	if err := tape.Backward(loss); err != nil {
		panic(err)
	}

	// 6. Access Automatically Computed Gradients (dL/dW)
	fmt.Printf("dL/dW[0,0]: %.6f\n", w.Grad().Data()[0])
	fmt.Printf("dL/dW[1,0]: %.6f\n", w.Grad().Data()[1])

	// 7. Clean Tape for Next Epoch
	tape.Clear()
}
```
<!-- END_QUICKSTART_CODE -->

---

## Running Tests & Demonstrations

### Run All Unit Tests
Force clean test execution across all packages disabling Go test caching:

```powershell
go test -count=1 -v ./...
```

### Run XOR Demonstration
Train an MLP (2-2-1) from scratch to solve the non-linear XOR gate using automatic differentiation:

```powershell
go run ./examples/xor/main.go
```

**Verified Output:**
<!-- START_XOR_OUTPUT -->
```text
==========================================================================
 go-transformer-engine: XOR Problem Demo (Phase 3 - Autograd Engine)
==========================================================================
Training MLP (2-2-1) for 20000 epochs with Learning Rate = 1.00...

Epoch     1/20000 | MSE Loss: 0.284102
Epoch  2000/20000 | MSE Loss: 0.000525
Epoch  4000/20000 | MSE Loss: 0.000253
Epoch  6000/20000 | MSE Loss: 0.000165
Epoch  8000/20000 | MSE Loss: 0.000122
Epoch 10000/20000 | MSE Loss: 0.000097
Epoch 12000/20000 | MSE Loss: 0.000080
Epoch 14000/20000 | MSE Loss: 0.000068
Epoch 16000/20000 | MSE Loss: 0.000060
Epoch 18000/20000 | MSE Loss: 0.000053
Epoch 20000/20000 | MSE Loss: 0.000047

--------------------------------------------------------------------------
Initial MSE Loss: 0.284102
Final MSE Loss:   0.000047
--------------------------------------------------------------------------

Final XOR Predictions vs Ground Truth Targets:
Input: [0, 0] | Target: 0 | Prediction: 0.0062
Input: [0, 1] | Target: 1 | Prediction: 0.9927
Input: [1, 0] | Target: 1 | Prediction: 0.9927
Input: [1, 1] | Target: 0 | Prediction: 0.0065
==========================================================================
```
<!-- END_XOR_OUTPUT -->

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.