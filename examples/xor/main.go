package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/Joanoni/go-transformer-engine/nn"
	"github.com/Joanoni/go-transformer-engine/tensor"
)

// initRandomTensor creates a Tensor with the given shape populated with random float64 values in [-1.0, 1.0].
func initRandomTensor(shape ...int) (*tensor.Tensor, error) {
	ts, err := tensor.New(shape...)
	if err != nil {
		return nil, err
	}

	for i := 0; i < ts.Size(); i++ {
		// Random float in [-1.0, 1.0]
		val := rand.Float64()*2.0 - 1.0
		ts.Data()[i] = val
	}
	return ts, nil
}

// addBias broadcasts a 1xN bias row vector across an MxN matrix.
// Resulting shape: [M, N]
func addBias(mat, bias *tensor.Tensor) (*tensor.Tensor, error) {
	matShape := mat.Shape()
	biasShape := bias.Shape()

	if len(matShape) != 2 || len(biasShape) != 2 {
		return nil, fmt.Errorf("addBias requires 2D matrices, got ranks %d and %d", len(matShape), len(biasShape))
	}

	m, n := matShape[0], matShape[1]
	if biasShape[1] != n {
		return nil, fmt.Errorf("bias dimension mismatch: matrix cols %d vs bias cols %d", n, biasShape[1])
	}

	out, err := tensor.New(m, n)
	if err != nil {
		return nil, err
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			matVal, _ := mat.At(i, j)
			biasVal, _ := bias.At(0, j)
			if err := out.Set(matVal+biasVal, i, j); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// sumRows sums gradient values across the batch dimension (axis 0) to produce a 1xN bias gradient.
func sumRows(grad *tensor.Tensor) (*tensor.Tensor, error) {
	shape := grad.Shape()
	m, n := shape[0], shape[1]

	out, err := tensor.New(1, n)
	if err != nil {
		return nil, err
	}

	for j := 0; j < n; j++ {
		var colSum float64
		for i := 0; i < m; i++ {
			val, _ := grad.At(i, j)
			colSum += val
		}
		if err := out.Set(colSum, 0, j); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// updateWeights updates weight parameters using Stochastic Gradient Descent (SGD):
//
//	W_new = W - lr * dW
func updateWeights(param, grad *tensor.Tensor, lr float64) (*tensor.Tensor, error) {
	scaledGrad, err := tensor.Apply(grad, func(g float64) float64 {
		return g * lr
	})
	if err != nil {
		return nil, err
	}
	return tensor.Sub(param, scaledGrad)
}

func main() {
	fmt.Println("==================================================================")
	fmt.Println("  go-transformer-engine: XOR Multi-Layer Perceptron (2-2-1 MLP)   ")
	fmt.Println("==================================================================")

	// 1. Prepare XOR Dataset
	// Inputs X: 4 samples, 2 features -> Shape [4, 2]
	x, err := tensor.New(4, 2)
	if err != nil {
		log.Fatalf("failed to create input tensor: %v", err)
	}
	_ = x.Set(0.0, 0, 0)
	_ = x.Set(0.0, 0, 1) // [0, 0]
	_ = x.Set(0.0, 1, 0)
	_ = x.Set(1.0, 1, 1) // [0, 1]
	_ = x.Set(1.0, 2, 0)
	_ = x.Set(0.0, 2, 1) // [1, 0]
	_ = x.Set(1.0, 3, 0)
	_ = x.Set(1.0, 3, 1) // [1, 1]

	// Targets Y: 4 samples, 1 label -> Shape [4, 1]
	y, err := tensor.New(4, 1)
	if err != nil {
		log.Fatalf("failed to create target tensor: %v", err)
	}
	_ = y.Set(0.0, 0, 0) // XOR(0, 0) = 0
	_ = y.Set(1.0, 1, 0) // XOR(0, 1) = 1
	_ = y.Set(1.0, 2, 0) // XOR(1, 0) = 1
	_ = y.Set(0.0, 3, 0) // XOR(1, 1) = 0

	// 2. Initialize Model Parameters (2-2-1 Topology)
	// Hidden Layer 1: W1 [2, 2], b1 [1, 2]
	w1, _ := initRandomTensor(2, 2)
	b1, _ := tensor.New(1, 2)

	// Output Layer 2: W2 [2, 1], b2 [1, 1]
	w2, _ := initRandomTensor(2, 1)
	b2, _ := tensor.New(1, 1)

	epochs := 25000
	learningRate := 0.5

	fmt.Printf("\n[INFO] Starting training for %d epochs with Learning Rate = %.2f...\n\n", epochs, learningRate)

	// 3. Training Loop
	for epoch := 1; epoch <= epochs; epoch++ {
		// --- FORWARD PASS ---
		// Z1 = X * W1 + b1  -> Shape [4, 2]
		xw1, _ := tensor.MatMul(x, w1)
		z1, _ := addBias(xw1, b1)
		a1, _ := nn.Sigmoid(z1) // Hidden activations [4, 2]

		// Z2 = A1 * W2 + b2 -> Shape [4, 1]
		a1w2, _ := tensor.MatMul(a1, w2)
		z2, _ := addBias(a1w2, b2)
		yHat, _ := nn.Sigmoid(z2) // Predictions [4, 1]

		// Compute Loss
		lossVal, err := nn.MSE(yHat, y)
		if err != nil {
			log.Fatalf("error computing loss: %v", err)
		}

		if epoch == 1 || epoch%5000 == 0 || epoch == epochs {
			fmt.Printf("Epoch %5d/%d | Loss (MSE): %.6f\n", epoch, epochs, lossVal)
		}

		// --- BACKWARD PASS (Chain Rule) ---
		// 1. Loss Gradient: dL/dyHat [4, 1]
		dLdyHat, _ := nn.MSEGrad(yHat, y)

		// 2. Output Error: dZ2 = dL/dyHat * SigmoidPrime(yHat) [4, 1]
		sigPrimeYHat, _ := nn.SigmoidPrime(yHat)
		dZ2, _ := tensor.Mul(dLdyHat, sigPrimeYHat)

		// 3. Output Gradients: dW2 = A1^T * dZ2 [2, 1], db2 = sumRows(dZ2) [1, 1]
		a1T, _ := tensor.Transpose(a1, 0, 1)
		dW2, _ := tensor.MatMul(a1T, dZ2)
		db2, _ := sumRows(dZ2)

		// 4. Hidden Layer Error: dA1 = dZ2 * W2^T [4, 2]
		w2T, _ := tensor.Transpose(w2, 0, 1)
		dA1, _ := tensor.MatMul(dZ2, w2T)

		// 5. Hidden Delta: dZ1 = dA1 * SigmoidPrime(A1) [4, 2]
		sigPrimeA1, _ := nn.SigmoidPrime(a1)
		dZ1, _ := tensor.Mul(dA1, sigPrimeA1)

		// 6. Hidden Gradients: dW1 = X^T * dZ1 [2, 2], db1 = sumRows(dZ1) [1, 2]
		xT, _ := tensor.Transpose(x, 0, 1)
		dW1, _ := tensor.MatMul(xT, dZ1)
		db1, _ := sumRows(dZ1)

		// --- PARAMETER UPDATES (SGD) ---
		w2, _ = updateWeights(w2, dW2, learningRate)
		b2, _ = updateWeights(b2, db2, learningRate)
		w1, _ = updateWeights(w1, dW1, learningRate)
		b1, _ = updateWeights(b1, db1, learningRate)
	}

	// 4. Evaluate Final Predictions
	fmt.Println("\n==================================================================")
	fmt.Println("  Final Evaluation Results (Truth Table Prediction)               ")
	fmt.Println("==================================================================")

	xw1, _ := tensor.MatMul(x, w1)
	z1, _ := addBias(xw1, b1)
	a1, _ := nn.Sigmoid(z1)
	a1w2, _ := tensor.MatMul(a1, w2)
	z2, _ := addBias(a1w2, b2)
	yHat, _ := nn.Sigmoid(z2)

	for i := 0; i < 4; i++ {
		x0, _ := x.At(i, 0)
		x1, _ := x.At(i, 1)
		target, _ := y.At(i, 0)
		pred, _ := yHat.At(i, 0)

		fmt.Printf("Input: [%d, %d] | Expected: %.1f | Predicted: %.4f | Rounded: %d\n",
			int(x0), int(x1), target, pred, int(pred+0.5))
	}
	fmt.Println("==================================================================")
}
