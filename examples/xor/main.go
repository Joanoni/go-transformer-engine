package main

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/Joanoni/go-transformer-engine/pkg/autograd"
	"github.com/Joanoni/go-transformer-engine/pkg/nn"
	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

// initRandomWeights allocates a new tensor with the specified shape and fills it
// with random values drawn uniform-randomly from [-1.0, 1.0].
func initRandomWeights(shape ...int) (*tensor.Tensor, error) {
	t, err := tensor.New(shape...)
	if err != nil {
		return nil, err
	}
	data := t.Data()
	for i := range data {
		data[i] = (rand.Float64() * 2.0) - 1.0
	}
	return t, nil
}

// updateSGD applies standard Stochastic Gradient Descent weight updates:
//
//	P = P - lr * dL/dP
//
// After updating parameters, the accumulated gradient is reset to nil.
func updateSGD(p *tensor.Tensor, lr float64) {
	if p == nil || p.Grad() == nil {
		return
	}
	pData := p.Data()
	gData := p.Grad().Data()
	for i := range pData {
		pData[i] -= lr * gData[i]
	}
	p.SetGrad(nil)
}

func main() {
	fmt.Println("==========================================================================")
	fmt.Println(" go-transformer-engine: XOR Problem Demo (Phase 3 - Autograd Engine)")
	fmt.Println("==========================================================================")

	// XOR Inputs: 4 samples of 2 features
	// [0, 0] -> 0
	// [0, 1] -> 1
	// [1, 0] -> 1
	// [1, 1] -> 0
	x, err := tensor.New(4, 2)
	if err != nil {
		panic(err)
	}
	xData := []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}
	copy(x.Data(), xData)

	y, err := tensor.New(4, 1)
	if err != nil {
		panic(err)
	}
	yData := []float64{
		0,
		1,
		1,
		0,
	}
	copy(y.Data(), yData)

	// Initialize MLP 2-2-1 Parameters
	w1, err := initRandomWeights(2, 2)
	if err != nil {
		panic(err)
	}
	w1.SetRequiresGrad(true)

	b1, err := initRandomWeights(4, 2)
	if err != nil {
		panic(err)
	}
	b1.SetRequiresGrad(true)

	w2, err := initRandomWeights(2, 1)
	if err != nil {
		panic(err)
	}
	w2.SetRequiresGrad(true)

	b2, err := initRandomWeights(4, 1)
	if err != nil {
		panic(err)
	}
	b2.SetRequiresGrad(true)

	learningRate := 1.0
	epochs := 20000

	var initialLoss float64
	var finalLoss float64

	fmt.Printf("Training MLP (2-2-1) for %d epochs with Learning Rate = %.2f...\n\n", epochs, learningRate)

	for epoch := 1; epoch <= epochs; epoch++ {
		// 1. Create Tape Context for Forward Pass
		ctx, tape := autograd.WithTape(context.Background())

		// 2. Forward Pass:
		// H1 = Sigmoid(X * W1 + B1)
		// Y_hat = Sigmoid(H1 * W2 + B2)
		h1Mat, err := tensor.MatMul(ctx, x, w1)
		if err != nil {
			panic(err)
		}
		h1Add, err := tensor.Add(ctx, h1Mat, b1)
		if err != nil {
			panic(err)
		}
		a1, err := nn.Sigmoid(ctx, h1Add)
		if err != nil {
			panic(err)
		}

		h2Mat, err := tensor.MatMul(ctx, a1, w2)
		if err != nil {
			panic(err)
		}
		h2Add, err := tensor.Add(ctx, h2Mat, b2)
		if err != nil {
			panic(err)
		}
		yHat, err := nn.Sigmoid(ctx, h2Add)
		if err != nil {
			panic(err)
		}

		// 3. Compute Loss
		loss, err := nn.MSE(ctx, yHat, y)
		if err != nil {
			panic(err)
		}

		lossVal := loss.Data()[0]
		if epoch == 1 {
			initialLoss = lossVal
		}
		finalLoss = lossVal

		// 4. Automatic Backward Pass via Autograd Tape
		if err := tape.Backward(loss); err != nil {
			panic(err)
		}

		// 5. Parameter Update (SGD Step)
		updateSGD(w1, learningRate)
		updateSGD(b1, learningRate)
		updateSGD(w2, learningRate)
		updateSGD(b2, learningRate)

		// 6. Clear Tape for next iteration
		tape.Clear()

		if epoch%2000 == 0 || epoch == 1 {
			fmt.Printf("Epoch %5d/%d | MSE Loss: %.6f\n", epoch, epochs, lossVal)
		}
	}

	fmt.Println("\n--------------------------------------------------------------------------")
	fmt.Printf("Initial MSE Loss: %.6f\n", initialLoss)
	fmt.Printf("Final MSE Loss:   %.6f\n", finalLoss)
	fmt.Println("--------------------------------------------------------------------------")

	// Verify Predictions
	ctx := context.Background()
	h1Mat, _ := tensor.MatMul(ctx, x, w1)
	h1Add, _ := tensor.Add(ctx, h1Mat, b1)
	a1, _ := nn.Sigmoid(ctx, h1Add)
	h2Mat, _ := tensor.MatMul(ctx, a1, w2)
	h2Add, _ := tensor.Add(ctx, h2Mat, b2)
	predictions, _ := nn.Sigmoid(ctx, h2Add)

	fmt.Println("\nFinal XOR Predictions vs Ground Truth Targets:")
	for i := 0; i < 4; i++ {
		x0, _ := x.At(i, 0)
		x1, _ := x.At(i, 1)
		pred, _ := predictions.At(i, 0)
		target, _ := y.At(i, 0)
		fmt.Printf("Input: [%.0f, %.0f] | Target: %.0f | Prediction: %.4f\n", x0, x1, target, pred)
	}
	fmt.Println("==========================================================================")
}
