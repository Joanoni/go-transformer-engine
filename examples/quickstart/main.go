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
