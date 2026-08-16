package nn_test

import (
	"context"
	"math"
	"testing"

	"github.com/Joanoni/go-transformer-engine/pkg/autograd"
	"github.com/Joanoni/go-transformer-engine/pkg/nn"
	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

func TestSigmoid_ForwardAndAutograd(t *testing.T) {
	ctx, tape := autograd.WithTape(context.Background())

	x, _ := tensor.New(1, 2)
	x.SetRequiresGrad(true)
	_ = x.Set(0.0, 0, 0)
	_ = x.Set(2.0, 0, 1)

	out, err := nn.Sigmoid(ctx, x)
	if err != nil {
		t.Fatalf("Sigmoid failed: %v", err)
	}

	val0, _ := out.At(0, 0)
	val1, _ := out.At(0, 1)

	expected0 := 0.5
	expected1 := 1.0 / (1.0 + math.Exp(-2.0))

	if math.Abs(val0-expected0) > 1e-6 || math.Abs(val1-expected1) > 1e-6 {
		t.Errorf("Sigmoid expected [%f, %f], got [%f, %f]", expected0, expected1, val0, val1)
	}

	if err := tape.Backward(out); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	// dSigmoid/dx = sig * (1 - sig)
	grad0, _ := x.Grad().At(0, 0)
	grad1, _ := x.Grad().At(0, 1)

	expGrad0 := expected0 * (1.0 - expected0) // 0.25
	expGrad1 := expected1 * (1.0 - expected1)

	if math.Abs(grad0-expGrad0) > 1e-6 || math.Abs(grad1-expGrad1) > 1e-6 {
		t.Errorf("Sigmoid grad expected [%f, %f], got [%f, %f]", expGrad0, expGrad1, grad0, grad1)
	}
}

func TestReLU_ForwardAndAutograd(t *testing.T) {
	ctx, tape := autograd.WithTape(context.Background())

	x, _ := tensor.New(1, 3)
	x.SetRequiresGrad(true)
	_ = x.Set(-2.0, 0, 0)
	_ = x.Set(0.0, 0, 1)
	_ = x.Set(3.0, 0, 2)

	out, err := nn.ReLU(ctx, x)
	if err != nil {
		t.Fatalf("ReLU failed: %v", err)
	}

	val0, _ := out.At(0, 0)
	val1, _ := out.At(0, 1)
	val2, _ := out.At(0, 2)

	if val0 != 0.0 || val1 != 0.0 || val2 != 3.0 {
		t.Errorf("ReLU expected [0, 0, 3], got [%f, %f, %f]", val0, val1, val2)
	}

	if err := tape.Backward(out); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	grad0, _ := x.Grad().At(0, 0)
	grad1, _ := x.Grad().At(0, 1)
	grad2, _ := x.Grad().At(0, 2)

	if grad0 != 0.0 || grad1 != 0.0 || grad2 != 1.0 {
		t.Errorf("ReLU grad expected [0, 0, 1], got [%f, %f, %f]", grad0, grad1, grad2)
	}
}
