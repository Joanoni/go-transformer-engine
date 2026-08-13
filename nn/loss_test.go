package nn_test

import (
	"math"
	"testing"

	"github.com/Joanoni/go-transformer-engine/nn"
	"github.com/Joanoni/go-transformer-engine/tensor"
)

func TestMSE(t *testing.T) {
	yHat, _ := tensor.New(2, 1)
	y, _ := tensor.New(2, 1)

	_ = yHat.Set(0.8, 0, 0)
	_ = yHat.Set(0.2, 1, 0)

	_ = y.Set(1.0, 0, 0)
	_ = y.Set(0.0, 1, 0)

	// diffs: [-0.2, 0.2]
	// squared diffs: [0.04, 0.04]
	// sum = 0.08
	// MSE = 0.08 / 2 = 0.04
	loss, err := nn.MSE(yHat, y)
	if err != nil {
		t.Fatalf("unexpected error during MSE: %v", err)
	}

	if math.Abs(loss-0.04) > 1e-6 {
		t.Errorf("expected MSE loss 0.04, got %f", loss)
	}
}

func TestMSEGrad(t *testing.T) {
	yHat, _ := tensor.New(2, 1)
	y, _ := tensor.New(2, 1)

	_ = yHat.Set(0.8, 0, 0)
	_ = yHat.Set(0.2, 1, 0)

	_ = y.Set(1.0, 0, 0)
	_ = y.Set(0.0, 1, 0)

	// scale = 2 / 2 = 1.0
	// grad = diff = [-0.2, 0.2]
	grad, err := nn.MSEGrad(yHat, y)
	if err != nil {
		t.Fatalf("unexpected error during MSEGrad: %v", err)
	}

	val0, _ := grad.At(0, 0)
	val1, _ := grad.At(1, 0)

	if math.Abs(val0-(-0.2)) > 1e-6 {
		t.Errorf("expected grad[0] == -0.2, got %f", val0)
	}
	if math.Abs(val1-0.2) > 1e-6 {
		t.Errorf("expected grad[1] == 0.2, got %f", val1)
	}
}
