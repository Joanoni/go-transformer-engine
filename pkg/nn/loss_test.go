package nn_test

import (
	"context"
	"math"
	"testing"

	"github.com/Joanoni/go-transformer-engine/pkg/autograd"
	"github.com/Joanoni/go-transformer-engine/pkg/nn"
	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

func TestMSE_ForwardAndAutograd(t *testing.T) {
	ctx, tape := autograd.WithTape(context.Background())

	yHat, _ := tensor.New(2, 1)
	yHat.SetRequiresGrad(true)
	_ = yHat.Set(2.5, 0, 0)
	_ = yHat.Set(0.0, 1, 0)

	y, _ := tensor.New(2, 1)
	_ = y.Set(2.0, 0, 0)
	_ = y.Set(1.0, 1, 0)

	// Loss = 1/2 * ((2.5 - 2.0)^2 + (0.0 - 1.0)^2) = 1/2 * (0.25 + 1.0) = 0.625
	loss, err := nn.MSE(ctx, yHat, y)
	if err != nil {
		t.Fatalf("MSE failed: %v", err)
	}

	val, _ := loss.At(0, 0)
	if math.Abs(val-0.625) > 1e-6 {
		t.Errorf("expected MSE loss 0.625, got %f", val)
	}

	if err := tape.Backward(loss); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	// dL/dyHat = 2/N * (yHat - y) = 2/2 * (yHat - y) = (yHat - y)
	// dyHat_0 = 2.5 - 2.0 = 0.5
	// dyHat_1 = 0.0 - 1.0 = -1.0
	g0, _ := yHat.Grad().At(0, 0)
	g1, _ := yHat.Grad().At(1, 0)

	if math.Abs(g0-0.5) > 1e-6 || math.Abs(g1-(-1.0)) > 1e-6 {
		t.Errorf("expected yHat.Grad [0.5, -1.0], got [%f, %f]", g0, g1)
	}
}
