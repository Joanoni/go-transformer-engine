package nn_test

import (
	"math"
	"testing"

	"github.com/Joanoni/go-transformer-engine/nn"
	"github.com/Joanoni/go-transformer-engine/tensor"
)

const epsilon = 1e-6

func floatEquals(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestSigmoid(t *testing.T) {
	ts, _ := tensor.New(1, 3)
	_ = ts.Set(-2.0, 0, 0)
	_ = ts.Set(0.0, 0, 1)
	_ = ts.Set(2.0, 0, 2)

	out, err := nn.Sigmoid(ts)
	if err != nil {
		t.Fatalf("unexpected error during Sigmoid: %v", err)
	}

	// Sigmoid(0) = 0.5
	valZero, _ := out.At(0, 1)
	if !floatEquals(valZero, 0.5) {
		t.Errorf("expected Sigmoid(0) == 0.5, got %f", valZero)
	}

	// Sigmoid(-2) approx 0.1192029
	valNeg, _ := out.At(0, 0)
	expectedNeg := 1.0 / (1.0 + math.Exp(2.0))
	if !floatEquals(valNeg, expectedNeg) {
		t.Errorf("expected Sigmoid(-2) == %f, got %f", expectedNeg, valNeg)
	}
}

func TestSigmoidPrime(t *testing.T) {
	// Suppose sigmoidOut has value 0.5 (corresponding to input x = 0)
	ts, _ := tensor.New(1, 1)
	_ = ts.Set(0.5, 0, 0)

	out, err := nn.SigmoidPrime(ts)
	if err != nil {
		t.Fatalf("unexpected error during SigmoidPrime: %v", err)
	}

	// SigmoidPrime(0.5) = 0.5 * (1 - 0.5) = 0.25
	val, _ := out.At(0, 0)
	if !floatEquals(val, 0.25) {
		t.Errorf("expected SigmoidPrime(0.5) == 0.25, got %f", val)
	}
}

func TestReLU(t *testing.T) {
	ts, _ := tensor.New(1, 3)
	_ = ts.Set(-5.0, 0, 0)
	_ = ts.Set(0.0, 0, 1)
	_ = ts.Set(5.0, 0, 2)

	out, err := nn.ReLU(ts)
	if err != nil {
		t.Fatalf("unexpected error during ReLU: %v", err)
	}

	valNeg, _ := out.At(0, 0)
	if valNeg != 0.0 {
		t.Errorf("expected ReLU(-5) == 0.0, got %f", valNeg)
	}

	valPos, _ := out.At(0, 2)
	if valPos != 5.0 {
		t.Errorf("expected ReLU(5) == 5.0, got %f", valPos)
	}
}

func TestReLUPrime(t *testing.T) {
	ts, _ := tensor.New(1, 3)
	_ = ts.Set(-3.0, 0, 0)
	_ = ts.Set(0.0, 0, 1)
	_ = ts.Set(3.0, 0, 2)

	out, err := nn.ReLUPrime(ts)
	if err != nil {
		t.Fatalf("unexpected error during ReLUPrime: %v", err)
	}

	valNeg, _ := out.At(0, 0)
	if valNeg != 0.0 {
		t.Errorf("expected ReLU'(-3) == 0.0, got %f", valNeg)
	}

	valPos, _ := out.At(0, 2)
	if valPos != 1.0 {
		t.Errorf("expected ReLU'(3) == 1.0, got %f", valPos)
	}
}
