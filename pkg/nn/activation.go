package nn

import (
	"context"
	"math"

	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

// Sigmoid computes the element-wise Sigmoid activation function:
//
//	sigma(x) = 1 / (1 + e^-x)
func Sigmoid(ctx context.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	out, err := tensor.Apply(ctx, x, func(val float64) float64 {
		return 1.0 / (1.0 + math.Exp(-val))
	})
	if err != nil {
		return nil, err
	}

	tensor.RegisterOp(ctx, "Sigmoid", out, []*tensor.Tensor{x}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if x.RequiresGrad() {
			sigmoidGrad, err := SigmoidPrime(out)
			if err != nil {
				return err
			}
			gradX, err := tensor.Mul(context.Background(), out.Grad(), sigmoidGrad)
			if err != nil {
				return err
			}
			if err := x.AccGrad(gradX); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// SigmoidPrime computes the element-wise derivative of Sigmoid given precomputed Sigmoid output:
//
//	sigma'(x) = sigma(x) * (1 - sigma(x))
func SigmoidPrime(sigmoidOut *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Apply(context.Background(), sigmoidOut, func(s float64) float64 {
		return s * (1.0 - s)
	})
}

// ReLU computes the element-wise Rectified Linear Unit activation function:
//
//	ReLU(x) = max(0, x)
func ReLU(ctx context.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	out, err := tensor.Apply(ctx, x, func(val float64) float64 {
		return math.Max(0.0, val)
	})
	if err != nil {
		return nil, err
	}

	tensor.RegisterOp(ctx, "ReLU", out, []*tensor.Tensor{x}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if x.RequiresGrad() {
			reluGrad, err := ReLUPrime(x)
			if err != nil {
				return err
			}
			gradX, err := tensor.Mul(context.Background(), out.Grad(), reluGrad)
			if err != nil {
				return err
			}
			if err := x.AccGrad(gradX); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// ReLUPrime computes the element-wise derivative of ReLU function for input x:
//
//	ReLU'(x) = 1 if x > 0 else 0
func ReLUPrime(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Apply(context.Background(), x, func(val float64) float64 {
		if val > 0 {
			return 1.0
		}
		return 0.0
	})
}
