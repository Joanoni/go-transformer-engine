package nn

import (
	"math"

	"github.com/Joanoni/go-transformer-engine/tensor"
)

// Sigmoid computes the element-wise Sigmoid activation function:
//
//	sigma(x) = 1 / (1 + e^-x)
//
// Maps input values to the open interval (0, 1).
func Sigmoid(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Apply(x, func(v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	})
}

// SigmoidPrime computes the element-wise derivative of the Sigmoid function:
//
//	sigma'(x) = sigma(x) * (1 - sigma(x))
//
// Expects the input tensor `sigmoidOut` to be the pre-computed Sigmoid output sigma(x).
func SigmoidPrime(sigmoidOut *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Apply(sigmoidOut, func(s float64) float64 {
		return s * (1.0 - s)
	})
}

// ReLU computes the element-wise Rectified Linear Unit activation function:
//
//	ReLU(x) = max(0, x)
func ReLU(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Apply(x, func(v float64) float64 {
		if v > 0 {
			return v
		}
		return 0.0
	})
}

// ReLUPrime computes the element-wise derivative of the ReLU function:
//
//	ReLU'(x) = 1 if x > 0 else 0
func ReLUPrime(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Apply(x, func(v float64) float64 {
		if v > 0 {
			return 1.0
		}
		return 0.0
	})
}
