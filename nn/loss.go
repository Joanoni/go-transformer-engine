package nn

import (
	"fmt"

	"github.com/Joanoni/go-transformer-engine/tensor"
)

// MSE computes the Mean Squared Error loss scalar between prediction tensor yHat and target tensor y:
//
//	MSE(yHat, y) = (1 / N) * sum((yHat_i - y_i)^2)
func MSE(yHat, y *tensor.Tensor) (float64, error) {
	if yHat.Size() != y.Size() {
		return 0, fmt.Errorf("size mismatch between predictions (%d) and targets (%d)", yHat.Size(), y.Size())
	}

	diff, err := tensor.Sub(yHat, y)
	if err != nil {
		return 0, fmt.Errorf("failed to subtract targets from predictions: %w", err)
	}

	squared, err := tensor.Apply(diff, func(x float64) float64 {
		return x * x
	})
	if err != nil {
		return 0, fmt.Errorf("failed to square diff tensor: %w", err)
	}

	var totalSum float64
	for _, val := range squared.Data() {
		totalSum += val
	}

	return totalSum / float64(yHat.Size()), nil
}

// MSEGrad computes the gradient tensor of the Mean Squared Error loss with respect to predictions yHat:
//
//	dL / dyHat = (2 / N) * (yHat - y)
func MSEGrad(yHat, y *tensor.Tensor) (*tensor.Tensor, error) {
	if yHat.Size() != y.Size() {
		return nil, fmt.Errorf("size mismatch between predictions (%d) and targets (%d)", yHat.Size(), y.Size())
	}

	diff, err := tensor.Sub(yHat, y)
	if err != nil {
		return nil, fmt.Errorf("failed to subtract targets from predictions: %w", err)
	}

	scale := 2.0 / float64(yHat.Size())
	return tensor.Apply(diff, func(x float64) float64 {
		return x * scale
	})
}
