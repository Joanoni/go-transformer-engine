package nn

import (
	"context"
	"fmt"

	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

// MSE computes the Mean Squared Error loss scalar tensor [1, 1] between predictions yHat and targets y:
//
//	MSE(yHat, y) = (1 / N) * sum((yHat_i - y_i)^2)
func MSE(ctx context.Context, yHat, y *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if yHat.Size() != y.Size() {
		return nil, fmt.Errorf("MSE shape mismatch: yHat size %d != y size %d", yHat.Size(), y.Size())
	}

	n := float64(yHat.Size())
	var sumSq float64

	for i := 0; i < yHat.Size(); i++ {
		diff := yHat.Data()[i] - y.Data()[i]
		sumSq += diff * diff
	}
	mseVal := sumSq / n

	out, err := tensor.New(1, 1)
	if err != nil {
		return nil, err
	}
	if err := out.Set(mseVal, 0, 0); err != nil {
		return nil, err
	}

	tensor.RegisterOp(ctx, "MSE", out, []*tensor.Tensor{yHat, y}, func() error {
		if out.Grad() == nil {
			return nil
		}
		gradScalar := out.Grad().Data()[0]

		if yHat.RequiresGrad() {
			gYHat, err := tensor.New(yHat.Shape()...)
			if err != nil {
				return err
			}
			for i := 0; i < yHat.Size(); i++ {
				gYHat.Data()[i] = (2.0 / n) * (yHat.Data()[i] - y.Data()[i]) * gradScalar
			}
			if err := yHat.AccGrad(gYHat); err != nil {
				return err
			}
		}

		if y.RequiresGrad() {
			gY, err := tensor.New(y.Shape()...)
			if err != nil {
				return err
			}
			for i := 0; i < y.Size(); i++ {
				gY.Data()[i] = -(2.0 / n) * (yHat.Data()[i] - y.Data()[i]) * gradScalar
			}
			if err := y.AccGrad(gY); err != nil {
				return err
			}
		}

		return nil
	})

	return out, nil
}

// MSEGrad computes the analytical gradient tensor of Mean Squared Error with respect to predictions yHat:
//
//	dL / dyHat = (2 / N) * (yHat - y)
func MSEGrad(yHat, y *tensor.Tensor) (*tensor.Tensor, error) {
	if yHat.Size() != y.Size() {
		return nil, fmt.Errorf("MSEGrad shape mismatch: yHat size %d != y size %d", yHat.Size(), y.Size())
	}
	n := float64(yHat.Size())
	out, err := tensor.New(yHat.Shape()...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < yHat.Size(); i++ {
		out.Data()[i] = (2.0 / n) * (yHat.Data()[i] - y.Data()[i])
	}
	return out, nil
}
