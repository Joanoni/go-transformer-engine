package tensor_test

import (
	"context"
	"errors"
	"testing"

	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

func TestNewTensor_ValidShapes(t *testing.T) {
	tests := []struct {
		name            string
		shape           []int
		expectedSize    int
		expectedStrides []int
	}{
		{
			name:            "1D Vector",
			shape:           []int{5},
			expectedSize:    5,
			expectedStrides: []int{1},
		},
		{
			name:            "2D Matrix (2x3)",
			shape:           []int{2, 3},
			expectedSize:    6,
			expectedStrides: []int{3, 1},
		},
		{
			name:            "3D Tensor (2x3x4)",
			shape:           []int{2, 3, 4},
			expectedSize:    24,
			expectedStrides: []int{12, 4, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts, err := tensor.New(tt.shape...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if ts.Size() != tt.expectedSize {
				t.Errorf("expected size %d, got %d", tt.expectedSize, ts.Size())
			}

			strides := ts.Strides()
			if len(strides) != len(tt.expectedStrides) {
				t.Fatalf("strides length mismatch: expected %d, got %d", len(tt.expectedStrides), len(strides))
			}

			for i := range strides {
				if strides[i] != tt.expectedStrides[i] {
					t.Errorf("stride at axis %d: expected %d, got %d", i, tt.expectedStrides[i], strides[i])
				}
			}
		})
	}
}

func TestNewTensor_InvalidShapes(t *testing.T) {
	invalidShapes := [][]int{
		{},
		{0},
		{-1},
		{2, -3},
	}

	for _, shape := range invalidShapes {
		_, err := tensor.New(shape...)
		if err == nil {
			t.Errorf("expected error for invalid shape %v, got nil", shape)
		}
	}
}

func TestTensor_AtAndSet(t *testing.T) {
	ts, err := tensor.New(2, 3)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}

	err = ts.Set(42.5, 1, 2)
	if err != nil {
		t.Fatalf("failed to set value: %v", err)
	}

	val, err := ts.At(1, 2)
	if err != nil {
		t.Fatalf("failed to get value: %v", err)
	}

	if val != 42.5 {
		t.Errorf("expected 42.5 at (1, 2), got %f", val)
	}

	if ts.Data()[5] != 42.5 {
		t.Errorf("expected underlying 1D slice index 5 to be 42.5, got %f", ts.Data()[5])
	}
}

func TestTensor_OutOfBounds(t *testing.T) {
	ts, err := tensor.New(2, 3)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}

	if _, err := ts.At(0); err == nil {
		t.Error("expected error when accessing 2D tensor with 1 index, got nil")
	}

	if _, err := ts.At(2, 0); err == nil {
		t.Error("expected error when axis 0 index is >= shape[0], got nil")
	}

	if err := ts.Set(1.0, 0, 3); err == nil {
		t.Error("expected error when axis 1 index is >= shape[1], got nil")
	}
}

func TestTensor_Offset(t *testing.T) {
	ts, _ := tensor.New(2, 2)
	if ts.Offset() != 0 {
		t.Errorf("expected initial offset 0, got %d", ts.Offset())
	}
}

func TestTensor_GradGettersAndSetters(t *testing.T) {
	ts, err := tensor.New(2, 2)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}

	if ts.RequiresGrad() {
		t.Error("expected default requiresGrad to be false")
	}

	ts.SetRequiresGrad(true)
	if !ts.RequiresGrad() {
		t.Error("expected requiresGrad to be true after SetRequiresGrad(true)")
	}

	if ts.Grad() != nil {
		t.Error("expected default Grad() to be nil")
	}

	gradTensor, _ := tensor.New(2, 2)
	_ = gradTensor.Set(1.5, 0, 0)
	ts.SetGrad(gradTensor)

	if ts.Grad() == nil || ts.Grad().Data()[0] != 1.5 {
		t.Errorf("expected Grad() to return assigned gradient tensor")
	}
}

func TestTensor_AccGrad(t *testing.T) {
	t.Run("Nil Gradient Accumulation", func(t *testing.T) {
		ts, _ := tensor.New(2, 2)
		if err := ts.AccGrad(nil); err != nil {
			t.Fatalf("unexpected error when accumulating nil gradient: %v", err)
		}
		if ts.Grad() != nil {
			t.Error("expected grad to remain nil when accumulating nil")
		}
	})

	t.Run("First Gradient Allocation and Accumulation", func(t *testing.T) {
		ts, _ := tensor.New(2, 2)
		g1, _ := tensor.New(2, 2)
		_ = g1.Set(2.0, 0, 1)

		if err := ts.AccGrad(g1); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if ts.Grad() == nil {
			t.Fatal("expected grad to be allocated")
		}

		val, _ := ts.Grad().At(0, 1)
		if val != 2.0 {
			t.Errorf("expected accumulated gradient at (0,1) to be 2.0, got %f", val)
		}

		g2, _ := tensor.New(2, 2)
		_ = g2.Set(3.5, 0, 1)

		if err := ts.AccGrad(g2); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		val, _ = ts.Grad().At(0, 1)
		if val != 5.5 {
			t.Errorf("expected accumulated gradient at (0,1) to be 5.5 (2.0 + 3.5), got %f", val)
		}
	})

	t.Run("Non-contiguous Strided Gradient Accumulation", func(t *testing.T) {
		ctx := context.Background()
		ts, _ := tensor.New(2, 3)

		base, _ := tensor.New(3, 2)
		_ = base.Set(10.0, 0, 1)
		stridedGrad, err := tensor.Transpose(ctx, base, 0, 1)
		if err != nil {
			t.Fatalf("failed to transpose: %v", err)
		}

		if err := ts.AccGrad(stridedGrad); err != nil {
			t.Fatalf("failed to accumulate strided gradient: %v", err)
		}

		val, _ := ts.Grad().At(1, 0)
		if val != 10.0 {
			t.Errorf("expected strided accumulated gradient at (1,0) to be 10.0, got %f", val)
		}
	})

	t.Run("Shape Mismatch Error", func(t *testing.T) {
		ts, _ := tensor.New(2, 2)
		invalidGrad, _ := tensor.New(3, 3)

		err := ts.AccGrad(invalidGrad)
		if err == nil {
			t.Fatal("expected error for shape mismatch, got nil")
		}
		if !errors.Is(err, tensor.ErrGradShapeMismatch) {
			t.Errorf("expected ErrGradShapeMismatch, got %v", err)
		}
	})
}
