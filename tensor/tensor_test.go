package tensor_test

import (
	"testing"

	"github.com/Joanoni/go-transformer-engine/tensor"
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
	// Matrix of shape [2, 3] -> strides [3, 1]
	ts, err := tensor.New(2, 3)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}

	// Set value at (1, 2) which maps to 1D index: 1*3 + 2*1 = 5
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

	// Direct 1D memory verification
	if ts.Data()[5] != 42.5 {
		t.Errorf("expected underlying 1D slice index 5 to be 42.5, got %f", ts.Data()[5])
	}
}

func TestTensor_OutOfBounds(t *testing.T) {
	ts, err := tensor.New(2, 3)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}

	// Incorrect dimension count (Rank 2 tensor accessed with 1 index)
	if _, err := ts.At(0); err == nil {
		t.Error("expected error when accessing 2D tensor with 1 index, got nil")
	}

	// Out of bounds on axis 0
	if _, err := ts.At(2, 0); err == nil {
		t.Error("expected error when axis 0 index is >= shape[0], got nil")
	}

	// Out of bounds on axis 1
	if err := ts.Set(1.0, 0, 3); err == nil {
		t.Error("expected error when axis 1 index is >= shape[1], got nil")
	}
}
