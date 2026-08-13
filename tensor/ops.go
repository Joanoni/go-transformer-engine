package tensor

import (
	"errors"
	"fmt"
)

var (
	ErrShapeMismatch = errors.New("tensor shapes are incompatible for this operation")
	ErrRankMismatch  = errors.New("tensor ranks are incompatible for this operation")
)

// Add performs element-wise addition between two tensors A and B of identical shape.
// Mathematical formulation:
//
//	C[i1, ..., in] = A[i1, ..., in] + B[i1, ..., in]
//
// Supports general strided memory layouts (e.g., transposed tensor views).
func Add(a, b *Tensor) (*Tensor, error) {
	if a.Rank() != b.Rank() {
		return nil, fmt.Errorf("%w: rank mismatch (%d vs %d)", ErrRankMismatch, a.Rank(), b.Rank())
	}

	aShape := a.Shape()
	bShape := b.Shape()
	for i := range aShape {
		if aShape[i] != bShape[i] {
			return nil, fmt.Errorf("%w: shape mismatch at axis %d (%d vs %d)", ErrShapeMismatch, i, aShape[i], bShape[i])
		}
	}

	out, err := New(aShape...)
	if err != nil {
		return nil, err
	}

	// N-dimensional recursive traversal supporting arbitrary strided views
	indices := make([]int, a.Rank())
	var iterate func(dim int) error
	iterate = func(dim int) error {
		if dim == a.Rank() {
			valA, errA := a.At(indices...)
			valB, errB := b.At(indices...)
			if errA != nil || errB != nil {
				return fmt.Errorf("access error during element-wise add: %v / %v", errA, errB)
			}
			return out.Set(valA+valB, indices...)
		}

		for i := 0; i < aShape[dim]; i++ {
			indices[dim] = i
			if err := iterate(dim + 1); err != nil {
				return err
			}
		}
		return nil
	}

	if err := iterate(0); err != nil {
		return nil, err
	}

	return out, nil
}

// MatMul performs 2D matrix multiplication between tensor A [M, K] and tensor B [K, N],
// producing a result tensor C [M, N] where:
//
//	C[i, j] = sum_{k=0}^{K-1} A[i, k] * B[k, j]
//
// Input Shapes:  A: [M, K], B: [K, N]
// Output Shape:  C: [M, N]
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, fmt.Errorf("%w: MatMul requires 2D matrices, got ranks %d and %d", ErrRankMismatch, a.Rank(), b.Rank())
	}

	aShape := a.Shape()
	bShape := b.Shape()

	m, kA := aShape[0], aShape[1]
	kB, n := bShape[0], bShape[1]

	if kA != kB {
		return nil, fmt.Errorf("%w: inner matrix dimensions must match, got %d and %d", ErrShapeMismatch, kA, kB)
	}

	out, err := New(m, n)
	if err != nil {
		return nil, err
	}

	// Standard matrix multiplication loop
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < kA; k++ {
				valA, errA := a.At(i, k)
				valB, errB := b.At(k, j)
				if errA != nil || errB != nil {
					return nil, fmt.Errorf("error accessing elements during MatMul: %v / %v", errA, errB)
				}
				sum += valA * valB
			}
			if err := out.Set(sum, i, j); err != nil {
				return nil, err
			}
		}
	}

	return out, nil
}

// Transpose creates a Zero-Copy view of the tensor with dimensions dim0 and dim1 swapped.
// It modifies shape and strides metadata without allocating new memory or copying the underlying data slice.
func Transpose(a *Tensor, dim0, dim1 int) (*Tensor, error) {
	rank := a.Rank()
	if dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank {
		return nil, fmt.Errorf("%w: transpose dimensions (%d, %d) out of range for rank %d", ErrIndexOutOfBounds, dim0, dim1, rank)
	}

	newShape := a.Shape()
	newStrides := a.Strides()

	// Swap shape dimensions and corresponding strides
	newShape[dim0], newShape[dim1] = newShape[dim1], newShape[dim0]
	newStrides[dim0], newStrides[dim1] = newStrides[dim1], newStrides[dim0]

	return &Tensor{
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
		data:    a.data, // Shared 1D slice (Zero-Copy)
	}, nil
}

// Reshape creates a new view or tensor with the specified new shape.
// The total number of elements in the new shape must match the current tensor size.
func Reshape(a *Tensor, newShape ...int) (*Tensor, error) {
	if len(newShape) == 0 {
		return nil, fmt.Errorf("%w: empty shape provided for reshape", ErrInvalidShape)
	}

	newSize := 1
	shapeCopy := make([]int, len(newShape))
	for i, dim := range newShape {
		if dim <= 0 {
			return nil, fmt.Errorf("%w: dimension %d must be > 0, got %d", ErrInvalidShape, i, dim)
		}
		shapeCopy[i] = dim
		newSize *= dim
	}

	if newSize != a.Size() {
		return nil, fmt.Errorf("%w: cannot reshape tensor of size %d to size %d", ErrShapeMismatch, a.Size(), newSize)
	}

	strides := computeStrides(shapeCopy)

	return &Tensor{
		shape:   shapeCopy,
		strides: strides,
		offset:  a.offset,
		data:    a.data, // Shared 1D slice (Zero-Copy)
	}, nil
}

// Apply applies a custom scalar mapping function `fn` element-wise over the tensor.
// It produces a new Tensor with the exact same shape, supporting arbitrary strided views.
func Apply(a *Tensor, fn func(x float64) float64) (*Tensor, error) {
	aShape := a.Shape()
	out, err := New(aShape...)
	if err != nil {
		return nil, err
	}

	indices := make([]int, a.Rank())
	var iterate func(dim int) error
	iterate = func(dim int) error {
		if dim == a.Rank() {
			val, err := a.At(indices...)
			if err != nil {
				return fmt.Errorf("access error during Apply: %w", err)
			}
			return out.Set(fn(val), indices...)
		}

		for i := 0; i < aShape[dim]; i++ {
			indices[dim] = i
			if err := iterate(dim + 1); err != nil {
				return err
			}
		}
		return nil
	}

	if err := iterate(0); err != nil {
		return nil, err
	}

	return out, nil
}
