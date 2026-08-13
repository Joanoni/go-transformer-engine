package tensor

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidShape      = errors.New("shape must contain at least one dimension with positive size")
	ErrDimensionMismatch = errors.New("number of indices does not match tensor rank")
	ErrIndexOutOfBounds  = errors.New("index is out of bounds for tensor dimension")
)

// Tensor represents an N-dimensional array stored in contiguous memory (Row-Major / C-style layout).
// It encapsulates shape, stride, offset, and raw 1D slice data.
type Tensor struct {
	shape   []int
	strides []int
	offset  int
	data    []float64
}

// New creates a new Tensor with the given shape dimensions.
// Memory is initialized with zeros in a single contiguous 1D slice.
// Strides are automatically computed using standard Row-Major (C-contiguous) ordering:
//
//	strides[len-1] = 1
//	strides[i]     = strides[i+1] * shape[i+1]
func New(shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("%w: empty shape provided", ErrInvalidShape)
	}

	size := 1
	shapeCopy := make([]int, len(shape))
	for i, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("%w: dimension %d must be > 0, got %d", ErrInvalidShape, i, dim)
		}
		shapeCopy[i] = dim
		size *= dim
	}

	strides := computeStrides(shapeCopy)
	data := make([]float64, size)

	return &Tensor{
		shape:   shapeCopy,
		strides: strides,
		offset:  0,
		data:    data,
	}, nil
}

// computeStrides calculates Row-Major strides for a given shape.
// Example: shape [2, 3, 4] -> strides [12, 4, 1]
func computeStrides(shape []int) []int {
	rank := len(shape)
	strides := make([]int, rank)
	if rank == 0 {
		return strides
	}

	strides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// computeIndex maps multidimensional coordinates (indices) to the corresponding 1D index
// in the underlying continuous data slice using:
//
//	index1D = offset + sum(indices[i] * strides[i])
func (t *Tensor) computeIndex(indices []int) (int, error) {
	if len(indices) != len(t.shape) {
		return 0, fmt.Errorf("%w: expected %d indices, got %d", ErrDimensionMismatch, len(t.shape), len(indices))
	}

	idx1D := t.offset
	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			return 0, fmt.Errorf("%w: index %d out of bounds for axis %d with size %d", ErrIndexOutOfBounds, idx, i, t.shape[i])
		}
		idx1D += idx * t.strides[i]
	}

	return idx1D, nil
}

// At returns the float64 scalar value at the specified multidimensional coordinates.
func (t *Tensor) At(indices ...int) (float64, error) {
	idx1D, err := t.computeIndex(indices)
	if err != nil {
		return 0, err
	}
	return t.data[idx1D], nil
}

// Set stores a float64 scalar value at the specified multidimensional coordinates.
func (t *Tensor) Set(val float64, indices ...int) error {
	idx1D, err := t.computeIndex(indices)
	if err != nil {
		return err
	}
	t.data[idx1D] = val
	return nil
}

// Shape returns a copy of the tensor's shape dimensions.
func (t *Tensor) Shape() []int {
	cp := make([]int, len(t.shape))
	copy(cp, t.shape)
	return cp
}

// Strides returns a copy of the tensor's strides.
func (t *Tensor) Strides() []int {
	cp := make([]int, len(t.strides))
	copy(cp, t.strides)
	return cp
}

// Offset returns the current memory starting offset.
func (t *Tensor) Offset() int {
	return t.offset
}

// Data returns the raw 1D underlying memory slice.
func (t *Tensor) Data() []float64 {
	return t.data
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	return len(t.data)
}

// Rank returns the number of dimensions (axes) of the tensor.
func (t *Tensor) Rank() int {
	return len(t.shape)
}
