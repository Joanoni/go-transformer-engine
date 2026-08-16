package tensor

import (
	"context"
	"errors"
	"fmt"
)

var (
	ErrShapeMismatch = errors.New("tensor shapes are incompatible for this operation")
	ErrRankMismatch  = errors.New("tensor ranks are incompatible for this operation")
)

// Recorder defines the contract for recording forward operations onto an autograd Tape
// without creating cyclic package dependencies between tensor and autograd packages.
type Recorder interface {
	Record(op string, out *Tensor, inputs []*Tensor, backwardFn func() error)
}

type tapeKeyType struct{}

var tapeKey = tapeKeyType{}

// ContextWithRecorder attaches a Recorder implementation to the given context.
func ContextWithRecorder(parent context.Context, r Recorder) context.Context {
	if parent == nil {
		parent = context.Background()
	}
	return context.WithValue(parent, tapeKey, r)
}

func getRecorder(ctx context.Context) Recorder {
	if ctx == nil {
		return nil
	}
	r, _ := ctx.Value(tapeKey).(Recorder)
	return r
}

// RegisterOp registers an operation and its analytical backward closure on the active Recorder in ctx.
func RegisterOp(ctx context.Context, op string, out *Tensor, inputs []*Tensor, backwardFn func() error) {
	if ctx == nil {
		return
	}
	rec := getRecorder(ctx)
	if rec == nil {
		return
	}

	hasGrad := false
	for _, in := range inputs {
		if in != nil && in.RequiresGrad() {
			hasGrad = true
			break
		}
	}

	if hasGrad {
		out.SetRequiresGrad(true)
		rec.Record(op, out, inputs, backwardFn)
	}
}

// Add performs element-wise addition between two tensors A and B of identical shape.
// Mathematical formulation:
//
//	C[i1, ..., in] = A[i1, ..., in] + B[i1, ..., in]
func Add(ctx context.Context, a, b *Tensor) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
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

	RegisterOp(ctx, "Add", out, []*Tensor{a, b}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if a.RequiresGrad() {
			if err := a.AccGrad(out.Grad()); err != nil {
				return err
			}
		}
		if b.RequiresGrad() {
			if err := b.AccGrad(out.Grad()); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// Sub performs element-wise subtraction between two tensors A and B of identical shape.
// Mathematical formulation:
//
//	C[i1, ..., in] = A[i1, ..., in] - B[i1, ..., in]
func Sub(ctx context.Context, a, b *Tensor) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
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

	indices := make([]int, a.Rank())
	var iterate func(dim int) error
	iterate = func(dim int) error {
		if dim == a.Rank() {
			valA, errA := a.At(indices...)
			valB, errB := b.At(indices...)
			if errA != nil || errB != nil {
				return fmt.Errorf("access error during element-wise sub: %v / %v", errA, errB)
			}
			return out.Set(valA-valB, indices...)
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

	RegisterOp(ctx, "Sub", out, []*Tensor{a, b}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if a.RequiresGrad() {
			if err := a.AccGrad(out.Grad()); err != nil {
				return err
			}
		}
		if b.RequiresGrad() {
			negGrad, err := Apply(context.Background(), out.Grad(), func(x float64) float64 { return -x })
			if err != nil {
				return err
			}
			if err := b.AccGrad(negGrad); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// Mul performs element-wise multiplication (Hadamard product) between two tensors A and B of identical shape.
// Mathematical formulation:
//
//	C[i1, ..., in] = A[i1, ..., in] * B[i1, ..., in]
func Mul(ctx context.Context, a, b *Tensor) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
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

	indices := make([]int, a.Rank())
	var iterate func(dim int) error
	iterate = func(dim int) error {
		if dim == a.Rank() {
			valA, errA := a.At(indices...)
			valB, errB := b.At(indices...)
			if errA != nil || errB != nil {
				return fmt.Errorf("access error during element-wise mul: %v / %v", errA, errB)
			}
			return out.Set(valA*valB, indices...)
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

	RegisterOp(ctx, "Mul", out, []*Tensor{a, b}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if a.RequiresGrad() {
			gradA, err := Mul(context.Background(), out.Grad(), b)
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		if b.RequiresGrad() {
			gradB, err := Mul(context.Background(), out.Grad(), a)
			if err != nil {
				return err
			}
			if err := b.AccGrad(gradB); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// MatMul performs 2D matrix multiplication between tensor A [M, K] and tensor B [K, N],
// producing a result tensor C [M, N] where:
//
//	C[i, j] = sum_{k=0}^{K-1} A[i, k] * B[k, j]
//
// Analytical backward pass:
//
//	dL/dA = dL/dC * B^T  ([M, N] x [N, K] -> [M, K])
//	dL/dB = A^T * dL/dC  ([K, M] x [M, N] -> [K, N])
func MatMul(ctx context.Context, a, b *Tensor) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
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

	RegisterOp(ctx, "MatMul", out, []*Tensor{a, b}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if a.RequiresGrad() {
			bT, err := Transpose(context.Background(), b, 0, 1)
			if err != nil {
				return err
			}
			gradA, err := MatMul(context.Background(), out.Grad(), bT)
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		if b.RequiresGrad() {
			aT, err := Transpose(context.Background(), a, 0, 1)
			if err != nil {
				return err
			}
			gradB, err := MatMul(context.Background(), aT, out.Grad())
			if err != nil {
				return err
			}
			if err := b.AccGrad(gradB); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// Transpose creates a Zero-Copy view of the tensor with dimensions dim0 and dim1 swapped.
func Transpose(ctx context.Context, a *Tensor, dim0, dim1 int) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	rank := a.Rank()
	if dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank {
		return nil, fmt.Errorf("%w: transpose dimensions (%d, %d) out of range for rank %d", ErrIndexOutOfBounds, dim0, dim1, rank)
	}

	newShape := a.Shape()
	newStrides := a.Strides()

	newShape[dim0], newShape[dim1] = newShape[dim1], newShape[dim0]
	newStrides[dim0], newStrides[dim1] = newStrides[dim1], newStrides[dim0]

	out := &Tensor{
		shape:   newShape,
		strides: newStrides,
		offset:  a.offset,
		data:    a.data,
	}

	RegisterOp(ctx, "Transpose", out, []*Tensor{a}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if a.RequiresGrad() {
			gradA, err := Transpose(context.Background(), out.Grad(), dim0, dim1)
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// Reshape creates a new view or tensor with the specified new shape.
func Reshape(ctx context.Context, a *Tensor, newShape ...int) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
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

	out := &Tensor{
		shape:   shapeCopy,
		strides: strides,
		offset:  a.offset,
		data:    a.data,
	}

	RegisterOp(ctx, "Reshape", out, []*Tensor{a}, func() error {
		if out.Grad() == nil {
			return nil
		}
		if a.RequiresGrad() {
			gradA, err := Reshape(context.Background(), out.Grad(), a.Shape()...)
			if err != nil {
				return err
			}
			if err := a.AccGrad(gradA); err != nil {
				return err
			}
		}
		return nil
	})

	return out, nil
}

// Apply applies a custom scalar mapping function `fn` element-wise over the tensor.
func Apply(ctx context.Context, a *Tensor, fn func(x float64) float64) (*Tensor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
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
