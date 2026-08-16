package autograd_test

import (
	"context"
	"errors"
	"testing"

	"github.com/Joanoni/go-transformer-engine/pkg/autograd"
	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

func TestTape_Recording(t *testing.T) {
	ctx, tape := autograd.WithTape(context.Background())

	a, _ := tensor.New(2, 2)
	a.SetRequiresGrad(true)
	b, _ := tensor.New(2, 2)

	// Forward: C = Add(A, B)
	c, err := tensor.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("failed Add: %v", err)
	}

	if tape.Len() != 1 {
		t.Fatalf("expected tape len 1, got %d", tape.Len())
	}

	nodes := tape.Nodes()
	if nodes[0].Op != "Add" {
		t.Errorf("expected op 'Add', got '%s'", nodes[0].Op)
	}
	if nodes[0].Out != c {
		t.Error("expected node output tensor to match result tensor C")
	}
}

func TestTape_BackwardLinearPass(t *testing.T) {
	// Formula: Z = MatMul(X, W) + B
	// X: [1, 2] = [[2.0, 3.0]]
	// W: [2, 2] = [[1.0, 0.0], [0.0, 1.0]] (RequiresGrad)
	// B: [1, 2] = [[0.5, 0.5]]            (RequiresGrad)
	ctx, tape := autograd.WithTape(context.Background())

	x, _ := tensor.New(1, 2)
	_ = x.Set(2.0, 0, 0)
	_ = x.Set(3.0, 0, 1)

	w, _ := tensor.New(2, 2)
	w.SetRequiresGrad(true)
	_ = w.Set(1.0, 0, 0)
	_ = w.Set(0.0, 0, 1)
	_ = w.Set(0.0, 1, 0)
	_ = w.Set(1.0, 1, 1)

	b, _ := tensor.New(1, 2)
	b.SetRequiresGrad(true)
	_ = b.Set(0.5, 0, 0)
	_ = b.Set(0.5, 0, 1)

	// Forward 1: XW = MatMul(X, W) -> [[2.0, 3.0]]
	xw, err := tensor.MatMul(ctx, x, w)
	if err != nil {
		t.Fatalf("failed MatMul: %v", err)
	}

	// Forward 2: Z = Add(XW, B) -> [[2.5, 3.5]]
	z, err := tensor.Add(ctx, xw, b)
	if err != nil {
		t.Fatalf("failed Add: %v", err)
	}

	if tape.Len() != 2 {
		t.Fatalf("expected 2 recorded ops, got %d", tape.Len())
	}

	// Backward pass
	if err := tape.Backward(z); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	// Analytical gradient verification:
	// dL/dZ = [[1.0, 1.0]]
	// dL/dB = [[1.0, 1.0]]
	// dL/dW = X^T * dL/dZ = [[2.0], [3.0]] * [[1.0, 1.0]] = [[2.0, 2.0], [3.0, 3.0]]
	bGradVal00, _ := b.Grad().At(0, 0)
	bGradVal01, _ := b.Grad().At(0, 1)
	if bGradVal00 != 1.0 || bGradVal01 != 1.0 {
		t.Errorf("expected b.Grad to be [[1.0, 1.0]], got [[%f, %f]]", bGradVal00, bGradVal01)
	}

	wGrad00, _ := w.Grad().At(0, 0)
	wGrad01, _ := w.Grad().At(0, 1)
	wGrad10, _ := w.Grad().At(1, 0)
	wGrad11, _ := w.Grad().At(1, 1)

	if wGrad00 != 2.0 || wGrad01 != 2.0 || wGrad10 != 3.0 || wGrad11 != 3.0 {
		t.Errorf("expected w.Grad to be [[2.0, 2.0], [3.0, 3.0]], got [[%f, %f], [%f, %f]]",
			wGrad00, wGrad01, wGrad10, wGrad11)
	}
}

func TestTape_Clear(t *testing.T) {
	ctx, tape := autograd.WithTape(context.Background())

	a, _ := tensor.New(2, 2)
	a.SetRequiresGrad(true)
	b, _ := tensor.New(2, 2)

	_, _ = tensor.Add(ctx, a, b)
	if tape.Len() != 1 {
		t.Fatalf("expected tape length 1 before clear, got %d", tape.Len())
	}

	tape.Clear()
	if tape.Len() != 0 {
		t.Errorf("expected tape length 0 after Clear(), got %d", tape.Len())
	}
}

func TestTape_ErrorHandling(t *testing.T) {
	t.Run("Empty Tape Error", func(t *testing.T) {
		tape := autograd.NewTape()
		ts, _ := tensor.New(2, 2)
		err := tape.Backward(ts)
		if !errors.Is(err, autograd.ErrEmptyTape) {
			t.Errorf("expected ErrEmptyTape, got %v", err)
		}
	})

	t.Run("Nil Loss Error", func(t *testing.T) {
		_, tape := autograd.WithTape(context.Background())
		a, _ := tensor.New(2, 2)
		a.SetRequiresGrad(true)
		b, _ := tensor.New(2, 2)
		_, _ = tensor.Add(context.Background(), a, b)

		// Record a dummy op manually
		tape.Record("Dummy", a, []*tensor.Tensor{a, b}, nil)

		err := tape.Backward(nil)
		if !errors.Is(err, autograd.ErrNilLossTensor) {
			t.Errorf("expected ErrNilLossTensor, got %v", err)
		}
	})
}
