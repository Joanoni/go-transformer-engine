package tensor_test

import (
	"context"
	"testing"

	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

type mockRecord struct {
	op         string
	out        *tensor.Tensor
	inputs     []*tensor.Tensor
	backwardFn func() error
}

type mockTape struct {
	records []mockRecord
}

func (m *mockTape) Record(op string, out *tensor.Tensor, inputs []*tensor.Tensor, backwardFn func() error) {
	m.records = append(m.records, mockRecord{
		op:         op,
		out:        out,
		inputs:     inputs,
		backwardFn: backwardFn,
	})
}

func TestAdd_MatchingShapes(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 2)
	b, _ := tensor.New(2, 2)

	_ = a.Set(1.0, 0, 0)
	_ = a.Set(2.0, 0, 1)
	_ = a.Set(3.0, 1, 0)
	_ = a.Set(4.0, 1, 1)

	_ = b.Set(10.0, 0, 0)
	_ = b.Set(20.0, 0, 1)
	_ = b.Set(30.0, 1, 0)
	_ = b.Set(40.0, 1, 1)

	c, err := tensor.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error during Add: %v", err)
	}

	expectedValues := [][]float64{
		{11.0, 22.0},
		{33.0, 44.0},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			val, _ := c.At(i, j)
			if val != expectedValues[i][j] {
				t.Errorf("at (%d, %d): expected %f, got %f", i, j, expectedValues[i][j], val)
			}
		}
	}
}

func TestAdd_ShapeMismatch(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 3)
	b, _ := tensor.New(2, 2)

	_, err := tensor.Add(ctx, a, b)
	if err == nil {
		t.Error("expected shape mismatch error, got nil")
	}
}

func TestSub_ElementWise(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 2)
	b, _ := tensor.New(2, 2)

	_ = a.Set(10.0, 0, 0)
	_ = a.Set(20.0, 0, 1)
	_ = a.Set(30.0, 1, 0)
	_ = a.Set(40.0, 1, 1)

	_ = b.Set(1.0, 0, 0)
	_ = b.Set(2.0, 0, 1)
	_ = b.Set(3.0, 1, 0)
	_ = b.Set(4.0, 1, 1)

	c, err := tensor.Sub(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error during Sub: %v", err)
	}

	expected := [][]float64{
		{9.0, 18.0},
		{27.0, 36.0},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			val, _ := c.At(i, j)
			if val != expected[i][j] {
				t.Errorf("at (%d, %d): expected %f, got %f", i, j, expected[i][j], val)
			}
		}
	}
}

func TestMul_ElementWise(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 2)
	b, _ := tensor.New(2, 2)

	_ = a.Set(2.0, 0, 0)
	_ = a.Set(3.0, 0, 1)
	_ = a.Set(4.0, 1, 0)
	_ = a.Set(5.0, 1, 1)

	_ = b.Set(10.0, 0, 0)
	_ = b.Set(0.5, 0, 1)
	_ = b.Set(2.0, 1, 0)
	_ = b.Set(-1.0, 1, 1)

	c, err := tensor.Mul(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error during Mul: %v", err)
	}

	expected := [][]float64{
		{20.0, 1.5},
		{8.0, -5.0},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			val, _ := c.At(i, j)
			if val != expected[i][j] {
				t.Errorf("at (%d, %d): expected %f, got %f", i, j, expected[i][j], val)
			}
		}
	}
}

func TestMatMul_2D(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 3)
	_ = a.Set(1.0, 0, 0)
	_ = a.Set(2.0, 0, 1)
	_ = a.Set(3.0, 0, 2)
	_ = a.Set(4.0, 1, 0)
	_ = a.Set(5.0, 1, 1)
	_ = a.Set(6.0, 1, 2)

	b, _ := tensor.New(3, 2)
	_ = b.Set(7.0, 0, 0)
	_ = b.Set(8.0, 0, 1)
	_ = b.Set(9.0, 1, 0)
	_ = b.Set(10.0, 1, 1)
	_ = b.Set(11.0, 2, 0)
	_ = b.Set(12.0, 2, 1)

	c, err := tensor.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error during MatMul: %v", err)
	}

	expected := [][]float64{
		{58.0, 64.0},
		{139.0, 154.0},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			val, _ := c.At(i, j)
			if val != expected[i][j] {
				t.Errorf("at C(%d, %d): expected %f, got %f", i, j, expected[i][j], val)
			}
		}
	}
}

func TestTranspose_ZeroCopy(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 3)
	_ = a.Set(99.0, 0, 1)

	at, err := tensor.Transpose(ctx, a, 0, 1)
	if err != nil {
		t.Fatalf("unexpected error during Transpose: %v", err)
	}

	if at.Shape()[0] != 3 || at.Shape()[1] != 2 {
		t.Errorf("expected transposed shape [3, 2], got %v", at.Shape())
	}

	val, err := at.At(1, 0)
	if err != nil {
		t.Fatalf("failed to read from transposed tensor: %v", err)
	}
	if val != 99.0 {
		t.Errorf("expected 99.0 at transposed (1, 0), got %f", val)
	}
}

func TestReshape_ZeroCopy(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(6)
	for i := 0; i < 6; i++ {
		_ = a.Set(float64(i+1), i)
	}

	b, err := tensor.Reshape(ctx, a, 2, 3)
	if err != nil {
		t.Fatalf("unexpected error during Reshape: %v", err)
	}

	val, _ := b.At(1, 1)
	if val != 5.0 {
		t.Errorf("expected 5.0 at (1, 1), got %f", val)
	}
}

func TestApply(t *testing.T) {
	ctx := context.Background()
	a, _ := tensor.New(2, 2)
	_ = a.Set(1.0, 0, 0)
	_ = a.Set(2.0, 0, 1)
	_ = a.Set(3.0, 1, 0)
	_ = a.Set(4.0, 1, 1)

	out, err := tensor.Apply(ctx, a, func(x float64) float64 {
		return x * 2.0
	})
	if err != nil {
		t.Fatalf("unexpected error during Apply: %v", err)
	}

	expected := [][]float64{
		{2.0, 4.0},
		{6.0, 8.0},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			val, _ := out.At(i, j)
			if val != expected[i][j] {
				t.Errorf("at (%d, %d): expected %f, got %f", i, j, expected[i][j], val)
			}
		}
	}
}

func TestAutogradRecordingAndBackwardClosures(t *testing.T) {
	tape := &mockTape{}
	ctx := tensor.ContextWithRecorder(context.Background(), tape)

	// A: [2, 2], requiresGrad = true
	a, _ := tensor.New(2, 2)
	a.SetRequiresGrad(true)
	_ = a.Set(1.0, 0, 0)
	_ = a.Set(2.0, 0, 1)
	_ = a.Set(3.0, 1, 0)
	_ = a.Set(4.0, 1, 1)

	// B: [2, 2], requiresGrad = false
	b, _ := tensor.New(2, 2)
	_ = b.Set(5.0, 0, 0)
	_ = b.Set(6.0, 0, 1)
	_ = b.Set(7.0, 1, 0)
	_ = b.Set(8.0, 1, 1)

	// C = A * B (MatMul)
	c, err := tensor.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("failed MatMul: %v", err)
	}

	if !c.RequiresGrad() {
		t.Error("expected output tensor C to inherit requiresGrad = true")
	}

	if len(tape.records) != 1 {
		t.Fatalf("expected 1 record on mock tape, got %d", len(tape.records))
	}

	record := tape.records[0]
	if record.op != "MatMul" {
		t.Errorf("expected recorded op 'MatMul', got '%s'", record.op)
	}

	// Simulate backward pass on C with dL/dC = Ones(2, 2)
	gradC, _ := tensor.New(2, 2)
	_ = gradC.Set(1.0, 0, 0)
	_ = gradC.Set(1.0, 0, 1)
	_ = gradC.Set(1.0, 1, 0)
	_ = gradC.Set(1.0, 1, 1)
	c.SetGrad(gradC)

	// Execute backward closure
	if err := record.backwardFn(); err != nil {
		t.Fatalf("failed executing backward closure: %v", err)
	}

	// dL/dA = dL/dC * B^T
	// B^T = [[5, 7], [6, 8]]
	// dL/dA = [[1, 1], [1, 1]] * [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
	val00, _ := a.Grad().At(0, 0)
	val01, _ := a.Grad().At(0, 1)
	if val00 != 11.0 || val01 != 15.0 {
		t.Errorf("expected dL/dA at (0,0)=11 and (0,1)=15, got %f and %f", val00, val01)
	}
}
