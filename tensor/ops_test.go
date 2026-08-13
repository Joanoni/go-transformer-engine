package tensor_test

import (
	"testing"

	"github.com/Joanoni/go-transformer-engine/tensor"
)

func TestAdd_MatchingShapes(t *testing.T) {
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

	c, err := tensor.Add(a, b)
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
	a, _ := tensor.New(2, 3)
	b, _ := tensor.New(2, 2)

	_, err := tensor.Add(a, b)
	if err == nil {
		t.Error("expected shape mismatch error, got nil")
	}
}

func TestSub_ElementWise(t *testing.T) {
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

	c, err := tensor.Sub(a, b)
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

	c, err := tensor.Mul(a, b)
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
	// A: Shape [2, 3]
	// [1.0, 2.0, 3.0]
	// [4.0, 5.0, 6.0]
	a, _ := tensor.New(2, 3)
	_ = a.Set(1.0, 0, 0)
	_ = a.Set(2.0, 0, 1)
	_ = a.Set(3.0, 0, 2)
	_ = a.Set(4.0, 1, 0)
	_ = a.Set(5.0, 1, 1)
	_ = a.Set(6.0, 1, 2)

	// B: Shape [3, 2]
	// [7.0,  8.0]
	// [9.0,  10.0]
	// [11.0, 12.0]
	b, _ := tensor.New(3, 2)
	_ = b.Set(7.0, 0, 0)
	_ = b.Set(8.0, 0, 1)
	_ = b.Set(9.0, 1, 0)
	_ = b.Set(10.0, 1, 1)
	_ = b.Set(11.0, 2, 0)
	_ = b.Set(12.0, 2, 1)

	// C = A * B -> Shape [2, 2]
	// C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	// C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
	// C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	// C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
	c, err := tensor.MatMul(a, b)
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

func TestMatMul_DimensionMismatch(t *testing.T) {
	a, _ := tensor.New(2, 3)
	b, _ := tensor.New(2, 2) // Inner dims: 3 vs 2 (Mismatch)

	_, err := tensor.MatMul(a, b)
	if err == nil {
		t.Error("expected dimension mismatch error for MatMul, got nil")
	}
}

func TestTranspose_ZeroCopy(t *testing.T) {
	// Matrix of shape [2, 3] -> strides [3, 1]
	a, _ := tensor.New(2, 3)
	_ = a.Set(99.0, 0, 1) // Row 0, Col 1

	// Transpose dimensions 0 and 1 -> Shape [3, 2], strides [1, 3]
	at, err := tensor.Transpose(a, 0, 1)
	if err != nil {
		t.Fatalf("unexpected error during Transpose: %v", err)
	}

	if at.Shape()[0] != 3 || at.Shape()[1] != 2 {
		t.Errorf("expected transposed shape [3, 2], got %v", at.Shape())
	}

	// In transposed matrix, (0, 1) of original becomes (1, 0)
	val, err := at.At(1, 0)
	if err != nil {
		t.Fatalf("failed to read from transposed tensor: %v", err)
	}
	if val != 99.0 {
		t.Errorf("expected 99.0 at transposed (1, 0), got %f", val)
	}

	// Verify Zero-Copy behavior: modifying `at` modifies `a`
	_ = at.Set(77.0, 1, 0)
	origVal, _ := a.At(0, 1)
	if origVal != 77.0 {
		t.Errorf("zero-copy verification failed: modifying transposed view did not update original tensor. Got %f", origVal)
	}
}

func TestReshape_ZeroCopy(t *testing.T) {
	// Vector of shape [6]
	a, _ := tensor.New(6)
	for i := 0; i < 6; i++ {
		_ = a.Set(float64(i+1), i)
	}

	// Reshape to [2, 3]
	b, err := tensor.Reshape(a, 2, 3)
	if err != nil {
		t.Fatalf("unexpected error during Reshape: %v", err)
	}

	val, _ := b.At(1, 1) // Should be 5th element (index 4 in 1D -> 5.0)
	if val != 5.0 {
		t.Errorf("expected 5.0 at (1, 1), got %f", val)
	}
}

func TestApply(t *testing.T) {
	a, _ := tensor.New(2, 2)
	_ = a.Set(1.0, 0, 0)
	_ = a.Set(2.0, 0, 1)
	_ = a.Set(3.0, 1, 0)
	_ = a.Set(4.0, 1, 1)

	// Double every element
	out, err := tensor.Apply(a, func(x float64) float64 {
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
