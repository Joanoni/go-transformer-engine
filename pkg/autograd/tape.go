package autograd

import (
	"context"
	"errors"
	"fmt"

	"github.com/Joanoni/go-transformer-engine/pkg/tensor"
)

var (
	ErrEmptyTape     = errors.New("cannot perform backward pass on an empty tape")
	ErrNilLossTensor = errors.New("loss tensor provided to Backward cannot be nil")
)

// Node represents a single recorded operation step in the dynamic computation graph.
type Node struct {
	Op         string
	Out        *tensor.Tensor
	Inputs     []*tensor.Tensor
	BackwardFn func() error
}

// Tape (GradientTape) records dynamic tensor operations executed during the forward pass.
// Reversing the execution sequence naturally provides reverse topological ordering for the backward pass.
type Tape struct {
	nodes []Node
}

// NewTape allocates a new empty GradientTape with initial capacity.
func NewTape() *Tape {
	return &Tape{
		nodes: make([]Node, 0, 16),
	}
}

// WithTape wraps parent context with a new active Tape recorder.
func WithTape(parent context.Context) (context.Context, *Tape) {
	tape := NewTape()
	ctx := tensor.ContextWithRecorder(parent, tape)
	return ctx, tape
}

// Record appends a new operation node onto the tape, satisfying tensor.Recorder interface.
func (t *Tape) Record(op string, out *tensor.Tensor, inputs []*tensor.Tensor, backwardFn func() error) {
	if t == nil {
		return
	}
	t.nodes = append(t.nodes, Node{
		Op:         op,
		Out:        out,
		Inputs:     inputs,
		BackwardFn: backwardFn,
	})
}

// Nodes returns a copy of all recorded operation nodes.
func (t *Tape) Nodes() []Node {
	if t == nil {
		return nil
	}
	cp := make([]Node, len(t.nodes))
	copy(cp, t.nodes)
	return cp
}

// Len returns the total number of operations recorded on the tape.
func (t *Tape) Len() int {
	if t == nil {
		return 0
	}
	return len(t.nodes)
}

// Clear releases all recorded nodes and closures, resetting the tape
// and permitting GC or CUDA memory pools to collect intermediate tensors.
func (t *Tape) Clear() {
	if t == nil {
		return
	}
	t.nodes = nil
}

// Backward executes automatic differentiation by traversing the recorded graph
// in reverse topological order (from loss back to inputs/parameters).
// If loss tensor gradient is nil, a default seed gradient of ones (1.0) is initialized.
func (t *Tape) Backward(loss *tensor.Tensor) error {
	if t == nil || len(t.nodes) == 0 {
		return ErrEmptyTape
	}
	if loss == nil {
		return ErrNilLossTensor
	}

	// Initialize seed gradient dL/dL = 1.0 if not provided
	if loss.Grad() == nil {
		seedGrad, err := tensor.New(loss.Shape()...)
		if err != nil {
			return fmt.Errorf("failed to allocate seed gradient for loss tensor: %w", err)
		}
		for i := 0; i < seedGrad.Size(); i++ {
			seedGrad.Data()[i] = 1.0
		}
		loss.SetGrad(seedGrad)
	}

	// Traverse the tape in reverse topological order
	for i := len(t.nodes) - 1; i >= 0; i-- {
		node := t.nodes[i]
		if node.BackwardFn != nil {
			if err := node.BackwardFn(); err != nil {
				return fmt.Errorf("error executing backward for op '%s' at node %d: %w", node.Op, i, err)
			}
		}
	}

	return nil
}
