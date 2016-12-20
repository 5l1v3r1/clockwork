package clockwork

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func TestBlockInit(t *testing.T) {
	block := NewBlock(3, []int{1, 3, 2}, []int{1, 1, 2})
	if !reflect.DeepEqual(block.Frequencies(), []int{1, 2, 3}) {
		t.Errorf("frequencies not sorted correctly: %v", block.Frequencies())
	}

	stateSizes := []int{
		block.stateTransformers[0].Weights.Rows,
		block.stateTransformers[1].Weights.Rows,
		block.stateTransformers[2].Weights.Rows,
	}
	if !reflect.DeepEqual(stateSizes, []int{1, 2, 1}) {
		t.Errorf("state sizes should be %v but got %v", []int{1, 2, 1}, stateSizes)
	}
}

func TestBlock(t *testing.T) {
	for _, fc := range []bool{false, true} {
		t.Run(fmt.Sprintf("fc%v", fc), func(t *testing.T) {
			var block *Block
			if fc {
				block = NewBlockFC(3, []int{1, 3, 2}, []int{1, 1, 2})
			} else {
				block = NewBlock(3, []int{1, 3, 2}, []int{1, 1, 2})
			}
			seqs := [][]*autofunc.Variable{
				{{Vector: []float64{1, -1, 1}}},
				{{Vector: []float64{1, -1, 1}}, {Vector: []float64{1, 1, -1}},
					{Vector: []float64{-1, 1, 1}}},
				{{Vector: []float64{1, -1, 1}}, {Vector: []float64{1, 1, -1}},
					{Vector: []float64{-1, 1, 1}}, {Vector: []float64{0, 1, -1}}},
			}
			vars := block.Parameters()
			for _, x := range seqs {
				for _, y := range x {
					vars = append(vars, y)
				}
			}
			rv := autofunc.RVector{}
			for _, v := range vars {
				rv[v] = make(linalg.Vector, len(v.Vector))
				for i := range rv[v] {
					rv[v][i] = rand.NormFloat64()
				}
			}
			checker := functest.SeqRFuncChecker{
				F:     &rnn.BlockSeqFunc{B: block},
				Vars:  vars,
				Input: seqs,
				RV:    rv,
			}
			checker.FullCheck(t)
		})
	}
}
