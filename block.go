package clockwork

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func init() {
	var d denseList
	serializer.RegisterTypedDeserializer(d.SerializerType(), deserializeDenseList)
	var b Block
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBlock)
}

type blockState struct {
	states   linalg.Vector
	timestep int
}

type blockRState struct {
	states   linalg.Vector
	statesR  linalg.Vector
	timestep int
}

// Block is an rnn.Block which implements the traditional
// Clockwork RNN architecture.
type Block struct {
	stateTransformers denseList
	inputTransformers denseList
	squasher          neuralnet.Network
	metadata          struct {
		Frequencies    []int
		InitState      *autofunc.Variable
		FullyConnected bool
	}
}

// DeserializeBlock deserializes a block.
func DeserializeBlock(d []byte) (*Block, error) {
	var res Block
	var jsonData serializer.Bytes
	err := serializer.DeserializeAny(d, &res.stateTransformers, &res.inputTransformers,
		&res.squasher, &jsonData)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(jsonData, &res.metadata); err != nil {
		return nil, err
	}
	return &res, nil
}

// NewBlock creates a traditional clockwork RNN with the
// specified frequencies (which needn't be sorted).
func NewBlock(inSize int, freqs []int, stateSizes []int) *Block {
	freqs, stateSizes = sortFreqs(freqs, stateSizes)
	res := &Block{
		stateTransformers: make(denseList, len(freqs)),
		inputTransformers: make(denseList, len(freqs)),
	}
	var totalState int
	for i := len(freqs) - 1; i >= 0; i-- {
		size := stateSizes[i]
		totalState += size
		trans := neuralnet.NewDenseLayer(totalState, size)
		inTrans := neuralnet.NewDenseLayer(inSize, size)
		res.stateTransformers[i] = trans
		res.inputTransformers[i] = inTrans
	}
	res.squasher = neuralnet.Network{
		&neuralnet.HyperbolicTangent{},
	}
	res.metadata.Frequencies = freqs
	res.metadata.InitState = &autofunc.Variable{
		Vector: make(linalg.Vector, totalState),
	}
	return res
}

// NewBlockFC creates a fully-connected clockwork RNN,
// in which higher and lower frequency blocks pass info
// back and forth.
func NewBlockFC(inSize int, freqs []int, stateSizes []int) *Block {
	freqs, stateSizes = sortFreqs(freqs, stateSizes)
	res := &Block{
		stateTransformers: make(denseList, len(freqs)),
		inputTransformers: make(denseList, len(freqs)),
	}
	var totalState int
	for _, x := range stateSizes {
		totalState += x
	}
	for i, state := range stateSizes {
		st := neuralnet.NewDenseLayer(totalState, state)
		res.stateTransformers[i] = st
		it := neuralnet.NewDenseLayer(inSize, state)
		res.inputTransformers[i] = it
	}
	res.squasher = neuralnet.Network{
		&neuralnet.HyperbolicTangent{},
	}
	res.metadata.Frequencies = freqs
	res.metadata.InitState = &autofunc.Variable{
		Vector: make(linalg.Vector, totalState),
	}
	res.metadata.FullyConnected = true
	return res
}

// StartState returns the initial state.
func (b *Block) StartState() rnn.State {
	return &blockState{
		timestep: 0,
		states:   b.metadata.InitState.Vector,
	}
}

// StartRState returns the initial r-state.
func (b *Block) StartRState(rv autofunc.RVector) rnn.RState {
	v := autofunc.NewRVariable(b.metadata.InitState, rv)
	return &blockRState{
		timestep: 0,
		states:   v.Output(),
		statesR:  v.ROutput(),
	}
}

// PropagateStart propagates through the start state.
func (b *Block) PropagateStart(s []rnn.State, u []rnn.StateGrad, g autofunc.Gradient) {
	rnn.PropagateVarState(b.metadata.InitState, u, g)
}

// PropagateStartR propagates through the start state.
func (b *Block) PropagateStartR(s []rnn.RState, u []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) {
	rnn.PropagateVarStateR(b.metadata.InitState, u, rg, g)
}

// ApplyBlock applies the block.
//
// The input states must be in sync, meaning that they are
// all from the same timestep mod the LCM of the
// frequencies.
func (b *Block) ApplyBlock(s []rnn.State, in []autofunc.Result) rnn.BlockResult {
	var statePool []*autofunc.Variable
	var stateRes []autofunc.Result
	for _, state := range s {
		stateObj := state.(*blockState)
		p := &autofunc.Variable{Vector: stateObj.states}
		statePool = append(statePool, p)
		stateRes = append(stateRes, p)
	}

	newStates := make([]autofunc.Result, len(b.metadata.Frequencies))
	for i := range b.metadata.Frequencies {
		newStates[i] = b.applySubBlock(i, s, stateRes, in)
	}
	joinedRes := b.weaveStates(len(in), newStates)

	splitOut := autofunc.Split(len(in), joinedRes)
	var outVecs []linalg.Vector
	var outStates []rnn.State
	for i, x := range splitOut {
		outVecs = append(outVecs, x.Output())
		outStates = append(outStates, &blockState{
			states:   x.Output(),
			timestep: s[i].(*blockState).timestep + 1,
		})
	}

	return &blockResult{
		StatePool: statePool,
		Results:   joinedRes,
		OutVecs:   outVecs,
		OutStates: outStates,
	}
}

// ApplyBlockR is like ApplyBlock but for RResults.
func (b *Block) ApplyBlockR(rv autofunc.RVector, s []rnn.RState,
	in []autofunc.RResult) rnn.BlockRResult {
	var statePool []*autofunc.Variable
	var stateRes []autofunc.RResult
	for _, state := range s {
		stateObj := state.(*blockRState)
		p := &autofunc.Variable{Vector: stateObj.states}
		statePool = append(statePool, p)
		stateRes = append(stateRes, &autofunc.RVariable{
			Variable:   p,
			ROutputVec: stateObj.statesR,
		})
	}

	newStates := make([]autofunc.RResult, len(b.metadata.Frequencies))
	for i := range b.metadata.Frequencies {
		newStates[i] = b.applySubBlockR(rv, i, s, stateRes, in)
	}
	joinedRes := b.weaveStatesR(len(in), newStates)

	splitOut := autofunc.SplitR(len(in), joinedRes)
	var outVecs, outVecsR []linalg.Vector
	var outStates []rnn.RState
	for i, x := range splitOut {
		outVecs = append(outVecs, x.Output())
		outVecsR = append(outVecsR, x.ROutput())
		outStates = append(outStates, &blockRState{
			states:   x.Output(),
			statesR:  x.ROutput(),
			timestep: s[i].(*blockRState).timestep + 1,
		})
	}

	return &blockRResult{
		StatePool: statePool,
		Results:   joinedRes,
		OutVecs:   outVecs,
		ROutVecs:  outVecsR,
		OutStates: outStates,
	}
}

// OutSize returns the total vector size of the block's
// output vectors (the sum of the sub-block state sizes).
func (b *Block) OutSize() int {
	var sum int
	for _, x := range b.inputTransformers {
		sum += x.Weights.Rows
	}
	return sum
}

// Frequencies returns the frequencies of the internal
// blocks, sorted from lowest to highest.
func (b *Block) Frequencies() []int {
	return append([]int{}, b.metadata.Frequencies...)
}

// Parameters returns the parameters of the CWRNN.
func (b *Block) Parameters() []*autofunc.Variable {
	res := []*autofunc.Variable{b.metadata.InitState}
	res = append(res, b.stateTransformers.Parameters()...)
	res = append(res, b.inputTransformers.Parameters()...)
	return append(res, b.squasher.Parameters()...)
}

// SerializerType returns the unique ID used to serialize
// a Block with the serializer package.
func (b *Block) SerializerType() string {
	return "github.com/unixpickle/clockwork.Block"
}

// Serialize serializes a block.
func (b *Block) Serialize() ([]byte, error) {
	jsonData, err := json.Marshal(&b.metadata)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeAny(b.stateTransformers, b.inputTransformers,
		b.squasher, serializer.Bytes(jsonData))
}

func (b *Block) applySubBlock(subIdx int, inState []rnn.State, pool []autofunc.Result,
	in []autofunc.Result) autofunc.Result {
	stateIdx := 0
	for i := 0; i < subIdx; i++ {
		stateIdx += b.stateTransformers[i].Weights.Rows
	}
	stateSize := b.stateTransformers[subIdx].Weights.Rows
	freq := b.metadata.Frequencies[subIdx]

	var transformStates []autofunc.Result
	var transformIns []autofunc.Result
	for i, s := range inState {
		if s.(*blockState).timestep%freq == 0 {
			transformIns = append(transformIns, in[i])
			if b.metadata.FullyConnected {
				transformStates = append(transformStates, pool[i])
			} else {
				inSize := b.stateTransformers[subIdx].Weights.Cols
				slowerOrSame := autofunc.Slice(pool[i], stateIdx, stateIdx+inSize)
				transformStates = append(transformStates, slowerOrSame)
			}
		}
	}

	if len(transformStates) == 0 {
		var prevStates []autofunc.Result
		for _, x := range pool {
			p := autofunc.Slice(x, stateIdx, stateIdx+stateSize)
			prevStates = append(prevStates, p)
		}
		return autofunc.Concat(prevStates...)
	} else if len(transformStates) != len(in) {
		panic("input states are out of sync")
	}

	transformState := autofunc.Concat(transformStates...)
	transformIn := autofunc.Concat(transformIns...)

	return b.squasher.Apply(
		autofunc.Add(
			b.stateTransformers[subIdx].Batch(transformState, len(transformIns)),
			b.inputTransformers[subIdx].Batch(transformIn, len(transformIns)),
		),
	)
}

func (b *Block) applySubBlockR(rv autofunc.RVector, subIdx int, inState []rnn.RState,
	pool []autofunc.RResult, in []autofunc.RResult) autofunc.RResult {
	stateIdx := 0
	for i := 0; i < subIdx; i++ {
		stateIdx += b.stateTransformers[i].Weights.Rows
	}
	stateSize := b.stateTransformers[subIdx].Weights.Rows
	freq := b.metadata.Frequencies[subIdx]

	var transformStates []autofunc.RResult
	var transformIns []autofunc.RResult
	for i, s := range inState {
		if s.(*blockRState).timestep%freq == 0 {
			transformIns = append(transformIns, in[i])
			if b.metadata.FullyConnected {
				transformStates = append(transformStates, pool[i])
			} else {
				inSize := b.stateTransformers[subIdx].Weights.Cols
				slowerOrSame := autofunc.SliceR(pool[i], stateIdx, stateIdx+inSize)
				transformStates = append(transformStates, slowerOrSame)
			}
		}
	}

	if len(transformStates) == 0 {
		var prevStates []autofunc.RResult
		for _, x := range pool {
			p := autofunc.SliceR(x, stateIdx, stateIdx+stateSize)
			prevStates = append(prevStates, p)
		}
		return autofunc.ConcatR(prevStates...)
	} else if len(transformStates) != len(in) {
		panic("input states are out of sync")
	}

	transformState := autofunc.ConcatR(transformStates...)
	transformIn := autofunc.ConcatR(transformIns...)

	return b.squasher.ApplyR(
		rv,
		autofunc.AddR(
			b.stateTransformers[subIdx].BatchR(rv, transformState, len(transformIns)),
			b.inputTransformers[subIdx].BatchR(rv, transformIn, len(transformIns)),
		),
	)
}

func (b *Block) weaveStates(numIn int, subStates []autofunc.Result) autofunc.Result {
	return autofunc.PoolAll(subStates, func(subStates []autofunc.Result) autofunc.Result {
		splitSub := make([][]autofunc.Result, len(subStates))
		for i, s := range subStates {
			splitSub[i] = autofunc.Split(numIn, s)
		}
		var ordered []autofunc.Result
		for i := 0; i < numIn; i++ {
			for _, x := range splitSub {
				ordered = append(ordered, x[i])
			}
		}
		return autofunc.Concat(ordered...)
	})
}

func (b *Block) weaveStatesR(numIn int, subStates []autofunc.RResult) autofunc.RResult {
	return autofunc.PoolAllR(subStates, func(subStates []autofunc.RResult) autofunc.RResult {
		splitSub := make([][]autofunc.RResult, len(subStates))
		for i, s := range subStates {
			splitSub[i] = autofunc.SplitR(numIn, s)
		}
		var ordered []autofunc.RResult
		for i := 0; i < numIn; i++ {
			for _, x := range splitSub {
				ordered = append(ordered, x[i])
			}
		}
		return autofunc.ConcatR(ordered...)
	})
}

type blockResult struct {
	StatePool []*autofunc.Variable
	Results   autofunc.Result
	OutVecs   []linalg.Vector
	OutStates []rnn.State
}

func (b *blockResult) Outputs() []linalg.Vector {
	return b.OutVecs
}

func (b *blockResult) States() []rnn.State {
	return b.OutStates
}

func (b *blockResult) PropagateGradient(u []linalg.Vector, s []rnn.StateGrad,
	g autofunc.Gradient) []rnn.StateGrad {
	var outUp linalg.Vector
	for i := range b.OutStates {
		var vec linalg.Vector
		if u != nil {
			vec = u[i].Copy()
		} else {
			vec = make(linalg.Vector, len(b.OutVecs[i]))
		}
		if s != nil && s[i] != nil {
			vec.Add(linalg.Vector(s[i].(rnn.VecStateGrad)))
		}
		outUp = append(outUp, vec...)
	}
	return rnn.PropagateVecStatePool(g, b.StatePool, func() {
		b.Results.PropagateGradient(outUp, g)
	})
}

type blockRResult struct {
	StatePool []*autofunc.Variable
	Results   autofunc.RResult
	OutVecs   []linalg.Vector
	ROutVecs  []linalg.Vector
	OutStates []rnn.RState
}

func (b *blockRResult) Outputs() []linalg.Vector {
	return b.OutVecs
}

func (b *blockRResult) ROutputs() []linalg.Vector {
	return b.ROutVecs
}

func (b *blockRResult) RStates() []rnn.RState {
	return b.OutStates
}

func (b *blockRResult) PropagateRGradient(u, uR []linalg.Vector, s []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []rnn.RStateGrad {
	var outUp, outUpR linalg.Vector
	for i := range b.OutStates {
		var vec, vecR linalg.Vector
		if u != nil {
			vec = u[i].Copy()
			vecR = uR[i].Copy()
		} else {
			vec = make(linalg.Vector, len(b.OutVecs[i]))
			vecR = make(linalg.Vector, len(b.OutVecs[i]))
		}
		if s != nil && s[i] != nil {
			vrsg := s[i].(rnn.VecRStateGrad)
			vec.Add(vrsg.State)
			vecR.Add(vrsg.RState)
		}
		outUp = append(outUp, vec...)
		outUpR = append(outUpR, vecR...)
	}
	return rnn.PropagateVecRStatePool(rg, g, b.StatePool, func() {
		b.Results.PropagateRGradient(outUp, outUpR, rg, g)
	})
}

type denseList []*neuralnet.DenseLayer

func deserializeDenseList(d []byte) (denseList, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	res := make(denseList, len(slice))
	for i, x := range slice {
		var ok bool
		res[i], ok = x.(*neuralnet.DenseLayer)
		if !ok {
			return nil, fmt.Errorf("not a DenseLayer: %T", x)
		}
	}
	return res, nil
}

func (d denseList) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, x := range d {
		res = append(res, x.Parameters()...)
	}
	return res
}

func (d denseList) SerializerType() string {
	return "github.com/unixpickle/clockwork.denseList"
}

func (d denseList) Serialize() ([]byte, error) {
	s := make([]serializer.Serializer, len(d))
	for i, x := range d {
		s[i] = x
	}
	return serializer.SerializeSlice(s)
}

type freqSorter struct {
	freqs []int
	sizes []int
}

func sortFreqs(freqs, sizes []int) (newFreqs, newSizes []int) {
	s := freqSorter{
		freqs: append([]int{}, freqs...),
		sizes: append([]int{}, sizes...),
	}
	sort.Sort(&s)
	return s.freqs, s.sizes
}

func (f *freqSorter) Len() int {
	return len(f.freqs)
}

func (f *freqSorter) Swap(i, j int) {
	f.freqs[i], f.freqs[j] = f.freqs[j], f.freqs[i]
	f.sizes[i], f.sizes[j] = f.sizes[j], f.sizes[i]
}

func (f *freqSorter) Less(i, j int) bool {
	return f.freqs[i] < f.freqs[j]
}
