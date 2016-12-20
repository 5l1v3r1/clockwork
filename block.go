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

type blockStateGrad struct {
	states linalg.Vector
}

type blockRStateGrad struct {
	states  linalg.Vector
	statesR linalg.Vector
}

// Block is an rnn.Block which implements the traditional
// Clockwork RNN architecture.
type Block struct {
	stateTransformers denseList         `json:"-"`
	inputTransformers denseList         `json:"-"`
	squasher          neuralnet.Network `json:"-"`
	frequencies       []int
	initState         *autofunc.Variable
	fullyConnected    bool
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
	if err := json.Unmarshal(jsonData, &res); err != nil {
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
	res.frequencies = freqs
	res.initState = &autofunc.Variable{
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
	res.frequencies = freqs
	res.initState = &autofunc.Variable{
		Vector: make(linalg.Vector, totalState),
	}
	res.fullyConnected = true
	return res
}

// StartState returns the initial state.
func (b *Block) StartState() rnn.State {
	return &blockState{
		timestep: 0,
		states:   b.initState.Vector,
	}
}

// StartRState returns the initial r-state.
func (b *Block) StartRState(rv autofunc.RVector) rnn.RState {
	v := autofunc.NewRVariable(b.initState, rv)
	return &blockRState{
		timestep: 0,
		states:   v.Output(),
		statesR:  v.ROutput(),
	}
}

// PropagateStart propagates through the start state.
func (b *Block) PropagateStart(s []rnn.State, u []rnn.StateGrad, g autofunc.Gradient) {
	for _, sg := range u {
		grad := sg.(*blockStateGrad)
		b.initState.PropagateGradient(grad.states, g)
	}
}

// PropagateStartR propagates through the start state.
func (b *Block) PropagateStartR(s []rnn.RState, u []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) {
	rv := &autofunc.RVariable{Variable: b.initState}
	for _, sg := range u {
		grad := sg.(*blockRStateGrad)
		up := grad.states
		upR := grad.statesR
		rv.PropagateRGradient(up, upR, rg, g)
	}
}

// Frequencies returns the frequencies of the internal
// blocks, sorted from lowest to highest.
func (b *Block) Frequencies() []int {
	return append([]int{}, b.frequencies...)
}

// Parameters returns the parameters of the CWRNN.
func (b *Block) Parameters() []*autofunc.Variable {
	res := []*autofunc.Variable{b.initState}
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
	jsonData, err := json.Marshal(b)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeAny(b.stateTransformers, b.inputTransformers,
		b.squasher, serializer.Bytes(jsonData))
}

type blockResult struct {
	StatePool []*autofunc.Variable
	Results   []autofunc.Result
	OutVecs   []linalg.Vector
	OutStates []rnn.State
}

func (b *blockResult) Outputs() []linalg.Vector {
	return b.OutVecs
}

func (b *blockResult) States() []rnn.State {
	return b.OutStates
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
