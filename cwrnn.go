package clockwork

import (
	"encoding/json"
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var d denseList
	serializer.RegisterTypedDeserializer(d.SerializerType(), deserializeDenseList)
	var b Block
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBlock)
}

type blockState struct {
	subStates []linalg.Vector
	timestep  int
}

type blockRState struct {
	subStates  []linalg.Vector
	subStatesR []linalg.Vector
	timestep   int
}

type blockStateGrad struct {
	subStates []linalg.Vector
}

type blockStateRGrad struct {
	subStates  []linalg.Vector
	subStatesR []linalg.Vector
}

// Block is an rnn.Block which implements the traditional
// Clockwork RNN architecture.
type Block struct {
	stateTransformers denseList         `json:"-"`
	inputTransformers denseList         `json:"-"`
	squasher          neuralnet.Network `json:"-"`
	frequencies       []int
	initState         *autofunc.Variable
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
