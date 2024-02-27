package ggml

import "math/rand"

type RNG interface {
	Seed(seed int64)
	Random(dst []float32)
}

func NewStdRNG(seed int64) *StdRNG {
	r := new(StdRNG)
	r.Seed(seed)
	return r
}

type StdRNG struct {
	rng *rand.Rand
}

func (r *StdRNG) Seed(seed int64) {
	if seed < 0 {
		seed = rand.Int63()
	}
	r.rng = rand.New(rand.NewSource(seed))
}

func (r *StdRNG) Random(dst []float32) {
	for i := range dst {
		dst[i] = float32(r.rng.NormFloat64())
	}
}
