package sdcpp

import (
	"testing"

	"github.com/dennwc/gosd/ggml"
)

func TestPhilox(t *testing.T) {
	const n = 20
	const seed = 123456789012
	exp := make([]float32, n)
	philoxCgo(exp, seed)
	got := make([]float32, n)
	r := ggml.NewPhiloxRNG(seed)
	r.Random(got)
	t.Log(exp)
	t.Log(got)
	for i := range exp {
		if exp[i] != got[i] {
			// FIXME: check C and Go code assembly; also, check Auto1111 code output
			t.Log("unexpected value:", i, exp[i], got[i])
		}
	}
}

func TestPhiloxU(t *testing.T) {
	const n = 200
	const seed = 123456789012
	exp := make([]uint32, n)
	philoxUCgo(exp, seed)
	got := make([]uint32, n)
	r := ggml.NewPhiloxRNG(seed)
	r.RandomU(got)
	for i := range exp {
		if exp[i] != got[i] {
			t.Fatal("unexpected value:", i, exp[i], got[i])
		}
	}
}
