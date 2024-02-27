package ggml

import (
	"math"
	"math/rand"
)

var (
	philox_m = [2]uint32{0xD2511F53, 0xCD9E8D57}
	philox_w = [2]uint32{0x9E3779B9, 0xBB67AE85}
)

const (
	two_pow32_inv     = 2.3283064e-10
	two_pow32_inv_2pi = 2.3283064e-10 * 6.2831855
)

func NewPhiloxRNG(seed int64) *PhiloxRNG {
	r := &PhiloxRNG{}
	r.Seed(seed)
	return r
}

type PhiloxRNG struct {
	seed   uint64
	offset uint32
}

func (r *PhiloxRNG) Seed(seed int64) {
	if seed < 0 {
		seed = rand.Int63()
	}
	r.seed = uint64(seed)
	r.offset = 0
}

func (r *PhiloxRNG) Random(dst []float32) {
	n := len(dst)
	counter := make([][4]uint32, n)
	for i := 0; i < n; i++ {
		counter[i][0] = r.offset
	}
	for i := 0; i < n; i++ {
		counter[i][2] = uint32(i)
	}
	r.offset++
	key := make([][2]uint32, n)
	for i := range key {
		key[i] = splitU64(r.seed)
	}

	philox4_32(counter, key)
	for i := range dst {
		dst[i] = box_muller(float32(counter[i][0]), float32(counter[i][1]))
	}
}

func (r *PhiloxRNG) RandomU(dst []uint32) {
	n := len(dst) / 2
	counter := make([][4]uint32, n)
	for i := 0; i < n; i++ {
		counter[i][0] = r.offset
	}
	for i := 0; i < n; i++ {
		counter[i][2] = uint32(i)
	}
	r.offset++
	key := make([][2]uint32, n)
	for i := range key {
		key[i] = splitU64(r.seed)
	}

	philox4_32(counter, key)
	for i := 0; i < n; i++ {
		dst[2*i+0] = counter[i][0]
		dst[2*i+1] = counter[i][1]
	}
}

func splitU64(x uint64) [2]uint32 {
	return [2]uint32{uint32(x & 0xFFFFFFFF), uint32(x >> 32)}
}

// philox4_round is a single round of the Philox 4x32 random number generator.
func philox4_round(counter [][4]uint32, key [][2]uint32) {
	n := len(counter)
	for i := 0; i < n; i++ {
		v1 := splitU64(uint64(counter[i][0]) * uint64(philox_m[0]))
		v2 := splitU64(uint64(counter[i][2]) * uint64(philox_m[1]))

		counter[i][0] = v2[1] ^ counter[i][1] ^ key[i][0]
		counter[i][1] = v2[0]
		counter[i][2] = v1[1] ^ counter[i][3] ^ key[i][1]
		counter[i][3] = v1[0]
	}
}

// philox4_32 generates 32-bit random numbers using the Philox 4x32 random number generator.
// Parameters:
//
//	counter : A 4xN array of 32-bit integers representing the counter values (offset into generation).
//	key : A 2xN array of 32-bit integers representing the key values (seed).
//	rounds : The number of rounds to perform.
//
// Returns: A 4xN array of 32-bit integers containing the generated random numbers.
func philox4_32(counter [][4]uint32, key [][2]uint32) {
	const rounds = 10
	n := len(counter)
	for i := 0; i < rounds-1; i++ {
		philox4_round(counter, key)
		for j := 0; j < n; j++ {
			key[j][0] += philox_w[0]
			key[j][1] += philox_w[1]
		}
	}
	philox4_round(counter, key)
}

func box_muller(x, y float32) float32 {
	u := x*two_pow32_inv + two_pow32_inv/2
	v := y*two_pow32_inv_2pi + two_pow32_inv_2pi/2

	s := math.Sqrt(float64(-2.0 * float32(math.Log(float64(u)))))

	r1 := s * math.Sin(float64(v))
	return float32(r1)
}
