package llama2

type random struct {
	state uint64
}

func (r *random) U32() uint32 {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return uint32((r.state * 0x2545F4914F6CDD1D) >> 32)
}

func (r *random) F32() float32 { // random float32 in [0,1)
	return float32(r.U32()>>8) / 16777216.0
}
