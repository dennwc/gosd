package llama2

import (
	"github.com/chewxy/math32"
)

type RunState struct {
	// current wave of activations
	X      []float32   // activation at current time stamp [dim]
	Xb     []float32   // same, but inside a residual branch [dim]
	Xb2    []float32   // an additional buffer just for convenience [dim]
	Hb     []float32   // buffer for hidden dimension in the ffn [hidden_dim]
	Hb2    []float32   // buffer for hidden dimension in the ffn [hidden_dim]
	Q      []float32   // query [dim]
	K      []float32   // key [dim]
	V      []float32   // value [dim]
	Att    [][]float32 // buffer for scores/attention values [n_heads][seq_len]
	Logits []float32   // output logits
	// kv cache
	KeyCache   [][][]float32 // [layer][seq_len][dim]
	ValueCache [][][]float32 // [layer][seq_len][dim]
}

type Transformer struct {
	*Model
	State *RunState // buffers for the "wave" of activations in the forward pass
}

func NewRunState(p *Config) *RunState {
	kvDim := (p.Dim * p.KVHeads) / p.Heads
	return &RunState{
		X:          make([]float32, p.Dim),
		Xb:         make([]float32, p.Dim),
		Xb2:        make([]float32, p.Dim),
		Hb:         make([]float32, p.HiddenDim),
		Hb2:        make([]float32, p.HiddenDim),
		Q:          make([]float32, p.Dim),
		KeyCache:   make3d[float32](p.Layers, p.SeqLen, kvDim),
		ValueCache: make3d[float32](p.Layers, p.SeqLen, kvDim),
		Att:        make2d[float32](p.Heads, p.SeqLen),
		Logits:     make([]float32, p.VocabSize),
	}
}

func NewTransformer(path string) (*Transformer, error) {
	// read in the Config and the Weights from the checkpoint
	m, err := ReadCheckpoint(path)
	if err != nil {
		return nil, err
	}
	return &Transformer{
		Model: m,
		State: NewRunState(&m.Config),
	}, nil
}

func (t *Transformer) Close() {
	if t.Model != nil {
		t.Model.Close()
	}
}

func (t *Transformer) Forward(token Token, pos int) []float32 {
	// a few convenience variables
	p := &t.Config
	w := &t.Weights
	s := t.State
	x := s.X
	dim := p.Dim
	kvDim := (p.Dim * p.KVHeads) / p.Heads
	kvMul := p.Heads / p.KVHeads // integer multiplier of the kv sharing in multiquery
	headSize := dim / p.Heads

	// copy the token embedding into x
	contentRow := w.TokenEmbeddings[token]
	copy(x[:dim], contentRow[:dim])

	// forward all the layers
	for l := range p.Layers {
		keyCache := s.KeyCache[l]
		valCache := s.ValueCache[l]

		// attention rmsnorm
		rmsnorm(s.Xb[:dim], x[:dim], w.RmsAttWeight[l])

		// key and value point to the kv cache
		s.K = keyCache[pos]
		s.V = valCache[pos]

		// qkv matmuls for this position
		matmul(s.Q, s.Xb, w.Wq[l])
		matmul(s.K, s.Xb, w.Wk[l])
		matmul(s.V, s.Xb, w.Wv[l])

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		for i := 0; i < dim; i += 2 {
			headDim := i % headSize
			freq := 1.0 / math32.Pow(10000.0, float32(headDim)/float32(headSize))
			val := float32(pos) * freq
			fci, fcr := math32.Sincos(val)
			rotn := 1 // how many vectors? 2 = q & k, 1 = q only
			if i < kvDim {
				rotn = 2
			}
			for v := range rotn {
				vec := s.Q // the vector to rotate (query or key)
				if v != 0 {
					vec = s.K
				}
				v0 := vec[i+0]
				v1 := vec[i+1]
				vec[i+0] = v0*fcr - v1*fci
				vec[i+1] = v0*fci + v1*fcr
			}
		}

		// multihead attention. iterate over all heads
		headSizeSqrt := math32.Sqrt(float32(headSize))
		// pragma omp parallel for private(h)
		for h := range p.Heads {
			// get the query vector for this head
			q := s.Q[h*headSize : (h+1)*headSize]
			// attention scores for this head
			att := s.Att[h]
			// iterate over all timesteps, including the current one
			for ti := 0; ti <= pos; ti++ {
				// get the key vector for this head and at this timestep
				k := keyCache[ti][(h/kvMul)*headSize : (h/kvMul+1)*headSize]
				// calculate the attention score as the dot product of q and k
				score := dot(q, k)
				score /= headSizeSqrt
				// save the score to the attention buffer
				att[ti] = score
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att[:pos+1])

			// weighted sum of the values, store back into xb
			xb := s.Xb[h*headSize : (h+1)*headSize]
			memset0(xb)
			for ti := 0; ti <= pos; ti++ {
				// get the value vector for this head and at this timestep
				v := valCache[ti][(h/kvMul)*headSize:]
				// get the attention weight for this timestep
				a := att[ti]
				// accumulate the weighted value into xb
				for i := range xb {
					xb[i] += a * v[i]
				}
			}
		}

		// final matmul to get the output of the attention
		matmul(s.Xb2, s.Xb, w.Wo[l])

		// residual connection back into x
		for i := range x {
			x[i] += s.Xb2[i]
		}

		// ffn rmsnorm
		rmsnorm(s.Xb[:dim], x[:dim], w.RmsFfnWeight[l])

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s.Hb, s.Xb, w.W1[l])
		matmul(s.Hb2, s.Xb, w.W3[l])

		// SwiGLU non-linearity
		swiglu(s.Hb, s.Hb2)

		// final matmul to get the output of the ffn
		matmul(s.Xb, s.Hb, w.W2[l])

		// residual connection
		for i := range x {
			x[i] += s.Xb[i]
		}
	}

	// final rmsnorm
	rmsnorm(x[:dim], x[:dim], w.RmsFinalWeight)

	// classifier into logits
	matmul(s.Logits, x, w.WCls)
	return s.Logits
}
