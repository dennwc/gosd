package llama2

type Config struct {
	Dim       int // transformer dimension
	HiddenDim int // for ffn layers
	Layers    int // number of layers
	Heads     int // number of query heads
	KVHeads   int // number of key/value heads (can be < query heads because of multiquery)
	VocabSize int // vocabulary size, usually 256 (byte-level)
	SeqLen    int // max sequence length
}

type TransformerWeights struct {
	// token embedding table
	TokenEmbeddingTable []float32 // (vocab_size, dim)
	// weights for rmsnorms
	RmsAttWeight []float32 // (layer, dim) rmsnorm weights
	RmsFfnWeight []float32 // (layer, dim)
	// weights for matmuls. note dim == n_heads * head_size
	Wq []float32 // (layer, dim, n_heads * head_size)
	Wk []float32 // (layer, dim, n_kv_heads * head_size)
	Wv []float32 // (layer, dim, n_kv_heads * head_size)
	Wo []float32 // (layer, n_heads * head_size, dim)
	// weights for ffn
	W1 []float32 // (layer, hidden_dim, dim)
	W2 []float32 // (layer, dim, hidden_dim)
	W3 []float32 // (layer, hidden_dim, dim)
	// final rmsnorm
	RmsFinalWeight []float32 // (dim,)
	// (optional) classifier weights for the logits, on the last layer
	WCls []float32
}

type Model struct {
	Config  Config             // the hyperparameters of the architecture (the blueprint)
	Weights TransformerWeights // the weights of the model
	close   func()             // some more state needed to properly clean up the memory mapping (sigh)
}

func (m *Model) Close() {
	if m.close != nil {
		m.close()
		m.close = nil
	}
}

func ReadCheckpoint(checkpoint string) (*Model, error) {
	mmap, unmap, err := Memmap(checkpoint)
	if err != nil {
		return nil, err
	}

	hdr, err := AsSlice[int32](mmap)
	if err != nil {
		unmap()
		return nil, err
	}

	weights, err := AsSlice[float32](mmap[7*4:])
	if err != nil {
		unmap()
		return nil, err
	}
	m := &Model{
		Config: Config{
			Dim:       int(hdr[0]),
			HiddenDim: int(hdr[1]),
			Layers:    int(hdr[2]),
			Heads:     int(hdr[3]),
			KVHeads:   int(hdr[4]),
			VocabSize: int(hdr[5]),
			SeqLen:    int(hdr[6]),
		},
		close: unmap,
	}

	// negative vocab size is hacky way of signaling unshared weights. bit yikes.
	sharedWeights := m.Config.VocabSize > 0
	if m.Config.VocabSize < 0 {
		m.Config.VocabSize *= -1
	}
	// memory map the Transformer weights into the data pointer
	m.mapWeights(weights, sharedWeights)
	return m, nil
}

func (m *Model) mapWeights(ptr []float32, sharedWeights bool) {
	p := &m.Config
	w := &m.Weights
	head_size := p.Dim / p.Heads
	// make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
	n_layers := uint64(p.Layers)

	w.TokenEmbeddingTable = ptr[:p.VocabSize*p.Dim]
	ptr = ptr[len(w.TokenEmbeddingTable):]

	w.RmsAttWeight = ptr[:n_layers*uint64(p.Dim)]
	ptr = ptr[len(w.RmsAttWeight):]

	w.Wq = ptr[:n_layers*uint64(p.Dim)*(uint64(p.Heads)*uint64(head_size))]
	ptr = ptr[len(w.Wq):]

	w.Wk = ptr[:n_layers*uint64(p.Dim)*(uint64(p.KVHeads)*uint64(head_size))]
	ptr = ptr[len(w.Wk):]

	w.Wv = ptr[:n_layers*uint64(p.Dim)*(uint64(p.KVHeads)*uint64(head_size))]
	ptr = ptr[len(w.Wv):]

	w.Wo = ptr[:n_layers*(uint64(p.Heads)*uint64(head_size))*uint64(p.Dim)]
	ptr = ptr[len(w.Wo):]

	w.RmsFfnWeight = ptr[:n_layers*uint64(p.Dim)]
	ptr = ptr[len(w.RmsFfnWeight):]

	w.W1 = ptr[:n_layers*uint64(p.Dim)*uint64(p.HiddenDim)]
	ptr = ptr[len(w.W1):]

	w.W2 = ptr[:n_layers*uint64(p.HiddenDim)*uint64(p.Dim)]
	ptr = ptr[len(w.W2):]

	w.W3 = ptr[:n_layers*uint64(p.Dim)*uint64(p.HiddenDim)]
	ptr = ptr[len(w.W3):]

	w.RmsFinalWeight = ptr[:p.Dim]
	ptr = ptr[len(w.RmsFinalWeight):]

	ptr = ptr[p.SeqLen*head_size/2:] // skip what used to be freq_cis_real (for RoPE)
	ptr = ptr[p.SeqLen*head_size/2:] // skip what used to be freq_cis_imag (for RoPE)
	if sharedWeights {
		w.WCls = w.TokenEmbeddingTable
	} else {
		w.WCls = ptr
	}
}
