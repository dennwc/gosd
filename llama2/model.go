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
	TokenEmbeddings [][]float32 // [vocab][dim]
	// weights for rmsnorms
	RmsAttWeight [][]float32 // [layer][dim] rmsnorm weights
	RmsFfnWeight [][]float32 // [layer][dim]
	// weights for matmuls. note dim == n_heads * head_size
	Wq [][][]float32 // [layer][dim][n_heads * head_size]
	Wk [][][]float32 // [layer][dim][n_kv_heads * head_size]
	Wv [][][]float32 // [layer][dim][n_kv_heads * head_size]
	Wo [][][]float32 // [layer][n_heads * head_size][dim]
	// weights for ffn
	W1 [][][]float32 // [layer][hidden_dim][dim]
	W2 [][][]float32 // [layer][dim][hidden_dim]
	W3 [][][]float32 // [layer][hidden_dim][dim]
	// final rmsnorm
	RmsFinalWeight []float32 // [dim]
	// (optional) classifier weights for the logits, on the last layer
	WCls [][]float32 // [vocab][dim]
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

func map1d[T any](ptr *[]T, a int) []T {
	flat := (*ptr)[:a:a]
	*ptr = (*ptr)[len(flat):]
	return flat
}

func map2d[T any](ptr *[]T, a, b int) [][]T {
	flat := map1d(ptr, a*b)
	out := make([][]T, a)
	for i := range a {
		out[i] = map1d(&flat, b)
	}
	return out
}

func map3d[T any](ptr *[]T, a, b, c int) [][][]T {
	flat := map1d(ptr, a*b*c)
	out := make([][][]T, a)
	for i := range a {
		out[i] = map2d(&flat, b, c)
	}
	return out
}

func make2d[T any](a, b int) [][]T {
	flat := make([]T, a*b)
	return map2d(&flat, a, b)
}

func make3d[T any](a, b, c int) [][][]T {
	flat := make([]T, a*b*c)
	return map3d(&flat, a, b, c)
}

func (m *Model) mapWeights(ptr []float32, sharedWeights bool) {
	p := &m.Config
	w := &m.Weights
	headSize := p.Dim / p.Heads

	w.TokenEmbeddings = map2d(&ptr, p.VocabSize, p.Dim)
	w.RmsAttWeight = map2d(&ptr, p.Layers, p.Dim)

	w.Wq = map3d(&ptr, p.Layers, p.Dim, p.Heads*headSize)
	w.Wk = map3d(&ptr, p.Layers, p.Dim, p.KVHeads*headSize)
	w.Wv = map3d(&ptr, p.Layers, p.Dim, p.KVHeads*headSize)
	w.Wo = map3d(&ptr, p.Layers, p.Heads*headSize, p.Dim)

	w.RmsFfnWeight = map2d(&ptr, p.Layers, p.Dim)

	w.W1 = map3d(&ptr, p.Layers, p.HiddenDim, p.Dim)
	w.W2 = map3d(&ptr, p.Layers, p.Dim, p.HiddenDim)
	w.W3 = map3d(&ptr, p.Layers, p.HiddenDim, p.Dim)

	w.RmsFinalWeight = map1d(&ptr, p.Dim)

	ptr = ptr[p.SeqLen*headSize/2:] // skip what used to be freq_cis_real (for RoPE)
	ptr = ptr[p.SeqLen*headSize/2:] // skip what used to be freq_cis_imag (for RoPE)
	if sharedWeights {
		w.WCls = w.TokenEmbeddings
	} else {
		w.WCls = map2d(&ptr, p.VocabSize, p.Dim)
	}
}
