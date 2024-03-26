package llama2

import "slices"

type probIndex struct {
	prob  float32
	index int
}

// Sampler takes logits and returns a sampled token.
// Sampling can be done in a few ways: greedy argmax, sampling, top-p sampling.
type Sampler struct {
	probIndex   []probIndex // buffer used in top-p sampling
	temperature float32
	topp        float32
	rng         random
}

func sampleArgmax(probabilities []float32) int {
	// return the index that has the highest probability
	maxI := 0
	maxP := probabilities[0]
	for i := 1; i < len(probabilities); i++ {
		if probabilities[i] > maxP {
			maxI = i
			maxP = probabilities[i]
		}
	}
	return maxI
}

func sampleMult(probabilities []float32, coin float32) int {
	// sample index from probabilities (they must sum to 1!)
	// coin is a random number in [0, 1), usually from random_f32()
	cdf := float32(0.0)
	for i := 0; i < len(probabilities); i++ {
		cdf += probabilities[i]
		if coin < cdf {
			return i
		}
	}
	return len(probabilities) - 1 // in case of rounding errors
}

func compare(a, b probIndex) int {
	if a.prob > b.prob {
		return -1
	}
	if a.prob < b.prob {
		return 1
	}
	return 0
}

func sampleTopP(probabilities []float32, topp float32, probindex []probIndex, coin float32) int {
	// top-p sampling (or "nucleus sampling") samples from the smallest set of
	// tokens that exceed probability topp. This way we never sample tokens that
	// have very low probabilities and are less likely to go "off the rails".
	// coin is a random number in [0, 1), usually from random_f32()

	n0 := 0
	// quicksort indices in descending order of probabilities
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	// so for efficiency we crop these out as candidates before sorting
	cutoff := (1.0 - topp) / float32(len(probabilities)-1)
	for i := 0; i < len(probabilities); i++ {
		if probabilities[i] >= cutoff {
			probindex[n0].index = i
			probindex[n0].prob = probabilities[i]
			n0++
		}
	}
	slices.SortFunc(probindex[:n0], compare)

	// truncate the list where cumulative probability exceeds topp
	cumulativeProb := float32(0.0)
	lastIdx := n0 - 1 // in case of rounding errors consider all elements
	for i := 0; i < n0; i++ {
		cumulativeProb += probindex[i].prob
		if cumulativeProb > topp {
			lastIdx = i
			break // we've exceeded topp by including last_idx
		}
	}

	// sample from the truncated list
	r := coin * cumulativeProb
	cdf := float32(0.0)
	for i := 0; i <= lastIdx; i++ {
		cdf += probindex[i].prob
		if r < cdf {
			return probindex[i].index
		}
	}
	return probindex[lastIdx].index // in case of rounding errors
}

func NewSampler(vocabSize int, temperature float32, topp float32, seed int64) *Sampler {
	return &Sampler{
		temperature: temperature,
		topp:        topp,
		rng:         random{uint64(seed)},
		// buffer only used with nucleus sampling; may not need but it's ~small
		probIndex: make([]probIndex, vocabSize),
	}
}

func (s *Sampler) Sample(logits []float32) Token {
	// sample the token given the logits and some hyperparameters
	var next Token
	if s.temperature == 0.0 {
		// greedy argmax sampling: take the token with the highest probability
		next = Token(sampleArgmax(logits))
	} else {
		// apply the temperature to the logits
		for q := 0; q < len(logits); q++ {
			logits[q] /= s.temperature
		}
		// apply softmax to the logits to get the probabilities for next token
		softmax(logits)
		// flip a (float) coin (this is our source of entropy for sampling)
		coin := s.rng.F32()
		// we sample from this distribution to get the next token
		if s.topp <= 0 || s.topp >= 1 {
			// simply sample from the predicted probability distribution
			next = Token(sampleMult(logits, coin))
		} else {
			// top-p (nucleus) sampling, clamping the least likely tokens to zero
			next = Token(sampleTopP(logits, s.topp, s.probIndex, coin))
		}
	}
	return next
}
