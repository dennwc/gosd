package llama2

import (
	"math"
	"slices"
)

func rmsnorm(o, x, weight []float32) {
	// calculate sum of squares
	ss := float32(0.0)
	for j := range x {
		ss += x[j] * x[j]
	}
	ss /= float32(len(x))
	ss += 1e-5
	ss = float32(1.0 / math.Sqrt(float64(ss)))
	// normalize and scale
	for j := range o {
		o[j] = weight[j] * (ss * x[j])
	}
}

func softmax(x []float32) {
	// find max value (for numerical stability)
	maxVal := slices.Max(x)
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	// exp and sum
	sum := float32(0.0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	// normalize
	for i := range x {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32, n, d int) {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function

	// pragma omp parallel for private(i)
	for i := 0; i < d; i++ {
		val := float32(0.0)
		for j := 0; j < n; j++ {
			val += w[i*n+j] * x[j]
		}
		xout[i] = val
	}
}
