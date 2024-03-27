package llama2

import (
	"math"
	"slices"
	"unsafe"

	"github.com/chewxy/math32"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
)

const errLength = "length mismatch"

var blas32i = gonum.Implementation{}

func memset0(a []float32) {
	for i := range a {
		a[i] = 0
	}
}

func swiglu(hb, hb2 []float32) {
	for i, val := range hb {
		// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
		val *= 1.0 / (1.0 + math32.Exp(-val))
		// elementwise multiply with w3(x)
		val *= hb2[i]
		hb[i] = val
	}
}

func rmsnorm(o, x, weight []float32) {
	// calculate sum of squares
	ss := ddot(x, x)
	ss /= float64(len(x))
	ss += 1e-5
	ss = 1.0 / math.Sqrt(ss)
	ss1 := float32(ss)
	// normalize and scale
	for j := range o {
		o[j] = weight[j] * (ss1 * x[j])
	}
}

func softmax(x []float32) {
	// find max value (for numerical stability)
	maxVal := slices.Max(x)
	// exp and sum
	sum := float32(0.0)
	for i := range x {
		x[i] = math32.Exp(x[i] - maxVal)
		sum += x[i]
	}
	// normalize
	for i := range x {
		x[i] /= sum
	}
}

func matmul(xout, x []float32, w [][]float32) {
	if len(xout) != len(w) {
		panic(errLength)
	}
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	if false {
		m, k := len(w), len(x)
		const n = 1
		wf := unsafe.Slice(&w[0][0], m*k)
		blas32i.Sgemm(blas.NoTrans, blas.NoTrans, m, n, k, 1, wf, k, x, n, 0, xout, n)
	} else {
		for i := range xout {
			xout[i] = dot(w[i], x)
		}
	}
}

func dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic(errLength)
	}
	return blas32i.Sdot(len(a), a, 1, b, 1)
}

func ddot(a, b []float32) float64 {
	if len(a) != len(b) {
		panic(errLength)
	}
	return blas32i.Dsdot(len(a), a, 1, b, 1)
}

func div(a []float32, b float32) {
	blas32i.Sscal(len(a), 1/b, a, 1)
}
