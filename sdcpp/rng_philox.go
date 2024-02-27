package sdcpp

/*
#include <stdint.h>
void go_philox_rng(float* dst, int n, uint64_t seed);
void go_philox_rngU(uint32_t* dst, int n, uint64_t seed);
*/
import "C"

import (
	"unsafe"
)

func philoxCgo(dst []float32, seed int64) {
	C.go_philox_rng((*C.float)(unsafe.Pointer(&dst[0])), C.int(len(dst)), C.uint64_t(seed))
}

func philoxUCgo(dst []uint32, seed int64) {
	C.go_philox_rngU((*C.uint32_t)(unsafe.Pointer(&dst[0])), C.int(len(dst)/2), C.uint64_t(seed))
}
