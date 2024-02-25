package sdcpp

/*
#include "ggml.h"
#include "ggml-backend.h"
*/
import "C"

type cgoBackend struct {
	p C.ggml_backend_t
}

func (b cgoBackend) IsCPU() bool {
	return bool(C.ggml_backend_is_cpu(b.p))
}

func GGMLNewCPU() GGMLBackend {
	return cgoBackend{p: C.ggml_backend_cpu_init()}
}
