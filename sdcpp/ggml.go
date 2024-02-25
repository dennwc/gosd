package sdcpp

/*
#include "ggml.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

type GGMLInitParams struct {
	MemSize   uintptr
	MemBuffer unsafe.Pointer
	NoAlloc   bool
}

type GGMLContext C.struct_ggml_context

func (ctx *GGMLContext) C() *C.struct_ggml_context {
	return (*C.struct_ggml_context)(ctx)
}

func (ctx *GGMLContext) Free() {
	C.ggml_free(ctx.C())
}

func GGMLInit(params GGMLInitParams) (*GGMLContext, error) {
	ctx := (*GGMLContext)(C.ggml_init(C.struct_ggml_init_params{
		mem_size:   C.size_t(params.MemSize),
		mem_buffer: params.MemBuffer,
		no_alloc:   C.bool(params.NoAlloc),
	}))
	if ctx == nil {
		return nil, errors.New("ggml_init() failed")
	}
	return ctx, nil
}
