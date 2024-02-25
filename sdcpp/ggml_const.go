package sdcpp

/*
#include "ggml.h"
*/
import "C"

const (
	ggml_MAX_DIMS      = 4
	ggml_MAX_OP_PARAMS = 64
	ggml_MAX_SRC       = 10
	ggml_MAX_NAME      = C.GGML_MAX_NAME
)
