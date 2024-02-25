package sdcpp

/*
#include "ggml.h"
*/
import "C"
import (
	"image"
	"unsafe"
)

var _ = [1]struct{}{}[unsafe.Sizeof(Tensor{})-unsafe.Sizeof(C.struct_ggml_tensor{})]

// Tensor is a n-dimensional tensor.
type Tensor struct {
	typ     C.enum_ggml_type
	backend C.enum_ggml_backend_type

	buffer *C.struct_ggml_backend_buffer

	ne [ggml_MAX_DIMS]int64 // number of elements
	// nb is a stride in bytes:
	// nb[0] = ggml_type_size(type)
	// nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
	// nb[i] = nb[i-1] * ne[i-1]
	nb [ggml_MAX_DIMS]uintptr

	// compute data
	op C.enum_ggml_op

	// op params - allocated as int32_t for alignment
	op_params [ggml_MAX_OP_PARAMS / 4]int32

	is_param C.bool

	grad *Tensor
	src  [ggml_MAX_SRC]*Tensor

	perf_runs    C.int
	perf_cycles  int64
	perf_time_us int64

	view_src  *Tensor
	view_offs uintptr

	data unsafe.Pointer

	name [ggml_MAX_NAME]byte

	extra unsafe.Pointer // extra things e.g. for ggml-cuda.cu

	padding [8]byte
}

func (t *Tensor) C() *C.struct_ggml_tensor {
	return (*C.struct_ggml_tensor)(unsafe.Pointer(t))
}

func (t *Tensor) Type() GGMLType {
	return GGMLType(t.typ)
}

type GGMLType int

func (t GGMLType) C() C.enum_ggml_type {
	return C.enum_ggml_type(t)
}

func (t GGMLType) Size() int64 {
	return int64(C.ggml_type_size(t.C()))
}

func (t GGMLType) BlockSize() int64 {
	return int64(C.ggml_blck_size(t.C()))
}

const (
	GGMLTypeF16     = GGMLType(C.GGML_TYPE_F16)
	GGMLTypeF32     = GGMLType(C.GGML_TYPE_F32)
	GGMLTypeDefault = GGMLType(C.GGML_TYPE_COUNT)
)

func NewTensor1D(ctx *GGMLContext, typ GGMLType, ne0 int) *Tensor {
	return (*Tensor)(unsafe.Pointer(C.ggml_new_tensor_1d(ctx.C(), typ.C(), C.int64_t(ne0))))
}

func NewTensor2D(ctx *GGMLContext, typ GGMLType, ne0, ne1 int) *Tensor {
	return (*Tensor)(unsafe.Pointer(C.ggml_new_tensor_2d(ctx.C(), typ.C(), C.int64_t(ne0), C.int64_t(ne1))))
}

func NewTensor3D(ctx *GGMLContext, typ GGMLType, ne0, ne1, ne2 int) *Tensor {
	return (*Tensor)(unsafe.Pointer(C.ggml_new_tensor_3d(ctx.C(), typ.C(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2))))
}

func NewTensor4D(ctx *GGMLContext, typ GGMLType, ne0, ne1, ne2, ne3 int) *Tensor {
	return (*Tensor)(unsafe.Pointer(C.ggml_new_tensor_4d(ctx.C(), typ.C(), C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.int64_t(ne3))))
}

func (t *Tensor) Get1DF32(l int) float32 {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	return *(*float32)(unsafe.Add(t.data, uintptr(l)*t.nb[0]))
}

func (t *Tensor) Set1DF32(val float32, l int) {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	*(*float32)(unsafe.Add(t.data, uintptr(l)*t.nb[0])) = val
}

func (t *Tensor) Get2DF32(l, k int) float32 {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	return *(*float32)(unsafe.Add(t.data, uintptr(k)*t.nb[1]+
		uintptr(l)*t.nb[0]))
}

func (t *Tensor) Set2DF32(val float32, l, k int) {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	*(*float32)(unsafe.Add(t.data, uintptr(k)*t.nb[1]+
		uintptr(l)*t.nb[0])) = val
}

func (t *Tensor) Get3DF32(l, k, j int) float32 {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	return *(*float32)(unsafe.Add(t.data, uintptr(j)*t.nb[2]+
		uintptr(k)*t.nb[1]+
		uintptr(l)*t.nb[0]))
}

func (t *Tensor) Set3DF32(val float32, l, k, j int) {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	*(*float32)(unsafe.Add(t.data, uintptr(j)*t.nb[2]+
		uintptr(k)*t.nb[1]+
		uintptr(l)*t.nb[0])) = val
}

func (t *Tensor) Get4DF32(l, k, j, i int) float32 {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	return *(*float32)(unsafe.Add(t.data, uintptr(i)*t.nb[3]+
		uintptr(j)*t.nb[2]+
		uintptr(k)*t.nb[1]+
		uintptr(l)*t.nb[0]))
}

func (t *Tensor) Set4DF32(val float32, l, k, j, i int) {
	if t.nb[0] != unsafe.Sizeof(float32(0)) {
		panic("invalid element size")
	}
	*(*float32)(unsafe.Add(t.data, uintptr(i)*t.nb[3]+
		uintptr(j)*t.nb[2]+
		uintptr(k)*t.nb[1]+
		uintptr(l)*t.nb[0])) = val
}

func SD_tensor_to_image(input *Tensor) *image.NRGBA {
	width := int(input.ne[0])
	height := int(input.ne[1])
	channels := int(input.ne[2])
	if channels != 3 || input.Type() != GGMLTypeF32 {
		panic("unexpected tensor type")
	}
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ind := y*img.Stride + x*4
			for c := 0; c < 3; c++ {
				val := input.Get3DF32(x, y, c)
				img.Pix[ind+c] = byte(val * 255)
			}
			img.Pix[ind+3] = 255 // alpha
		}
	}
	return img
}

func SD_image_to_tensor(out *Tensor, img *image.NRGBA) {
	width := int(out.ne[0])
	height := int(out.ne[1])
	channels := int(out.ne[2])
	if channels != 3 || out.Type() != GGMLTypeF32 {
		panic("unexpected tensor type")
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ind := y*img.Stride + x*4
			for c := 0; c < 3; c++ {
				val := img.Pix[ind+c]
				out.Set3DF32(float32(val)/255, x, y, c)
			}
		}
	}
}
