package sdcpp

/*
#cgo CFLAGS: -O3 -march=native
#include "stable-diffusion.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

func SD_get_system_info() string {
	return C.GoString(C.sd_get_system_info())
}

type Ctx = C.sd_ctx_t

func Free_sd_ctx(p *Ctx) {
	C.free_sd_ctx(p)
}

type SD_type_t = C.enum_sd_type_t

const (
	SD_TYPE_F32     = C.SD_TYPE_F32
	SD_TYPE_F16     = C.SD_TYPE_F16
	SD_TYPE_Q4_0    = C.SD_TYPE_Q4_0
	SD_TYPE_Q4_1    = C.SD_TYPE_Q4_1
	SD_TYPE_Q5_0    = C.SD_TYPE_Q5_0
	SD_TYPE_Q5_1    = C.SD_TYPE_Q5_1
	SD_TYPE_Q8_0    = C.SD_TYPE_Q8_0
	SD_TYPE_Q8_1    = C.SD_TYPE_Q8_1
	SD_TYPE_Q2_K    = C.SD_TYPE_Q2_K
	SD_TYPE_Q3_K    = C.SD_TYPE_Q3_K
	SD_TYPE_Q4_K    = C.SD_TYPE_Q4_K
	SD_TYPE_Q5_K    = C.SD_TYPE_Q5_K
	SD_TYPE_Q6_K    = C.SD_TYPE_Q6_K
	SD_TYPE_Q8_K    = C.SD_TYPE_Q8_K
	SD_TYPE_IQ2_XXS = C.SD_TYPE_IQ2_XXS
	SD_TYPE_I8      = C.SD_TYPE_I8
	SD_TYPE_I16     = C.SD_TYPE_I16
	SD_TYPE_I32     = C.SD_TYPE_I32
	SD_TYPE_COUNT   = C.SD_TYPE_COUNT
)

type RNG_type_t = C.enum_rng_type_t

const (
	STD_DEFAULT_RNG = C.STD_DEFAULT_RNG
	CUDA_RNG        = C.CUDA_RNG
)

type Schedule_t = C.enum_schedule_t

const (
	DEFAULT     = C.DEFAULT
	DISCRETE    = C.DISCRETE
	KARRAS      = C.KARRAS
	N_SCHEDULES = C.N_SCHEDULES
)

func New_sd_ctx(
	modelPath, vaePath, taesdPath, ctrlNetPath, loraPath, embedDir string,
	vaeDecOnly, vaeTiling, freeParamsImmediately bool,
	threads int, typ SD_type_t, rnd RNG_type_t, sched Schedule_t,
	keepCtrlCPU bool,
) *Ctx {
	modelPathC, pfree := cstring(modelPath)
	defer pfree()
	vaePathC, vfree := cstring(vaePath)
	defer vfree()
	taesdPathC, tfree := cstring(taesdPath)
	defer tfree()
	ctrlNetPathC, cfree := cstring(ctrlNetPath)
	defer cfree()
	loraPathC, lfree := cstring(loraPath)
	defer lfree()
	embedDirC, efree := cstring(embedDir)
	defer efree()
	return C.new_sd_ctx(
		modelPathC, vaePathC, taesdPathC, ctrlNetPathC, loraPathC, embedDirC,
		C.bool(vaeDecOnly), C.bool(vaeTiling), C.bool(freeParamsImmediately),
		C.int(threads), typ, rnd, sched, C.bool(keepCtrlCPU),
	)
}

type Sample_method_t = C.enum_sample_method_t

const (
	EULER_A   = C.EULER_A
	EULER     = C.EULER
	HEUN      = C.HEUN
	DPM2      = C.DPM2
	DPMPP2S_A = C.DPMPP2S_A
	DPMPP2M   = C.DPMPP2M
	DPMPP2Mv2 = C.DPMPP2Mv2
	LCM       = C.LCM
)

type SD_image_t C.sd_image_t

func (img *SD_image_t) Size() (w, h int) {
	if img == nil {
		return 0, 0
	}
	return int(img.width), int(img.height)
}

func (img *SD_image_t) Channels() int {
	if img == nil {
		return 0
	}
	return int(img.channel)
}

func (img *SD_image_t) Data() []byte {
	if img == nil {
		return nil
	}
	w, h, ch := int(img.width), int(img.height), int(img.channel)
	ptr := (*byte)(img.data)
	if ptr == nil {
		return nil
	}
	return unsafe.Slice(ptr, w*h*ch)
}

func Txt2Img(
	ctx *Ctx, prompt, negative string, clipSkip int, cfgScale float32, w, h int,
	sample Sample_method_t, steps int, seed int64, batchCount int,
	controlCond *SD_image_t, controlStrength float32,
) *SD_image_t {
	promptC, pfree := cstring(prompt)
	defer pfree()
	negativeC, nfree := cstring(negative)
	defer nfree()
	return (*SD_image_t)(C.txt2img(
		ctx, promptC, negativeC, C.int(clipSkip), C.float(cfgScale), C.int(w), C.int(h),
		sample, C.int(steps), C.int64_t(seed), C.int(batchCount),
		(*C.sd_image_t)(controlCond), C.float(controlStrength),
	))
}

func cstring(s string) (*C.char, func()) {
	cs := C.CString(s)
	return cs, func() {
		Free(cs)
	}
}

func Free[T any](p *T) {
	C.free(unsafe.Pointer(p))
}

func FreeSlice[T any](p []T) {
	p = p[:1]
	C.free(unsafe.Pointer(&p[0]))
}
