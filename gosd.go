package gosd

import (
	"errors"
	"fmt"
	"image"
	"runtime"
	"strings"

	"github.com/dennwc/gosd/sdcpp"
)

const C = 0

func SystemInfo() string {
	return sdcpp.SD_get_system_info()
}

type Context struct {
	p *sdcpp.Ctx
}

type Type int

const (
	TypeF32     = Type(sdcpp.SD_TYPE_F32)
	TypeF16     = Type(sdcpp.SD_TYPE_F16)
	TypeQ4_0    = Type(sdcpp.SD_TYPE_Q4_0)
	TypeQ4_1    = Type(sdcpp.SD_TYPE_Q4_1)
	TypeQ5_0    = Type(sdcpp.SD_TYPE_Q5_0)
	TypeQ5_1    = Type(sdcpp.SD_TYPE_Q5_1)
	TypeQ8_0    = Type(sdcpp.SD_TYPE_Q8_0)
	TypeQ8_1    = Type(sdcpp.SD_TYPE_Q8_1)
	TypeQ2_K    = Type(sdcpp.SD_TYPE_Q2_K)
	TypeQ3_K    = Type(sdcpp.SD_TYPE_Q3_K)
	TypeQ4_K    = Type(sdcpp.SD_TYPE_Q4_K)
	TypeQ5_K    = Type(sdcpp.SD_TYPE_Q5_K)
	TypeQ6_K    = Type(sdcpp.SD_TYPE_Q6_K)
	TypeQ8_K    = Type(sdcpp.SD_TYPE_Q8_K)
	TypeIQ2_XXS = Type(sdcpp.SD_TYPE_IQ2_XXS)
	TypeI8      = Type(sdcpp.SD_TYPE_I8)
	TypeI16     = Type(sdcpp.SD_TYPE_I16)
	TypeI32     = Type(sdcpp.SD_TYPE_I32)
	TypeDefault = Type(sdcpp.SD_TYPE_COUNT)
)

type RNGType int

const (
	StdDefaultRNG = RNGType(sdcpp.STD_DEFAULT_RNG)
	CUDA_RNG      = RNGType(sdcpp.CUDA_RNG)
)

type Schedule int

const (
	ScheduleDefault = Schedule(sdcpp.DEFAULT)
	Discrete        = Schedule(sdcpp.DISCRETE)
	Karras          = Schedule(sdcpp.KARRAS)
	NSchedules      = Schedule(sdcpp.N_SCHEDULES)
)

type Params struct {
	ModelPath       string
	VAEPath         string
	TAESDPath       string
	CtrlNetPath     string
	LoraPath        string
	EmbeddingDir    string
	VAETiling       bool
	FreeImmediately bool
	Threads         int
	Type            Type
	RNG             RNGType
	Schedule        Schedule
	CtrlNetKeepCPU  bool
}

func New(p Params) (*Context, error) {
	c := sdcpp.New_sd_ctx(
		p.ModelPath,
		p.VAEPath,
		p.TAESDPath,
		p.CtrlNetPath,
		p.LoraPath,
		p.EmbeddingDir,
		true,
		p.VAETiling,
		p.FreeImmediately,
		runtime.NumCPU(),
		sdcpp.SD_type_t(p.Type),
		sdcpp.RNG_type_t(p.RNG),
		sdcpp.Schedule_t(p.Schedule),
		p.CtrlNetKeepCPU,
	)
	if c == nil {
		return nil, errors.New("cannot create context")
	}
	return &Context{p: c}, nil
}

func (c *Context) Close() {
	if c.p != nil {
		sdcpp.Free_sd_ctx(c.p)
		c.p = nil
	}
}

type SamplingMethod int

func ParseSamplingMethod(s string) (SamplingMethod, error) {
	m, ok := samplingMethodNames[strings.ToLower(s)]
	if !ok {
		return EULER_A, fmt.Errorf("unsupported sampling method: %q", s)
	}
	return m, nil
}

const (
	EULER_A   = SamplingMethod(sdcpp.EULER_A)
	EULER     = SamplingMethod(sdcpp.EULER)
	HEUN      = SamplingMethod(sdcpp.HEUN)
	DPM2      = SamplingMethod(sdcpp.DPM2)
	DPMPP2S_A = SamplingMethod(sdcpp.DPMPP2S_A)
	DPMPP2M   = SamplingMethod(sdcpp.DPMPP2M)
	DPMPP2Mv2 = SamplingMethod(sdcpp.DPMPP2Mv2)
	LCM       = SamplingMethod(sdcpp.LCM)
)

var samplingMethodNames = map[string]SamplingMethod{
	"euler_a":   EULER_A,
	"euler":     EULER,
	"heun":      HEUN,
	"dpm2":      DPM2,
	"dpm++2s_a": DPMPP2S_A,
	"dpm++2m":   DPMPP2M,
	"dpm++2mv2": DPMPP2Mv2,
	"lcm":       LCM,
}

type TextToImageParams struct {
	Prompt     string
	Negative   string
	ClipSkip   int
	CfgScale   float64
	Width      int
	Height     int
	Sampling   SamplingMethod
	Steps      int
	BatchCount int
	Seed       int64
}

func (c *Context) TextToImage(p TextToImageParams) *image.NRGBA {
	pimg := sdcpp.Txt2Img(
		c.p,
		p.Prompt,
		p.Negative,
		p.ClipSkip,
		float32(p.CfgScale),
		p.Width,
		p.Height,
		sdcpp.Sample_method_t(p.Sampling),
		p.Steps,
		p.Seed,
		p.BatchCount,
		nil,
		0.9,
	)
	if pimg == nil {
		return nil
	}
	defer sdcpp.Free(pimg)

	w, h := pimg.Size()
	ch := pimg.Channels()
	data := pimg.Data()
	if len(data) == 0 {
		return nil
	}
	defer sdcpp.FreeSlice(data)

	if ch != 3 {
		panic("unexpected number of channels")
	}
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src := data[3*(w*y+x) : 3*(w*y+x+1)]
			dst := img.Pix[y*img.Stride+x*4 : y*img.Stride+(x+1)*4]
			copy(dst, src)
			dst[3] = 0xff // alpha
		}
	}
	return img
}
