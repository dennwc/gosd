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

type Params struct {
	ModelPath string
}

func New(p Params) (*Context, error) {
	c := sdcpp.New_sd_ctx(
		p.ModelPath,
		"",
		"",
		"",
		"",
		"",
		true,
		false,
		true,
		runtime.NumCPU(),
		sdcpp.SD_TYPE_COUNT,
		sdcpp.CUDA_RNG,
		sdcpp.DEFAULT,
		false,
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
	Prompt   string
	Negative string
	CfgScale float64
	Width    int
	Height   int
	Sampling SamplingMethod
	Steps    int
	Seed     int64
}

func (c *Context) TextToImage(p TextToImageParams) *image.NRGBA {
	pimg := sdcpp.Txt2Img(
		c.p,
		p.Prompt,
		p.Negative,
		-1,
		float32(p.CfgScale),
		p.Width,
		p.Height,
		sdcpp.Sample_method_t(p.Sampling),
		p.Steps,
		p.Seed,
		1,
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
