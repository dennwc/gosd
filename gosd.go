package gosd

import (
	"encoding"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/dennwc/gosd/sdcpp"
)

func init() {
	for name, m := range samplingMethodByName {
		samplingMethodNamesShort[m] = name
	}
	for m, name := range samplingMethodNames {
		samplingMethodByName[strings.ToLower(name)] = m
	}
}

const C = 0

func SystemInfo() string {
	return sdcpp.SD_get_system_info()
}

type Context struct {
	p   *sdcpp.Ctx
	log *slog.Logger
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
	Log             *slog.Logger
}

func New(p Params) (*Context, error) {
	if p.Log == nil {
		p.Log = slog.Default()
	}
	if p.Threads <= 0 {
		p.Threads = runtime.NumCPU()
	}
	if _, err := os.Stat(p.ModelPath); err != nil {
		return nil, err
	}
	for _, path := range []string{
		p.VAEPath, p.TAESDPath, p.CtrlNetPath, p.LoraPath, p.EmbeddingDir,
	} {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err != nil {
			return nil, err
		}
	}
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
		p.Threads,
		sdcpp.SD_type_t(p.Type),
		sdcpp.RNG_type_t(p.RNG),
		sdcpp.Schedule_t(p.Schedule),
		p.CtrlNetKeepCPU,
	)
	if c == nil {
		return nil, errors.New("cannot create context")
	}
	return &Context{p: c, log: p.Log}, nil
}

func (sd *Context) Close() {
	if sd.p != nil {
		sdcpp.Free_sd_ctx(sd.p)
		sd.p = nil
	}
}

var (
	_ encoding.TextMarshaler   = SamplingMethod(0)
	_ encoding.TextUnmarshaler = (*SamplingMethod)(nil)
)

type SamplingMethod int

func (m SamplingMethod) ID() string {
	s, ok := samplingMethodNamesShort[m]
	if !ok {
		return strconv.Itoa(int(m))
	}
	return s
}

func (m SamplingMethod) String() string {
	s, ok := samplingMethodNames[m]
	if !ok {
		return fmt.Sprintf("SamplingMethod(%d)", int(m))
	}
	return s
}

func (m SamplingMethod) MarshalText() ([]byte, error) {
	return []byte(m.ID()), nil
}

func (m *SamplingMethod) Parse(text string) error {
	v, ok := samplingMethodByName[text]
	if ok {
		*m = v
		return nil
	}
	val, err := strconv.Atoi(text)
	if err != nil {
		return fmt.Errorf("unsupported sampling method: %q", text)
	}
	*m = SamplingMethod(val)
	return nil
}

func (m *SamplingMethod) UnmarshalText(text []byte) error {
	return m.Parse(string(text))
}

func ParseSamplingMethod(s string) (SamplingMethod, error) {
	var m SamplingMethod
	if err := m.Parse(s); err != nil {
		return EULER_A, err
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

var samplingMethodByName = map[string]SamplingMethod{
	"euler_a":   EULER_A,
	"euler":     EULER,
	"heun":      HEUN,
	"dpm2":      DPM2,
	"dpm++2s_a": DPMPP2S_A,
	"dpm++2m":   DPMPP2M,
	"dpm++2mv2": DPMPP2Mv2,
	"lcm":       LCM,
}

var samplingMethodNamesShort = make(map[SamplingMethod]string)

var samplingMethodNames = map[SamplingMethod]string{
	EULER_A:   "Euler A",
	EULER:     "Euler",
	HEUN:      "Heun",
	DPM2:      "DPM2",
	DPMPP2S_A: "DPM++ (2s)",
	DPMPP2M:   "DPM++ (2M)",
	DPMPP2Mv2: "modified DPM++ (2M)",
	LCM:       "LCM",
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

func (sd *Context) TextToImage(p TextToImageParams) *image.NRGBA {
	p.BatchCount = 1
	imgs, err := sd.TextToImages(p)
	if err != nil {
		sd.log.Error("txt2img failed", "err", err)
	}
	if len(imgs) == 0 {
		return nil
	}
	return imgs[0]
}

func (sd *Context) TextToImages(p TextToImageParams) ([]*image.NRGBA, error) {
	if p.BatchCount <= 0 {
		p.BatchCount = 1
	}
	imgs, err := sd.txt2img(p, nil, 0.9)
	if errors.Is(err, errUnsupportedTODO) {
		imgs, err = sd.txt2imgCgo(p)
	}
	return imgs, err
}

func (sd *Context) txt2imgCgo(p TextToImageParams) ([]*image.NRGBA, error) {
	pimgs := sdcpp.Txt2Img(
		sd.p,
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
	if len(pimgs) == 0 {
		return nil, errors.New("generation failed")
	}
	defer func() {
		for _, pimg := range pimgs {
			if data := pimg.Data(); data != nil {
				sdcpp.FreeSlice(data)
			}
		}
		sdcpp.FreeSlice(pimgs)
	}()

	out := make([]*image.NRGBA, len(pimgs))
	var last error
	for i, pimg := range pimgs {
		w, h := pimg.Size()
		ch := pimg.Channels()
		data := pimg.Data()
		if len(data) == 0 {
			last = errors.New("no image data")
			continue
		}
		if ch != 3 {
			last = errors.New("unexpected number of channels")
			continue
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
		out[i] = img
	}
	return out, last
}

var errUnsupportedTODO = errors.New("TODO: unsupported configuration")

func (sd *Context) txt2img(p TextToImageParams, controlCond *image.NRGBA, controlStrength float32) ([]*image.NRGBA, error) {
	sd.log.Debug("txt2img", "w", p.Width, "h", p.Height)
	if sd == nil {
		return nil, errors.New("nil SD context")
	}
	prompt, negative := p.Prompt, p.Negative
	// extract and remove lora
	loraF2M, prompt, err := extract_and_remove_lora(prompt)
	if err != nil {
		return nil, err
	}
	for k, v := range loraF2M {
		sd.log.Debug("lora", "file", k, "mul", v)
	}
	sd.log.Debug("prompt after lora extraction", "prompt", prompt)
	t0 := time.Now()
	if len(loraF2M) != 0 {
		//sd.sd.apply_loras(loraF2M)
		return nil, errUnsupportedTODO
	}
	t1 := time.Now()
	sd.log.Info("apply_loras completed", "t", t1.Sub(t0).Round(time.Millisecond))

	var params sdcpp.GGMLInitParams
	params.MemSize = 10 * 1024 * 1024 // 10 MB
	params.MemSize += uintptr(p.Width*p.Height) * 3 * unsafe.Sizeof(float32(0))
	params.MemSize *= uintptr(p.BatchCount)
	params.NoAlloc = false

	gctx, err := sdcpp.GGMLInit(params)
	if err != nil {
		return nil, err
	}
	defer gctx.Free()

	if p.Seed < 0 {
		p.Seed = rand.Int63()
	}

	t0 = time.Now()
	c, cvec := sdcpp.GetLearnedCondition(sd.p, gctx, prompt, p.ClipSkip, p.Width, p.Height)
	var uc, ucvec *sdcpp.Tensor
	if p.CfgScale != 1.0 {
		uc, ucvec = sdcpp.GetLearnedConditionNeg(sd.p, gctx, negative, p.ClipSkip, p.Width, p.Height)
	}
	t1 = time.Now()
	sd.log.Info("get_learned_condition completed", "t", t1.Sub(t0).Round(time.Millisecond))
	sdcpp.MaybeFreeCond(sd.p)
	var imageHint *sdcpp.Tensor
	if controlCond != nil {
		imageHint = sdcpp.NewTensor4D(gctx, sdcpp.GGMLTypeF32, p.Width, p.Height, 3, 1)
		sdcpp.SD_image_to_tensor(imageHint, controlCond)
	}
	var finalLatents []*sdcpp.Tensor // collect latents to decode
	C := 4
	W := p.Width / 8
	H := p.Height / 8
	sd.log.Info("sampling", "method", p.Sampling.ID(), "name", p.Sampling.String())
	for b := 0; b < p.BatchCount; b++ {
		sstart := time.Now()
		curSeed := p.Seed + int64(b)
		sd.log.Info("generating image", "i", b+1, "n", p.BatchCount, "seed", curSeed)
		xt := sdcpp.NewTensor4D(gctx, sdcpp.GGMLTypeF32, W, H, C, 1)
		x0 := sdcpp.Sample(sd.p, gctx, xt, nil, c, cvec, uc, ucvec, imageHint, curSeed, sdcpp.Sample_method_t(p.Sampling), p.Steps, float32(p.CfgScale), controlStrength)
		if p.BatchCount > 1 {
			sd.log.Info("sampling completed", "t", time.Since(sstart).Round(time.Millisecond))
		}
		finalLatents = append(finalLatents, x0)
	}
	sdcpp.MaybeFreeDiff(sd.p)

	t3 := time.Now()
	sd.log.Info("generating latent images completed", "n", len(finalLatents), "t", t3.Sub(t1).Round(time.Millisecond))
	sd.log.Info("decoding latents", "n", len(finalLatents))

	var decodedImages []*sdcpp.Tensor // collect decoded images
	for i := range finalLatents {
		t1 = time.Now()
		img := sdcpp.DecodeFirstStage(sd.p, gctx, finalLatents[i])
		if img != nil {
			decodedImages = append(decodedImages, img)
		}
		t2 := time.Now()
		sd.log.Info("latent decoded", "i", i+1, "t", t2.Sub(t1).Round(time.Millisecond))
	}

	t4 := time.Now()
	sd.log.Info("decode_first_stage completed", "t", t4.Sub(t3).Round(time.Millisecond))
	sdcpp.MaybeFreeFirst(sd.p)
	results := make([]*image.NRGBA, 0, len(decodedImages))
	for _, img := range decodedImages {
		results = append(results, sdcpp.SD_tensor_to_image(img))
	}
	sd.log.Info("txt2img completed", "t", t4.Sub(t0).Round(time.Millisecond))

	return results, nil
}

func extract_and_remove_lora(prompt string) (map[string]float64, string, error) {
	re := regexp.MustCompile(`<lora:([^:]+):([^>]+)>`)
	file2mult := make(map[string]float64)
	for _, sub := range re.FindAllStringSubmatch(prompt, -1) {
		filename := sub[1]
		mult, err := strconv.ParseFloat(sub[2], 64)
		prompt = strings.Replace(prompt, sub[0], "", 1)
		if err != nil {
			return nil, "", fmt.Errorf("error parsing Lora prompt: %w", err)
		}
		if mult == 0 {
			continue
		}
		if cur, ok := file2mult[filename]; ok {
			mult += cur
		}
		file2mult[filename] = mult
	}
	return file2mult, prompt, nil
}
