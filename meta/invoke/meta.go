package invoke

import (
	"encoding/json"

	"github.com/dennwc/gosd/meta"
	"github.com/dennwc/gosd/meta/pnginfo"
	"github.com/dennwc/gosd/runtime/types"
)

const PNGTextName = "invokeai_metadata"

var _ meta.RawMetaJSON = (*Metadata)(nil)

func init() {
	pnginfo.RegisterTextHandler(PNGTextName, func(s string) (meta.ImageMetadata, error) {
		m, err := Parse(s)
		if err != nil || m == nil {
			return nil, err
		}
		return m, nil
	})
}

type Metadata struct {
	GenerationMode       string  `json:"generation_mode"`
	PositivePrompt       string  `json:"positive_prompt"`
	NegativePrompt       string  `json:"negative_prompt"`
	Width                int     `json:"width"`
	Height               int     `json:"height"`
	Seed                 int64   `json:"seed"`
	RandDevice           string  `json:"rand_device"`
	CfgScale             float64 `json:"cfg_scale"`
	CfgRescaleMultiplier float64 `json:"cfg_rescale_multiplier"`
	Steps                int     `json:"steps"`
	Scheduler            string  `json:"scheduler"`
	ClipSkip             int     `json:"clip_skip"`
	Model                struct {
		Name string `json:"model_name"`
		Base string `json:"base_model"`
		Type string `json:"model_type"`
	} `json:"model"`
	RawJSON json.RawMessage `json:"-"`
}

func (m *Metadata) RawMetadata() json.RawMessage {
	return m.RawJSON
}

func (m *Metadata) TextToImage() *types.TextToImageParams {
	return &types.TextToImageParams{
		Prompt:   m.PositivePrompt,
		Negative: m.NegativePrompt,
		Model:    types.ModelID(m.Model.Name),
		Seed:     m.Seed,
		Sampler:  types.SamplerID(m.Scheduler),
		Steps:    m.Steps,
		CfgScale: m.CfgScale,
		ClipSkip: m.ClipSkip,
		Width:    m.Width,
		Height:   m.Height,
	}
}

func Parse(s string) (*Metadata, error) {
	if len(s) == 0 {
		return nil, nil
	}
	m := &Metadata{RawJSON: json.RawMessage(s)}
	err := json.Unmarshal(m.RawJSON, m)
	return m, err
}
