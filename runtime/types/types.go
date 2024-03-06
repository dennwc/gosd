package types

// ModelID is a unique ID of the model. Usually a hash.
type ModelID string

const (
	SamplerEuler = SamplerID("euler")
)

// SamplerID is an ID of the sampler method.
type SamplerID string

// ModelInfo is a base info about a model.
type ModelInfo struct {
	ID    ModelID
	Name  string
	Title string
	File  string
}

// SamplerInfo is a base info about a sampler.
type SamplerInfo struct {
	ID   SamplerID
	Name string
}

// TextToImageParams controls typical parameters for text-to-image generation.
type TextToImageParams struct {
	Model    ModelID   `json:"model_id,omitempty"`
	Prompt   string    `json:"positive_prompt,omitempty"`
	Negative string    `json:"negative_prompt,omitempty"`
	Seed     int64     `json:"seed,omitempty"`
	Sampler  SamplerID `json:"sampler_id,omitempty"`
	Steps    int       `json:"steps,omitempty"`
	CfgScale float64   `json:"cfg_scale,omitempty"`
	ClipSkip int       `json:"clip_skip,omitempty"`
	Width    int       `json:"width,omitempty"`
	Height   int       `json:"height,omitempty"`
}
