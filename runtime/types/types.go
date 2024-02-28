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
	Model    ModelID
	Prompt   string
	Negative string
	Seed     int64
	Sampler  SamplerID
	Steps    int
	CfgScale float64
	ClipSkip int
	Width    int
	Height   int
}
