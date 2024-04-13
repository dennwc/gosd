package protocol

import (
	"bytes"
	"context"
	"image"
	"image/png"
	"io"
	"net/http"

	"github.com/dennwc/gosd/runtime"
	"github.com/dennwc/gosd/runtime/types"
)

func NewRuntime(url string) runtime.Runtime {
	cli := NewStableDiffusionProtobufClient(url, http.DefaultClient)
	return &runtimeClient{cli: cli}
}

type runtimeClient struct {
	cli StableDiffusion
}

func (c *runtimeClient) Close() {}

func (c *runtimeClient) ListModels(ctx context.Context) ([]types.ModelInfo, error) {
	list, err := c.cli.ListModels(ctx, &ListModelsReq{})
	if err != nil {
		return nil, err
	}
	out := make([]types.ModelInfo, 0, len(list.Models))
	for _, m := range list.Models {
		out = append(out, types.ModelInfo{
			ID:    types.ModelID(m.Id),
			Name:  m.Name,
			Title: m.Title,
			File:  m.File,
		})
	}
	return out, nil
}

func (c *runtimeClient) ListSamplers(ctx context.Context) ([]types.SamplerInfo, error) {
	list, err := c.cli.ListSamplers(ctx, &ListSamplersReq{})
	if err != nil {
		return nil, err
	}
	out := make([]types.SamplerInfo, 0, len(list.Samplers))
	for _, m := range list.Samplers {
		out = append(out, types.SamplerInfo{
			ID:   types.SamplerID(m.Id),
			Name: m.Name,
		})
	}
	return out, nil
}

func (c *runtimeClient) textToImage(ctx context.Context, p types.TextToImageParams) (*TextToImageResp, error) {
	return c.cli.TextToImage(ctx, &TextToImageReq{
		ModelId:        string(p.Model),
		PositivePrompt: p.Prompt,
		NegativePrompt: p.Negative,
		CfgScale:       p.CfgScale,
		RngSeed:        p.Seed,
		SamplerId:      string(p.Sampler),
		SamplerSteps:   uint32(p.Steps),
		ImageWidth:     uint32(p.Width),
		ImageHeight:    uint32(p.Height),
		ClipSkip:       uint32(p.ClipSkip),
	})
}
func (c *runtimeClient) TextToImage(ctx context.Context, p types.TextToImageParams) (image.Image, error) {
	resp, err := c.textToImage(ctx, p)
	if err != nil {
		return nil, err
	}
	return resp.GetImage().Decode()
}

func (c *runtimeClient) TextToImagePNG(ctx context.Context, p types.TextToImageParams) (io.ReadCloser, error) {
	resp, err := c.textToImage(ctx, p)
	if err != nil {
		return nil, err
	}
	img := resp.GetImage()
	switch img := img.GetImage().(type) {
	case *Image_Png:
		return io.NopCloser(bytes.NewReader(img.Png)), nil
	}
	dec, err := img.Decode()
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	err = png.Encode(&buf, dec)
	if err != nil {
		return nil, err
	}
	return io.NopCloser(&buf), nil
}
