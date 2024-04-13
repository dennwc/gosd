package protocol

import (
	"context"
	"io"

	"github.com/dennwc/gosd/runtime"
	"github.com/dennwc/gosd/runtime/types"
)

func NewRuntimeServer(r runtime.Runtime) TwirpServer {
	return NewStableDiffusionServer(&runtimeServer{r: r})
}

type runtimeServer struct {
	r runtime.Runtime
}

func (s *runtimeServer) ListSamplers(ctx context.Context, req *ListSamplersReq) (*ListSamplersResp, error) {
	list, err := s.r.ListSamplers(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]*SamplerInfo, 0, len(list))
	for _, v := range list {
		out = append(out, &SamplerInfo{
			Id:   string(v.ID),
			Name: v.Name,
		})
	}
	return &ListSamplersResp{Samplers: out}, nil
}

func (s *runtimeServer) ListModels(ctx context.Context, req *ListModelsReq) (*ListModelsResp, error) {
	list, err := s.r.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]*ModelInfo, 0, len(list))
	for _, v := range list {
		out = append(out, &ModelInfo{
			Id:     string(v.ID),
			Name:   v.Name,
			Title:  v.Title,
			File:   v.File,
			Sha256: "",                    // TODO
			Kind:   ModelInfo_UNSPECIFIED, // TODO
		})
	}
	return &ListModelsResp{Models: out}, nil
}

func (s *runtimeServer) TextToImage(ctx context.Context, req *TextToImageReq) (*TextToImageResp, error) {
	rc, err := s.r.TextToImagePNG(ctx, types.TextToImageParams{
		Model:    types.ModelID(req.ModelId),
		Prompt:   req.PositivePrompt,
		Negative: req.NegativePrompt,
		Seed:     req.RngSeed,
		Sampler:  types.SamplerID(req.SamplerId),
		Steps:    int(req.SamplerSteps),
		CfgScale: req.CfgScale,
		ClipSkip: int(req.ClipSkip),
		Width:    int(req.ImageWidth),
		Height:   int(req.ImageHeight),
	})
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	data, err := io.ReadAll(rc)
	if err != nil {
		return nil, err
	}
	return &TextToImageResp{
		RngSeed: req.RngSeed,
		Image: &Image{
			Width:  req.ImageWidth,
			Height: req.ImageHeight,
			Image:  &Image_Png{Png: data},
		},
	}, nil
}
