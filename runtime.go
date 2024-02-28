package gosd

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/png"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/dennwc/gosd/runtime"
	"github.com/dennwc/gosd/runtime/types"
)

type RuntimeParams struct {
	ModelDir     string
	EmbeddingDir string
	RNG          RNGType
	Threads      int
	Type         Type
	Log          *slog.Logger
}

func NewRuntime(p RuntimeParams) (runtime.Runtime, error) {
	if p.Log == nil {
		p.Log = slog.Default()
	}
	if _, err := os.Stat(p.ModelDir); err != nil {
		return nil, err
	}
	return &runtimeSD{p: p, log: p.Log}, nil
}

type runtimeSD struct {
	p   RuntimeParams
	log *slog.Logger
}

func (r *runtimeSD) ListSamplers(ctx context.Context) ([]types.SamplerInfo, error) {
	var out []types.SamplerInfo
	for _, s := range []SamplingMethod{
		EULER_A, EULER,
		HEUN, DPM2,
		DPMPP2S_A, DPMPP2M,
		DPMPP2Mv2, LCM,
	} {
		out = append(out, types.SamplerInfo{
			ID:   types.SamplerID(s.ID()),
			Name: s.String(),
		})
	}
	return out, nil
}

func (r *runtimeSD) ListModels(ctx context.Context) ([]types.ModelInfo, error) {
	var out []types.ModelInfo
	err := filepath.Walk(r.p.ModelDir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil // continue
		}
		// TODO: filter known file extensions
		rel, err := filepath.Rel(r.p.ModelDir, path)
		if err != nil {
			return err
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		out = append(out, types.ModelInfo{
			ID:    types.ModelID(rel),
			Name:  strings.ReplaceAll(rel, string(filepath.Separator), "_"),
			Title: rel,
			File:  abs,
		})
		return nil
	})
	return out, err
}

func (r *runtimeSD) TextToImage(ctx context.Context, p types.TextToImageParams) (image.Image, error) {
	model := string(p.Model)
	model = filepath.Clean(model)
	c, err := New(Params{
		ModelPath:       filepath.Join(r.p.ModelDir, model),
		EmbeddingDir:    r.p.EmbeddingDir,
		FreeImmediately: true,
		Threads:         r.p.Threads,
		Type:            r.p.Type,
		RNG:             r.p.RNG,
		Schedule:        ScheduleDefault,
		Log:             r.log,
	})
	if err != nil {
		return nil, err
	}
	defer c.Close()
	var sampler SamplingMethod
	switch p.Sampler {
	case "", types.SamplerEuler:
		sampler = EULER
	default:
		sampler, err = ParseSamplingMethod(string(p.Sampler))
		if err != nil {
			return nil, err
		}
	}
	imgs, err := c.TextToImages(TextToImageParams{
		Prompt:     p.Prompt,
		Negative:   p.Negative,
		ClipSkip:   p.ClipSkip,
		CfgScale:   p.CfgScale,
		Width:      p.Width,
		Height:     p.Height,
		Sampling:   sampler,
		Steps:      p.Steps,
		BatchCount: 1,
		Seed:       p.Seed,
	})
	if err != nil {
		return nil, err
	}
	if len(imgs) == 0 || imgs[0] == nil {
		return nil, errors.New("no image returned")
	}
	return imgs[0], nil
}

func (r *runtimeSD) TextToImagePNG(ctx context.Context, p types.TextToImageParams) (io.ReadCloser, error) {
	img, err := r.TextToImage(ctx, p)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return io.NopCloser(&buf), nil
}

func (r *runtimeSD) Close() {}
