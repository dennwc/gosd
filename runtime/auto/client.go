// Package auto implements Automatic1111 Stable Diffusion WebUI client.
package auto

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"strings"

	_ "github.com/dennwc/gosd/meta/auto"

	"github.com/dennwc/gosd/runtime"
	"github.com/dennwc/gosd/runtime/types"
)

// https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

const (
	SamplerEuler = types.SamplerID("Euler")
)

var _ runtime.Runtime = (*Client)(nil)

// NewClient creates an Automatic1111 WebUI client for a given URL.
func NewClient(url string) *Client {
	return &Client{
		cli: http.DefaultClient,
		url: url,
	}
}

// Client for Automatic1111 WebUI API.
type Client struct {
	cli *http.Client
	url string
}

// Close the client.
func (c *Client) Close() {}

// ListModels lists all available Stable Diffusion models.
func (c *Client) ListModels(ctx context.Context) ([]types.ModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.url+"/sdapi/v1/sd-models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.cli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http status: %s", resp.Status)
	}
	var list []struct {
		Title  string `json:"title"`
		Name   string `json:"model_name"`
		Hash   string `json:"hash"`
		SHA256 string `json:"sha256"`
		File   string `json:"filename"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, err
	}
	out := make([]types.ModelInfo, 0, len(list))
	for _, v := range list {
		out = append(out, types.ModelInfo{
			ID:    types.ModelID(v.Hash),
			File:  v.File,
			Name:  v.Name,
			Title: v.Title,
		})
	}
	return out, nil
}

// ListSamplers lists all available samplers.
func (c *Client) ListSamplers(ctx context.Context) ([]types.SamplerInfo, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.url+"/sdapi/v1/samplers", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.cli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http status: %s", resp.Status)
	}
	var list []struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, err
	}
	out := make([]types.SamplerInfo, 0, len(list))
	for _, v := range list {
		out = append(out, types.SamplerInfo{
			ID:   types.SamplerID(v.Name),
			Name: v.Name,
		})
	}
	return out, nil
}

type textToImageSettings struct {
	ModelCheckpoint string `json:"sd_model_checkpoint,omitempty"`
	CLIPStop        int    `json:"CLIP_stop_at_last_layers,omitempty"`
}

type textToImageReq struct {
	Prompt                  string              `json:"prompt"`
	Negative                string              `json:"negative_prompt"`
	Seed                    int64               `json:"seed"`
	SubSeed                 int64               `json:"subseed"`
	BatchSize               int                 `json:"batch_size"`
	NIter                   int                 `json:"n_iter"`
	Steps                   int                 `json:"steps"`
	CfgScale                float64             `json:"cfg_scale"`
	Width                   int                 `json:"width"`
	Height                  int                 `json:"height"`
	SamplerName             string              `json:"sampler_name"`
	SamplerIndex            string              `json:"sampler_index"`
	SendImages              bool                `json:"send_images"`
	SaveImages              bool                `json:"save_images"`
	DoNotSaveSamples        bool                `json:"do_not_save_samples"`
	DoNotSaveGrid           bool                `json:"do_not_save_grid"`
	OverrideSettings        textToImageSettings `json:"override_settings"`
	OverrideSettingsRestore bool                `json:"override_settings_restore_afterwards"`
}

// TextToImage generates an image from a text prompt.
func (c *Client) TextToImage(ctx context.Context, params types.TextToImageParams) (image.Image, error) {
	rc, err := c.TextToImagePNG(ctx, params)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return png.Decode(rc)
}

// TextToImagePNG generates an image from a text prompt and encodes it as a PNG.
// Encoded image preserves any associated image metadata.
func (c *Client) TextToImagePNG(ctx context.Context, params types.TextToImageParams) (io.ReadCloser, error) {
	switch params.Sampler {
	case "", types.SamplerEuler:
		params.Sampler = SamplerEuler
	}
	body, err := json.Marshal(textToImageReq{
		Prompt:                  params.Prompt,
		Negative:                params.Negative,
		Seed:                    params.Seed,
		SubSeed:                 -1,
		BatchSize:               1,
		NIter:                   1,
		Steps:                   params.Steps,
		CfgScale:                params.CfgScale,
		Width:                   params.Width,
		Height:                  params.Height,
		SamplerIndex:            string(params.Sampler),
		SendImages:              true,
		SaveImages:              false,
		DoNotSaveSamples:        true,
		DoNotSaveGrid:           true,
		OverrideSettingsRestore: true,
		OverrideSettings: textToImageSettings{
			ModelCheckpoint: string(params.Model),
		},
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.url+"/sdapi/v1/txt2img", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	resp, err := c.cli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var out struct {
			Detail string `json:"detail"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		if out.Detail != "" {
			return nil, errors.New(out.Detail)
		}
		return nil, fmt.Errorf("http status: %s", resp.Status)
	}
	var out struct {
		Images []string `json:"images"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Images) == 0 {
		return nil, errors.New("no images returned")
	}
	r := base64.NewDecoder(base64.StdEncoding, strings.NewReader(out.Images[0]))
	return io.NopCloser(r), nil
}
