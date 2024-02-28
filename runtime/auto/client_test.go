package auto

import (
	"context"
	"io"
	"os"
	"testing"
	"time"

	"github.com/shoenig/test/must"

	"github.com/dennwc/gosd/runtime/types"
)

func TestAutomatic1111(t *testing.T) {
	url := os.Getenv("AUTO_URL")
	if url == "" {
		t.Skip("AUTO_URL must be set")
	}
	model := types.ModelID(os.Getenv("AUTO_MODEL"))
	cli := NewClient(url)
	ctx := context.Background()

	models, err := cli.ListModels(ctx)
	must.NoError(t, err)
	must.SliceNotEmpty(t, models)
	for _, m := range models {
		t.Logf("%+v", m)
	}

	samplers, err := cli.ListSamplers(ctx)
	must.NoError(t, err)
	must.SliceNotEmpty(t, samplers)
	for _, m := range samplers {
		t.Logf("%+v", m)
	}

	start := time.Now()
	rc, err := cli.TextToImagePNG(ctx, types.TextToImageParams{
		Model:    model,
		Prompt:   "photo of a cat",
		CfgScale: 7,
		Steps:    20,
		Seed:     -1,
		Width:    256,
		Height:   512,
	})
	t.Log("done in", time.Since(start))
	must.NoError(t, err)
	defer rc.Close()

	f, err := os.Create("output.png")
	must.NoError(t, err)
	defer f.Close()

	_, err = io.Copy(f, rc)
	must.NoError(t, err)
}
