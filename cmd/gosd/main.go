package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image/png"
	"log/slog"
	"math/rand"
	"os"
	"os/signal"
	"time"

	"github.com/dennwc/gosd"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	flag.Parse()
	if err := run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

var (
	fModel    = flag.String("m", "", "path to model")
	fVAE      = flag.String("vae", "", "path to vae")
	fEmbed    = flag.String("embd-dir", "", "path to embeddings")
	fOut      = flag.String("o", "output.png", "path to write result image to")
	fPrompt   = flag.String("p", "a cat", "the prompt to render")
	fNegative = flag.String("n", "", "the negative prompt")
	fWidth    = flag.Int("W", 256, "image width")
	fHeight   = flag.Int("H", 256, "image height")
	fSeed     = flag.Int64("seed", -1, "RNG seed")
	fSteps    = flag.Int("steps", 20, "number of sample steps")
	fCfgScale = flag.Float64("cfg-scale", 7, "unconditional guidance scale")
	fSampling = flag.String("sampling-method", "euler_a", "sampling method")
	fThreads  = flag.Int("threads", -1, "number of threads to use during computation")
)

func run(ctx context.Context) error {
	if *fModel == "" {
		return errors.New("model path must be specified")
	}
	smethod, err := gosd.ParseSamplingMethod(*fSampling)
	if err != nil {
		return err
	}
	slog.Info("loading model", "path", *fModel)
	c, err := gosd.New(gosd.Params{
		ModelPath:       *fModel,
		VAEPath:         *fVAE,
		EmbeddingDir:    *fEmbed,
		FreeImmediately: true,
		Threads:         *fThreads,
		Type:            gosd.TypeDefault,
		RNG:             gosd.CUDA_RNG,
		Schedule:        gosd.ScheduleDefault,
	})
	if err != nil {
		return err
	}
	defer c.Close()

	seed := *fSeed
	if seed < 0 {
		seed = rand.Int63()
	}

	start := time.Now()
	slog.Info("generating image...")
	imgs, err := c.TextToImages(gosd.TextToImageParams{
		Prompt:     *fPrompt,
		Negative:   *fNegative,
		ClipSkip:   -1,
		CfgScale:   *fCfgScale,
		Width:      *fWidth,
		Height:     *fHeight,
		Sampling:   smethod,
		Steps:      *fSteps,
		Seed:       seed,
		BatchCount: 1,
	})
	if err != nil {
		return err
	}
	img := imgs[0]
	slog.Info("generation complete", "t", time.Since(start).Round(time.Millisecond))
	slog.Info("saving to", "path", *fOut)
	f, err := os.Create(*fOut)
	if err != nil {
		return err
	}
	defer f.Close()
	if err = png.Encode(f, img); err != nil {
		return err
	}
	return f.Close()
}
