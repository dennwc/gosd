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
	fOut      = flag.String("o", "output.png", "path to write result image to")
	fPrompt   = flag.String("p", "a cat", "the prompt to render")
	fNegative = flag.String("n", "", "the negative prompt")
	fWidth    = flag.Int("W", 256, "image width")
	fHeight   = flag.Int("H", 256, "image height")
	fSeed     = flag.Int64("seed", -1, "RNG seed")
	fSteps    = flag.Int("steps", 20, "number of sample steps")
	fCfgScale = flag.Float64("cfg-scale", 7, "unconditional guidance scale")
	fSampling = flag.String("sampling-method", "euler_a", "sampling method")
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
		ModelPath: *fModel,
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
	img := c.TextToImage(gosd.TextToImageParams{
		Prompt:   *fPrompt,
		Negative: *fNegative,
		CfgScale: *fCfgScale,
		Width:    *fWidth,
		Height:   *fHeight,
		Sampling: smethod,
		Steps:    *fSteps,
		Seed:     seed,
	})
	if img == nil {
		return errors.New("image generation failed")
	}
	slog.Info("generation complete", "t", time.Since(start))
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
