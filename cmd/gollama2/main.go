package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/dennwc/gosd/llama2"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	flag.Parse()
	if err := run(ctx, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

var (
	fTemp   = flag.Float64("t", 1.0, "temperature in [0,inf]")
	fTopP   = flag.Float64("p", 0.9, "p value in top-p (nucleus) sampling in [0,1]")
	fSeed   = flag.Int64("s", 0, "random seed")
	fSteps  = flag.Int("n", 256, "number of steps to run for")
	fPrompt = flag.String("i", "", "input prompt")
	fToken  = flag.String("z", "tokenizer.bin", "path to custom tokenizer")
)

func run(ctx context.Context, args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("expected a path to a model")
	}
	st, err := llama2.Run(ctx, os.Stdout, llama2.Params{
		TokenizerPath:  *fToken,
		CheckpointPath: args[0],
		Steps:          *fSteps,
		Temperature:    float32(*fTemp),
		TopP:           float32(*fTopP),
		Seed:           *fSeed,
		Prompt:         *fPrompt,
	})
	if err != nil {
		return err
	}
	fmt.Println()
	fmt.Fprintf(os.Stderr, "achieved tok/s: %.6f\n", st.TokensPerSec)
	return nil
}
