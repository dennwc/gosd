package llama2

import (
	"context"
	"errors"
	"io"
	"time"
)

// https://github.com/karpathy/llama2.c/blob/2fbf7059aab6f7e44047da1ff5c0ba53057a248e/run.c

type GenerateStats struct {
	Tokens       int
	TokensPerSec float64
}

func Generate(ctx context.Context, w io.Writer, tr *Transformer, tk *Tokenizer, s *Sampler, prompt string, steps int) (*GenerateStats, error) {
	// encode the (string) prompt into tokens sequence
	promptTokens := tk.Encode(prompt, true, false)
	if len(promptTokens) < 1 {
		return nil, errors.New("something is wrong, expected at least 1 prompt token")
	}

	// start the main loop
	var (
		start time.Time         // used to time our code, only initialized after first iteration
		next  Token             // will store the next token in the sequence
		token = promptTokens[0] // kick off with the first token in the prompt
		pos   = 0               // position in the sequence
	)
	for pos < steps {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		// forward the transformer to get logits for the next token
		logits := tr.Forward(token, pos)

		// advance the state machine
		if pos < len(promptTokens)-1 {
			// if we are still processing the input prompt, force the next prompt token
			next = promptTokens[pos+1]
		} else {
			// otherwise sample the next token from the logits
			next = s.Sample(logits)
		}
		pos++

		// data-dependent terminating condition: the BOS (=1) token delimits sequences
		if next == 1 {
			break
		}

		// print the token as string, decode it with the Tokenizer object
		piece := tk.Decode(token, next)
		err := safeFprintf(w, piece) // same as printf("%s", piece), but skips "unsafe" bytes
		if err != nil {
			return nil, err
		}
		token = next

		// init the timer here because the first iteration can be slower
		if start.IsZero() {
			start = time.Now()
		}
	}

	return &GenerateStats{
		Tokens: pos,
		// report achieved tok/s (pos-1 because the timer starts after first iteration)
		TokensPerSec: float64(pos-1) / time.Since(start).Seconds(),
	}, nil
}

func NewParams() Params {
	return Params{
		TokenizerPath: "tokenizer.bin",
		Steps:         256,
		Temperature:   1.0, // 0.0 = greedy deterministic. 1.0 = original. don't set higher
		TopP:          0.9, // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	}
}

type Params struct {
	TokenizerPath  string
	CheckpointPath string
	Steps          int
	Temperature    float32
	TopP           float32
	Seed           int64
	Prompt         string
}

func (p *Params) validate() {
	// parameter validation/overrides
	if p.Seed <= 0 {
		p.Seed = time.Now().UnixNano()
	}
	if p.Temperature < 0.0 {
		p.Temperature = 0.0
	}
	if p.TopP < 0.0 || 1.0 < p.TopP {
		p.TopP = 0.9
	}
	if p.Steps < 0 {
		p.Steps = 0
	}
}

func Run(ctx context.Context, w io.Writer, p Params) (*GenerateStats, error) {
	p.validate()

	// build the Transformer via the model .bin file
	transformer, err := NewTransformer(p.CheckpointPath)
	if err != nil {
		return nil, err
	}
	defer transformer.Close()

	if p.Steps == 0 || p.Steps > transformer.Config.SeqLen {
		p.Steps = transformer.Config.SeqLen // override to ~max length
	}

	// build the Tokenizer via the tokenizer .bin file
	tokenizer, err := NewTokenizer(p.TokenizerPath, transformer.Config.VocabSize)
	if err != nil {
		return nil, err
	}

	// build the Sampler
	sampler := NewSampler(transformer.Config.VocabSize, p.Temperature, p.TopP, p.Seed)

	return Generate(ctx, w, transformer, tokenizer, sampler, p.Prompt, p.Steps)
}
