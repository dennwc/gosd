package llama2

import (
	"context"
	"errors"
	"io/fs"
	"strings"
	"testing"

	"github.com/shoenig/test/must"
)

func TestLocal(t *testing.T) {
	const exp = `One day, a little boy named Tim went to the park with his mom. Tim saw a big, brown dog. The dog had a sad look on its face. Tim wanted to help the dog feel better.
Tim's mom saw that the dog was upset. She knew how it felt to feel alone. She wanted to encourage the dog to be happy. So, she gave the dog a treat. The dog's eyes lit up with joy.
But then, something unexpected happened! The dog did not feel scared. The dog just wanted to play! Tim was so surprised. He gave the dog a treat from his pocket. Now, Tim and the dog were happy together at the park.`
	var buf strings.Builder
	st, err := Run(context.Background(), &buf, Params{
		TokenizerPath:  "tokenizer.bin",
		CheckpointPath: "stories15M.bin",
		Steps:          256,
		Temperature:    1.0, // 0.0 = greedy deterministic. 1.0 = original. don't set higher
		TopP:           0.9, // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
		Seed:           1234567890,
		Prompt:         "",
	})
	if errors.Is(err, fs.ErrNotExist) {
		t.SkipNow()
	}
	must.NoError(t, err)
	t.Logf("tokens per sec: %.2f", st.TokensPerSec)
	must.EqOp(t, exp, buf.String())
}
