package llama2

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"unicode"
)

type Token int

const (
	TokenBOS = Token(1)
	TokenEOS = Token(2)
)

type tokenIndex struct {
	str string
	id  Token
}

// Tokenizer is Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens.
type Tokenizer struct {
	vocab          []string
	vocabScores    []float32
	sortedVocab    []tokenIndex
	maxTokenLength uint
}

func compareTokens(a, b tokenIndex) int {
	return strings.Compare(a.str, b.str)
}

func NewTokenizer(path string, vocabSize int) (*Tokenizer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// TODO(karpathy): I should have written the vocab_size into the tokenizer file... sigh.
	t := &Tokenizer{
		vocab:       make([]string, vocabSize),
		vocabScores: make([]float32, vocabSize),
		sortedVocab: nil, // initialized lazily
	}

	var buf [8]byte
	if _, err = io.ReadFull(file, buf[:4]); err != nil {
		return nil, err
	}
	t.maxTokenLength = uint(binary.LittleEndian.Uint32(buf[:]))

	for i := range vocabSize {
		if _, err = io.ReadFull(file, buf[:8]); err != nil {
			return nil, err
		}
		t.vocabScores[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[0:]))
		sz := int(binary.LittleEndian.Uint32(buf[4:]))
		sbuf := make([]byte, sz)
		if _, err = io.ReadFull(file, sbuf); err != nil {
			return nil, err
		}
		t.vocab[i] = string(sbuf)
	}
	return t, nil
}

func (t *Tokenizer) Decode(prevToken, token Token) string {
	piece := t.vocab[token]
	// following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
	if prevToken == TokenBOS && piece[0] == ' ' {
		piece = piece[1:]
	}
	// careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
	// parse this and convert and return the actual byte
	var b byte
	if n, err := fmt.Sscanf(piece, "<0x%02X>", &b); err == nil && n == 1 {
		piece = string(b)
	}
	return piece
}

func safeFprintf(w io.Writer, piece string) error {
	// piece might be a raw byte token, and we only want to print printable chars or whitespace
	// because some of the other bytes can be various control codes, backspace, etc.
	if len(piece) == 0 {
		return nil
	}
	if len(piece) == 1 {
		b := rune(piece[0])
		if !(unicode.IsPrint(b) || unicode.IsSpace(b)) {
			return nil // bad byte, don't print it
		}
	}
	_, err := w.Write([]byte(piece))
	return err
}

func (t *Tokenizer) Lookup(str string) Token {
	// efficiently find the perfect match for str in vocab, return its index or -1 if not found
	tok := tokenIndex{str: str} // acts as the key to search for
	res, ok := slices.BinarySearchFunc(t.sortedVocab, tok, compareTokens)
	if !ok {
		return -1
	}
	return t.sortedVocab[res].id
}

func (t *Tokenizer) Encode(text string, addBOS, addEOS bool) []Token {
	tokens := make([]Token, 0, len(text)+3) // +3 for '\0', ?BOS, ?EOS

	// encode the string text (input) into an upper-bound tokens slice
	// bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)

	if t.sortedVocab == nil {
		// lazily malloc and sort the vocabulary
		t.sortedVocab = make([]tokenIndex, len(t.vocab))
		for i := range t.vocab {
			t.sortedVocab[i].str = t.vocab[i]
			t.sortedVocab[i].id = Token(i)
		}
		slices.SortFunc(t.sortedVocab, compareTokens)
	}

	// add optional BOS (=1) token, if desired
	if addBOS {
		tokens = append(tokens, TokenBOS)
	}

	// add_dummy_prefix is true by default
	// so prepend a dummy prefix token to the input string, but only if text != ""
	// TODO: pretty sure this isn't correct in the general case but I don't have the
	// energy to read more of the sentencepiece code to figure out what it's doing
	if len(text) != 0 {
		dummyPrefix := t.Lookup(" ")
		tokens = append(tokens, dummyPrefix)
	}

	// process the raw (UTF-8) byte sequence of the input string
	for _, c := range text {
		str := string(c)
		// ok c+1 is not a continuation byte, so we've read in a full codepoint
		id := t.Lookup(str)

		if id != -1 {
			// we found this codepoint in vocab, add it as a token
			tokens = append(tokens, id)
		} else {
			// byte_fallback encoding: just encode each byte as a token
			// +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
			// so the individual bytes only start at index 3
			raw := []byte(str)
			for i := range raw {
				tokens = append(tokens, Token(raw[i])+3)
			}
		}
	}

	// merge the best consecutive pair each iteration, according the scores in vocab_scores
	for {
		bestScore := float32(-1e10)
		bestID := Token(-1)
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			// check if we can merge the pair (tokens[i], tokens[i+1])
			str := t.vocab[tokens[i]] + t.vocab[tokens[i+1]]
			id := t.Lookup(str)
			if id != -1 && t.vocabScores[id] > bestScore {
				// this merge pair exists in vocab! record its score and position
				bestScore = t.vocabScores[id]
				bestID = id
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break // we couldn't find any more pairs to merge, so we're done
		}

		// merge the consecutive pair (best_idx, best_idx+1) into new token best_id
		tokens[bestIdx] = bestID
		// delete token at position best_idx+1, shift the entire sequence back 1
		for i := bestIdx + 1; i < len(tokens)-1; i++ {
			tokens[i] = tokens[i+1]
		}
		tokens = tokens[:len(tokens)-1] // token length decreased
	}

	// add optional EOS (=2) token, if desired
	if addEOS {
		tokens = append(tokens, TokenEOS)
	}
	return tokens
}
