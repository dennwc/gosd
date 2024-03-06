package auto

import (
	"testing"

	"github.com/shoenig/test/must"
)

var cases = []struct {
	name   string
	str    string
	params []Param
	meta   *Metadata
}{
	{
		name: "parameters",
		str:  "(111, 222 333), 444 (555 666), (777), 888, 999, (aaa bbb:1.2), (ccc:1.2), ddd\nNegative prompt: (aaa bbb:1.3), ccc, ddd\nSteps: 40, Sampler: DPM++ 2M SDE Karras, CFG scale: 6.5, Seed: 123456, Size: 256x512, Model hash: abcd1234, TI hashes: \"aaa: dddd1111, bbb-ccc: eeee2222\", Version: v1.7.0",
		params: []Param{
			{Value: "(111, 222 333), 444 (555 666), (777), 888, 999, (aaa bbb:1.2), (ccc:1.2), ddd"},
			{Name: "Negative prompt", Value: "(aaa bbb:1.3), ccc, ddd"},
			{Name: "Steps", Value: "40"},
			{Name: "Sampler", Value: "DPM++ 2M SDE Karras"},
			{Name: "CFG scale", Value: "6.5"},
			{Name: "Seed", Value: "123456"},
			{Name: "Size", Value: "256x512"},
			{Name: "Model hash", Value: "abcd1234"},
			{Name: "TI hashes", Value: "\"aaa: dddd1111, bbb-ccc: eeee2222\""},
			{Name: "Version", Value: "v1.7.0"},
		},
		meta: &Metadata{
			Prompt:    "(111, 222 333), 444 (555 666), (777), 888, 999, (aaa bbb:1.2), (ccc:1.2), ddd",
			Negative:  "(aaa bbb:1.3), ccc, ddd",
			Steps:     40,
			Sampler:   "DPM++ 2M SDE Karras",
			CfgScale:  6.5,
			Seed:      123456,
			Width:     256,
			Height:    512,
			ModelHash: "abcd1234",
			TIHashes: map[string]string{
				"aaa":     "dddd1111",
				"bbb-ccc": "eeee2222",
			},
			Version: "v1.7.0",
		},
	},
	{
		name: "model hashes",
		str:  "aaa: dddd1111, bbb-ccc: eeee2222",
		params: []Param{
			{Name: "aaa", Value: "dddd1111"},
			{Name: "bbb-ccc", Value: "eeee2222"},
		},
	},
}

func TestParseParams(t *testing.T) {
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			got := parseParams(c.str)
			must.Eq(t, c.params, got)
		})
	}
}

func TestParse(t *testing.T) {
	for _, c := range cases {
		if c.meta == nil {
			continue
		}
		c := c
		t.Run(c.name, func(t *testing.T) {
			m, err := Parse(c.str)
			must.NoError(t, err)
			exp := c.meta
			exp.RawText = c.str
			exp.RawParams = c.params
			must.Eq(t, exp, m)
		})
	}
}
