package auto

import (
	"errors"
	"strconv"
	"strings"

	"github.com/dennwc/gosd/meta"
	"github.com/dennwc/gosd/meta/pnginfo"
	"github.com/dennwc/gosd/runtime/types"
)

const PNGTextName = "parameters"

var _ meta.RawMetaText = (*Metadata)(nil)

func init() {
	pnginfo.RegisterTextHandler(PNGTextName, func(s string) (meta.ImageMetadata, error) {
		m, err := Parse(s)
		if err != nil || m == nil {
			return nil, err
		}
		return m, nil
	})
}

type Param struct {
	Name  string
	Value string
}

type Metadata struct {
	Prompt    string
	Negative  string
	Steps     int
	Sampler   string
	CfgScale  float64
	Seed      int64
	Width     int
	Height    int
	ModelHash string
	TIHashes  map[string]string
	Version   string
	RawText   string
	RawParams []Param
}

func (m *Metadata) RawMetadata() string {
	return m.RawText
}

func (m *Metadata) TextToImage() *types.TextToImageParams {
	return &types.TextToImageParams{
		Prompt:   m.Prompt,
		Negative: m.Negative,
		Model:    types.ModelID(m.ModelHash),
		Seed:     m.Seed,
		Sampler:  types.SamplerID(m.Sampler),
		Steps:    m.Steps,
		CfgScale: m.CfgScale,
		ClipSkip: 0, // TODO
		Width:    m.Width,
		Height:   m.Height,
	}
}

func ParseParams(s string) []Param {
	return parseLine(s, true)
}

func parseParams(s string) []Param {
	lines := strings.Split(s, "\n")
	var params []Param
	for li, line := range lines {
		i := strings.Index(line, ": ")
		j := strings.IndexByte(line, ',')
		if i < 0 || j < i {
			params = append(params, Param{Value: line})
			continue
		}
		params = append(params, parseLine(line, li == len(lines)-1)...)
	}
	return params
}

func parseLine(line string, isLast bool) []Param {
	var params []Param
	for len(line) > 0 {
		i := strings.Index(line, ": ")
		j := strings.IndexByte(line, ',')
		if i < 0 || (j >= 0 && j < i) {
			params = append(params, Param{Value: line})
			break
		}
		name := line[:i]
		line = line[i+2:]
		if len(line) == 0 {
			params = append(params, Param{Name: name})
			break
		}
		if !isLast {
			params = append(params, Param{Name: name, Value: line})
			break
		}
		if line[0] != '"' {
			i = strings.Index(line, ", ")
			if i < 0 {
				params = append(params, Param{Name: name, Value: line})
				break
			}
			val := line[:i]
			line = line[i+2:]
			params = append(params, Param{Name: name, Value: val})
		} else {
			var ei = -1
			for i := range line {
				if i == 0 {
					continue
				}
				if line[i] == '"' && line[i-1] != '\\' {
					ei = i
					break
				}
			}
			if ei < 0 {
				params = append(params, Param{Name: name, Value: line})
				break
			}
			val := line[:ei+1]
			line = line[ei+1:]
			params = append(params, Param{Name: name, Value: val})
			if len(line) != 0 && line[0] == ',' {
				line = line[1:]
			}
			if len(line) != 0 && line[0] == ' ' {
				line = line[1:]
			}
		}
	}
	return params
}

func Parse(s string) (*Metadata, error) {
	if len(s) == 0 {
		return nil, nil
	}
	m := &Metadata{RawParams: parseParams(s), RawText: s}
	var last error
	for i, p := range m.RawParams {
		val, err := maybeUnquote(p.Value)
		if err != nil {
			last = err
			val = p.Value
		}
		switch p.Name {
		case "":
			if i == 0 {
				m.Prompt = val
			}
		case "Negative prompt":
			m.Negative = val
		case "Steps":
			m.Steps, err = strconv.Atoi(val)
		case "Sampler":
			m.Sampler = val
		case "CFG scale":
			m.CfgScale, err = strconv.ParseFloat(val, 64)
		case "Seed":
			m.Seed, err = strconv.ParseInt(val, 10, 64)
		case "Size":
			sub := strings.SplitN(val, "x", 2)
			if len(sub) != 2 {
				err = errors.New("invalid size")
			} else {
				var err1, err2 error
				m.Width, err1 = strconv.Atoi(sub[0])
				m.Height, err2 = strconv.Atoi(sub[1])
				if err1 != nil {
					last = err1
				}
				if err2 != nil {
					last = err2
				}
			}
		case "Model hash":
			m.ModelHash = val
		case "TI hashes":
			m.TIHashes = make(map[string]string)
			for _, kv := range ParseParams(val) {
				m.TIHashes[kv.Name] = kv.Value
			}
		case "Version":
			m.Version = val
		}
		if err != nil {
			last = err
		}
	}
	return m, last
}

func maybeUnquote(s string) (string, error) {
	if len(s) == 0 {
		return s, nil
	}
	if s[0] != '"' {
		return s, nil
	}
	return strconv.Unquote(s)
}
