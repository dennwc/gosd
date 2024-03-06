package meta

import (
	"encoding/json"
	"io"
	"path/filepath"

	"github.com/dennwc/gosd/runtime/types"
)

type ImageReaderFunc func(r io.Reader) (ImageMetadata, error)

var imageByExt = make(map[string]ImageReaderFunc)

func RegisterImageByExt(ext string, fnc ImageReaderFunc) {
	if _, ok := imageByExt[ext]; ok {
		panic("already registered")
	}
	imageByExt[ext] = fnc
}

type ImageMetadata interface {
	TextToImage() *types.TextToImageParams
}

type RawMetaText interface {
	RawMetadata() string
}

type RawMetaJSON interface {
	RawMetadata() json.RawMessage
}

func ReadImage(name string, r io.Reader) (ImageMetadata, error) {
	ext := filepath.Ext(name)
	fnc := imageByExt[ext]
	if fnc == nil {
		return nil, nil
	}
	return fnc(r)
}
