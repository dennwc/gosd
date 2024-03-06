package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	_ "github.com/dennwc/gosd/meta/all"

	"github.com/dennwc/gosd/meta"
	"github.com/dennwc/gosd/runtime/types"
)

func main() {
	flag.Parse()
	if err := run(flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

type Meta struct {
	TextToImage *types.TextToImageParams `json:"text_to_image,omitempty"`
	Raw         any                      `json:"raw"`
}

func run(files []string) error {
	if len(files) == 0 {
		return errors.New("no files specified")
	}
	var last error
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "\t")
	for _, path := range files {
		m, err := readMeta(path)
		if err != nil {
			last = err
			slog.Error("error reading file", "path", path, "err", err)
		}
		if m == nil {
			if err = enc.Encode(nil); err != nil {
				return err
			}
			continue
		}
		var raw any = m
		switch m := m.(type) {
		case meta.RawMetaJSON:
			raw = m.RawMetadata()
		case meta.RawMetaText:
			raw = m.RawMetadata()
		}
		if err = enc.Encode(Meta{
			TextToImage: m.TextToImage(),
			Raw:         raw,
		}); err != nil {
			return err
		}
	}
	return last
}

func readMeta(path string) (meta.ImageMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return meta.ReadImage(filepath.Base(path), f)
}
