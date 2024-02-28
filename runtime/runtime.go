// Package runtime provides common types and interfaces that all Stable Diffusion runtimes share.
package runtime

import (
	"context"
	"image"
	"io"

	"github.com/dennwc/gosd/runtime/types"
)

// Runtime is an interface for Stable Diffusion runtimes.
type Runtime interface {
	// ListModels lists all available Stable Diffusion models.
	ListModels(ctx context.Context) ([]types.ModelInfo, error)

	// ListSamplers lists all available samplers.
	ListSamplers(ctx context.Context) ([]types.SamplerInfo, error)

	// TextToImage generates an image from a text prompt.
	TextToImage(ctx context.Context, params types.TextToImageParams) (image.Image, error)

	// TextToImagePNG generates an image from a text prompt and encodes it as a PNG.
	// Encoded image preserves any associated image metadata.
	TextToImagePNG(ctx context.Context, params types.TextToImageParams) (io.ReadCloser, error)

	// Close the runtime and free associated resources.
	Close()
}
