package protocol

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
)

func (p *Image) Decode() (image.Image, error) {
	w, h := int(p.Width), int(p.Height)
	rect := image.Rect(0, 0, w, h)
	switch img := p.Image.(type) {
	default:
		return nil, fmt.Errorf("unknown image type: %T", img)
	case *Image_Rgb8:
		if len(img.Rgb8) != w*h*3 {
			return nil, fmt.Errorf("unexpected image data size")
		}
		out := image.NewNRGBA(rect)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				src := img.Rgb8[(y*w+x)*3 : (y*w+x+1)*3]
				dst := out.Pix[y*out.Stride+x*4 : y*out.Stride+(x+1)*4]
				copy(dst, src[:3])
				dst[3] = 0xff // alpha
			}
		}
		return out, nil
	case *Image_Rgba8:
		if len(img.Rgba8) != w*h*4 {
			return nil, fmt.Errorf("unexpected image data size")
		}
		return &image.NRGBA{
			Pix:    img.Rgba8,
			Stride: w * 4,
			Rect:   rect,
		}, nil
	case *Image_Png:
		return png.Decode(bytes.NewReader(img.Png))
	case *Image_Jpeg:
		return jpeg.Decode(bytes.NewReader(img.Jpeg))
	}
}
