# Stable Diffusion parameters extractor

This tool extracts Stable Diffusion parameters from image files.

Currently, the following UIs are supported:
- [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Invoke](https://github.com/invoke-ai/InvokeAI)

## Using as a CLI

```shell
go install github.com/dennwc/gosd/cmd/sdinfo@latest
sdinfo image.png
```

Example output (Invoke):

```json
{
        "text_to_image": {
                "model_id": "stable-diffusion-v1-5",
                "positive_prompt": "a cat",
                "seed": 620742083,
                "sampler_id": "euler",
                "steps": 50,
                "cfg_scale": 7.5,
                "width": 680,
                "height": 384
        },
        "raw": {
                "generation_mode": "txt2img",
                "positive_prompt": "a cat",
                "negative_prompt": "",
                "width": 680,
                "height": 384,
                "seed": 620742083,
                "rand_device": "cpu",
                "cfg_scale": 7.5,
                "cfg_rescale_multiplier": 0.0,
                "steps": 50,
                "scheduler": "euler",
                "clip_skip": 0,
                "model": {
                        "model_name": "stable-diffusion-v1-5",
                        "base_model": "sd-1",
                        "model_type": "main"
                }
        }
}
```

## Using as a library

```shell
go get -u github.com/dennwc/gosd/meta
```

```go
import (
	_ "github.com/dennwc/gosd/meta/all"
	
	"github.com/dennwc/gosd/meta"
)

// ...

m, err := meta.ReadImage(filename, f)
```

## License

MIT