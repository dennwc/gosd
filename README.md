# Stable Diffusion Toolbox for Go
[![Go Reference](https://pkg.go.dev/badge/github.com/dennwc/gosd.svg)](https://pkg.go.dev/github.com/dennwc/gosd)

This repository contains Go libraries and tools for working with Stable Diffusion:
- [gosd](./cmd/gosd) - tool/library wrapper for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), allowing to run SD on CPU.
- [sdinfo](./cmd/sdinfo) - tool/library for extracting prompt metadata from SD images (Automatic1111 and Invoke).
- [runtime/auto](https://pkg.go.dev/github.com/dennwc/gosd/runtime/auto) - HTTP client for [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- [ggml](https://pkg.go.dev/github.com/dennwc/gosd/ggml) - work-in-progress bindings for [GGML](https://github.com/ggerganov/ggml).

## License

MIT