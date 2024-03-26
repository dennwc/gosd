# Machine Learning Toolbox for Go
[![Go Reference](https://pkg.go.dev/badge/github.com/dennwc/gosd.svg)](https://pkg.go.dev/github.com/dennwc/gosd)

This repository contains Go libraries and tools for working with LLMs and Stable Diffusion:
- [gosd](./cmd/gosd) - tool/library wrapper for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), allowing to run SD on CPU.
- [llama](./llama-cpp) - library wrapper for [LLama.cpp](https://github.com/ggerganov/llama.cpp), allowing to run LLAMA on CPU (fork of [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)).
- [sdinfo](./cmd/sdinfo) - tool/library for extracting prompt metadata from SD images (Automatic1111 and Invoke).
- [runtime/auto](https://pkg.go.dev/github.com/dennwc/gosd/runtime/auto) - HTTP client for [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- [ggml](https://pkg.go.dev/github.com/dennwc/gosd/ggml) - work-in-progress bindings for [GGML](https://github.com/ggerganov/ggml).

## License

MIT