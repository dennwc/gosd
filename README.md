# Stable Diffusion for Go
[![Go Reference](https://pkg.go.dev/badge/github.com/dennwc/gosd.svg)](https://pkg.go.dev/github.com/dennwc/gosd)

This is a Go wrapper for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

It embeds all required C/C++ files into the project and compiles it statically, so there's no need for additional libraries.

Only tested on Linux x64.

## Using as a CLI

```shell
go install github.com/dennwc/gosd/cmd/gosd@latest
gosd -m ./path/to/model.safetensors -W 256 -H 256 --steps 20 -p "an image of a cat" -o output.png
```

## Using as a library

```shell
go get -u github.com/dennwc/gosd
```

See [cmd/gosd](./cmd/gosd/main.go) for an example.

## License

MIT