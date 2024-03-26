# Pure Go LLAMA2 inference

This is a Go port of [karpathy/llama2.c](https://github.com/karpathy/llama2.c).

## Using as a CLI

```shell
# Download the model
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# Install GoLLAMA2
go install github.com/dennwc/gosd/cmd/gollama2@latest

# Run it!
gollama2 -z ./tokenizer.bin ./stories15M.bin
```

## Using as a library

```shell
go get -u github.com/dennwc/gosd/llama2
```

See [main.go](./main.go) for an example.

## License

MIT