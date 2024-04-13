package protocol

//go:generate protoc --go_opt=paths=source_relative --twirp_opt=paths=source_relative --go_out=. --twirp_out=. stable_diffusion.proto
