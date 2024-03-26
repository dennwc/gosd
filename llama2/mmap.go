package llama2

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/edsrzf/mmap-go"
)

func Memmap(path string) ([]byte, func(), error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	mmap, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	unmap := func() {
		mmap.Unmap()
		f.Close()
	}
	return mmap, unmap, nil
}

func AsSlice[D comparable, S comparable](src []S) ([]D, error) {
	var (
		szero S
		dzero D
	)
	srcSize := int(unsafe.Sizeof(szero))
	dstSize := int(unsafe.Sizeof(dzero))
	fullSize := srcSize * len(src)
	if fullSize%dstSize != 0 {
		return nil, fmt.Errorf("%d is not a multiple of %d bytes", fullSize, dstSize)
	}
	ptr := unsafe.Pointer(&src[0])
	return unsafe.Slice((*D)(ptr), fullSize/dstSize), nil
}
