package pnginfo

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"

	"github.com/dennwc/gosd/meta"
	"github.com/dennwc/gosd/runtime/types"
)

func init() {
	meta.RegisterImageByExt(".png", func(r io.Reader) (meta.ImageMetadata, error) {
		m, err := ReadInfo(r)
		if err != nil || m == nil {
			return nil, err
		}
		if len(m.Meta) == 0 {
			return nil, nil
		}
		return m.Meta[0], nil
	})
}

type MetadataFunc func(s string) (meta.ImageMetadata, error)

var textHandlers = make(map[string][]MetadataFunc)

func RegisterTextHandler(txtName string, fnc MetadataFunc) {
	textHandlers[txtName] = append(textHandlers[txtName], fnc)
}

const (
	header = "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"
)

type TextInfo struct {
	Name  string
	Value string
}

type Info struct {
	Meta   []meta.ImageMetadata
	Text   []TextInfo
	Chunks []Chunk
}

func (m *Info) TextToImage() *types.TextToImageParams {
	if len(m.Meta) == 0 {
		return nil
	}
	return m.Meta[0].TextToImage()
}

func ReadInfo(r io.Reader) (*Info, error) {
	if err := readHeader(r); err != nil {
		return nil, err
	}
	info := new(Info)
	var buf [8]byte
	for {
		// Size and type
		if _, err := io.ReadFull(r, buf[:8]); err == io.EOF {
			return info, nil
		} else if err != nil {
			return info, err
		}
		sz := binary.BigEndian.Uint32(buf[:4])
		typ := string(buf[4:8])
		cr := io.LimitReader(r, int64(sz))
		var c Chunk
		switch typ {
		//case "IHDR": // TODO
		case "IEND":
			if _, err := io.CopyN(io.Discard, r, int64(sz)+4); err != nil {
				return info, err
			}
			return info, nil
		case "IDAT":
			// ignore data
			if _, err := io.Copy(io.Discard, cr); err != nil {
				return info, err
			}
		case "tEXt":
			data := make([]byte, sz)
			if _, err := io.ReadFull(cr, data); err != nil {
				return nil, err
			}
			c = &RawChunk{
				Type: typ,
				Data: data,
			}
			var arr []string
			for len(data) > 0 {
				i := bytes.IndexByte(data, 0)
				if i < 0 {
					arr = append(arr, string(data))
					break
				}
				arr = append(arr, string(data[:i]))
				data = data[i+1:]
			}
			for i := 0; i < len(arr); i += 2 {
				name := arr[i]
				var val string
				if i+1 < len(arr) {
					val = arr[i+1]
				}
				info.Text = append(info.Text, TextInfo{
					Name:  name,
					Value: val,
				})
				for _, fnc := range textHandlers[name] {
					m, _ := fnc(val)
					if m != nil {
						info.Meta = append(info.Meta, m)
					}
				}
			}
		default:
			data := make([]byte, sz)
			if _, err := io.ReadFull(cr, data); err != nil {
				return nil, err
			}
			c = &RawChunk{
				Type: typ,
				Data: data,
			}
		}
		// CRC
		if _, err := io.ReadFull(r, buf[:4]); err != nil {
			return info, err
		}
		if c != nil {
			info.Chunks = append(info.Chunks, c)
		}
	}
}

func readHeader(r io.Reader) error {
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:8]); err != nil {
		return err
	}
	if string(buf[:8]) != header {
		return errors.New("not a png file")
	}
	return nil
}

type Chunk interface {
	ChunkType() string
	ChunkData() []byte
}

type RawChunk struct {
	Type string
	Data []byte
}

func (c *RawChunk) ChunkType() string {
	return c.Type
}

func (c *RawChunk) ChunkData() []byte {
	return c.Data
}
