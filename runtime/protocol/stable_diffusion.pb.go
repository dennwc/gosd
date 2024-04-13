// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.33.0
// 	protoc        v4.23.4
// source: stable_diffusion.proto

package protocol

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ModelInfo_Kind int32

const (
	ModelInfo_UNSPECIFIED ModelInfo_Kind = 0
	ModelInfo_SD15        ModelInfo_Kind = 1
	ModelInfo_SDXL10      ModelInfo_Kind = 2
)

// Enum value maps for ModelInfo_Kind.
var (
	ModelInfo_Kind_name = map[int32]string{
		0: "UNSPECIFIED",
		1: "SD15",
		2: "SDXL10",
	}
	ModelInfo_Kind_value = map[string]int32{
		"UNSPECIFIED": 0,
		"SD15":        1,
		"SDXL10":      2,
	}
)

func (x ModelInfo_Kind) Enum() *ModelInfo_Kind {
	p := new(ModelInfo_Kind)
	*p = x
	return p
}

func (x ModelInfo_Kind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelInfo_Kind) Descriptor() protoreflect.EnumDescriptor {
	return file_stable_diffusion_proto_enumTypes[0].Descriptor()
}

func (ModelInfo_Kind) Type() protoreflect.EnumType {
	return &file_stable_diffusion_proto_enumTypes[0]
}

func (x ModelInfo_Kind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelInfo_Kind.Descriptor instead.
func (ModelInfo_Kind) EnumDescriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{3, 0}
}

type SamplerInfo struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id   string `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`
	Name string `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
}

func (x *SamplerInfo) Reset() {
	*x = SamplerInfo{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SamplerInfo) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SamplerInfo) ProtoMessage() {}

func (x *SamplerInfo) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SamplerInfo.ProtoReflect.Descriptor instead.
func (*SamplerInfo) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{0}
}

func (x *SamplerInfo) GetId() string {
	if x != nil {
		return x.Id
	}
	return ""
}

func (x *SamplerInfo) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

type ListSamplersReq struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *ListSamplersReq) Reset() {
	*x = ListSamplersReq{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ListSamplersReq) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ListSamplersReq) ProtoMessage() {}

func (x *ListSamplersReq) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ListSamplersReq.ProtoReflect.Descriptor instead.
func (*ListSamplersReq) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{1}
}

type ListSamplersResp struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Samplers []*SamplerInfo `protobuf:"bytes,1,rep,name=samplers,proto3" json:"samplers,omitempty"`
}

func (x *ListSamplersResp) Reset() {
	*x = ListSamplersResp{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ListSamplersResp) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ListSamplersResp) ProtoMessage() {}

func (x *ListSamplersResp) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ListSamplersResp.ProtoReflect.Descriptor instead.
func (*ListSamplersResp) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{2}
}

func (x *ListSamplersResp) GetSamplers() []*SamplerInfo {
	if x != nil {
		return x.Samplers
	}
	return nil
}

type ModelInfo struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id     string         `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`         // backend-specific unique identifier
	Name   string         `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`     // name of the model (usually a base file name)
	Title  string         `protobuf:"bytes,3,opt,name=title,proto3" json:"title,omitempty"`   // human-readable name
	File   string         `protobuf:"bytes,4,opt,name=file,proto3" json:"file,omitempty"`     // file path to the model
	Sha256 string         `protobuf:"bytes,5,opt,name=sha256,proto3" json:"sha256,omitempty"` // SHA256 hash of the file
	Kind   ModelInfo_Kind `protobuf:"varint,6,opt,name=kind,proto3,enum=protocol.ModelInfo_Kind" json:"kind,omitempty"`
}

func (x *ModelInfo) Reset() {
	*x = ModelInfo{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ModelInfo) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelInfo) ProtoMessage() {}

func (x *ModelInfo) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelInfo.ProtoReflect.Descriptor instead.
func (*ModelInfo) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{3}
}

func (x *ModelInfo) GetId() string {
	if x != nil {
		return x.Id
	}
	return ""
}

func (x *ModelInfo) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ModelInfo) GetTitle() string {
	if x != nil {
		return x.Title
	}
	return ""
}

func (x *ModelInfo) GetFile() string {
	if x != nil {
		return x.File
	}
	return ""
}

func (x *ModelInfo) GetSha256() string {
	if x != nil {
		return x.Sha256
	}
	return ""
}

func (x *ModelInfo) GetKind() ModelInfo_Kind {
	if x != nil {
		return x.Kind
	}
	return ModelInfo_UNSPECIFIED
}

type ListModelsReq struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *ListModelsReq) Reset() {
	*x = ListModelsReq{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ListModelsReq) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ListModelsReq) ProtoMessage() {}

func (x *ListModelsReq) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ListModelsReq.ProtoReflect.Descriptor instead.
func (*ListModelsReq) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{4}
}

type ListModelsResp struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Models []*ModelInfo `protobuf:"bytes,1,rep,name=models,proto3" json:"models,omitempty"`
}

func (x *ListModelsResp) Reset() {
	*x = ListModelsResp{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ListModelsResp) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ListModelsResp) ProtoMessage() {}

func (x *ListModelsResp) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ListModelsResp.ProtoReflect.Descriptor instead.
func (*ListModelsResp) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{5}
}

func (x *ListModelsResp) GetModels() []*ModelInfo {
	if x != nil {
		return x.Models
	}
	return nil
}

type Image struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Width  uint32 `protobuf:"varint,1,opt,name=width,proto3" json:"width,omitempty"`
	Height uint32 `protobuf:"varint,2,opt,name=height,proto3" json:"height,omitempty"`
	// Types that are assignable to Image:
	//
	//	*Image_Rgb8
	//	*Image_Rgba8
	//	*Image_Png
	//	*Image_Jpeg
	Image isImage_Image `protobuf_oneof:"image"`
}

func (x *Image) Reset() {
	*x = Image{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Image) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Image) ProtoMessage() {}

func (x *Image) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Image.ProtoReflect.Descriptor instead.
func (*Image) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{6}
}

func (x *Image) GetWidth() uint32 {
	if x != nil {
		return x.Width
	}
	return 0
}

func (x *Image) GetHeight() uint32 {
	if x != nil {
		return x.Height
	}
	return 0
}

func (m *Image) GetImage() isImage_Image {
	if m != nil {
		return m.Image
	}
	return nil
}

func (x *Image) GetRgb8() []byte {
	if x, ok := x.GetImage().(*Image_Rgb8); ok {
		return x.Rgb8
	}
	return nil
}

func (x *Image) GetRgba8() []byte {
	if x, ok := x.GetImage().(*Image_Rgba8); ok {
		return x.Rgba8
	}
	return nil
}

func (x *Image) GetPng() []byte {
	if x, ok := x.GetImage().(*Image_Png); ok {
		return x.Png
	}
	return nil
}

func (x *Image) GetJpeg() []byte {
	if x, ok := x.GetImage().(*Image_Jpeg); ok {
		return x.Jpeg
	}
	return nil
}

type isImage_Image interface {
	isImage_Image()
}

type Image_Rgb8 struct {
	Rgb8 []byte `protobuf:"bytes,3,opt,name=rgb8,proto3,oneof"` // 8 bit per channel RGB
}

type Image_Rgba8 struct {
	Rgba8 []byte `protobuf:"bytes,4,opt,name=rgba8,proto3,oneof"` // 8 bit per channel RGBA
}

type Image_Png struct {
	Png []byte `protobuf:"bytes,5,opt,name=png,proto3,oneof"` // PNG bytes
}

type Image_Jpeg struct {
	Jpeg []byte `protobuf:"bytes,6,opt,name=jpeg,proto3,oneof"` // JPEG bytes
}

func (*Image_Rgb8) isImage_Image() {}

func (*Image_Rgba8) isImage_Image() {}

func (*Image_Png) isImage_Image() {}

func (*Image_Jpeg) isImage_Image() {}

type TextToImageReq struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	ModelId        string  `protobuf:"bytes,1,opt,name=model_id,json=modelId,proto3" json:"model_id,omitempty"`
	PositivePrompt string  `protobuf:"bytes,2,opt,name=positive_prompt,json=positivePrompt,proto3" json:"positive_prompt,omitempty"`
	NegativePrompt string  `protobuf:"bytes,3,opt,name=negative_prompt,json=negativePrompt,proto3" json:"negative_prompt,omitempty"`
	CfgScale       float64 `protobuf:"fixed64,4,opt,name=cfg_scale,json=cfgScale,proto3" json:"cfg_scale,omitempty"`
	RngSeed        int64   `protobuf:"varint,5,opt,name=rng_seed,json=rngSeed,proto3" json:"rng_seed,omitempty"`
	SamplerId      string  `protobuf:"bytes,6,opt,name=sampler_id,json=samplerId,proto3" json:"sampler_id,omitempty"`
	SamplerSteps   uint32  `protobuf:"varint,7,opt,name=sampler_steps,json=samplerSteps,proto3" json:"sampler_steps,omitempty"`
	ImageWidth     uint32  `protobuf:"varint,8,opt,name=image_width,json=imageWidth,proto3" json:"image_width,omitempty"`
	ImageHeight    uint32  `protobuf:"varint,9,opt,name=image_height,json=imageHeight,proto3" json:"image_height,omitempty"`
	ClipSkip       uint32  `protobuf:"varint,10,opt,name=clip_skip,json=clipSkip,proto3" json:"clip_skip,omitempty"`
}

func (x *TextToImageReq) Reset() {
	*x = TextToImageReq{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *TextToImageReq) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TextToImageReq) ProtoMessage() {}

func (x *TextToImageReq) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TextToImageReq.ProtoReflect.Descriptor instead.
func (*TextToImageReq) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{7}
}

func (x *TextToImageReq) GetModelId() string {
	if x != nil {
		return x.ModelId
	}
	return ""
}

func (x *TextToImageReq) GetPositivePrompt() string {
	if x != nil {
		return x.PositivePrompt
	}
	return ""
}

func (x *TextToImageReq) GetNegativePrompt() string {
	if x != nil {
		return x.NegativePrompt
	}
	return ""
}

func (x *TextToImageReq) GetCfgScale() float64 {
	if x != nil {
		return x.CfgScale
	}
	return 0
}

func (x *TextToImageReq) GetRngSeed() int64 {
	if x != nil {
		return x.RngSeed
	}
	return 0
}

func (x *TextToImageReq) GetSamplerId() string {
	if x != nil {
		return x.SamplerId
	}
	return ""
}

func (x *TextToImageReq) GetSamplerSteps() uint32 {
	if x != nil {
		return x.SamplerSteps
	}
	return 0
}

func (x *TextToImageReq) GetImageWidth() uint32 {
	if x != nil {
		return x.ImageWidth
	}
	return 0
}

func (x *TextToImageReq) GetImageHeight() uint32 {
	if x != nil {
		return x.ImageHeight
	}
	return 0
}

func (x *TextToImageReq) GetClipSkip() uint32 {
	if x != nil {
		return x.ClipSkip
	}
	return 0
}

type TextToImageResp struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Image   *Image `protobuf:"bytes,1,opt,name=image,proto3" json:"image,omitempty"`
	RngSeed int64  `protobuf:"varint,2,opt,name=rng_seed,json=rngSeed,proto3" json:"rng_seed,omitempty"`
}

func (x *TextToImageResp) Reset() {
	*x = TextToImageResp{}
	if protoimpl.UnsafeEnabled {
		mi := &file_stable_diffusion_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *TextToImageResp) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TextToImageResp) ProtoMessage() {}

func (x *TextToImageResp) ProtoReflect() protoreflect.Message {
	mi := &file_stable_diffusion_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TextToImageResp.ProtoReflect.Descriptor instead.
func (*TextToImageResp) Descriptor() ([]byte, []int) {
	return file_stable_diffusion_proto_rawDescGZIP(), []int{8}
}

func (x *TextToImageResp) GetImage() *Image {
	if x != nil {
		return x.Image
	}
	return nil
}

func (x *TextToImageResp) GetRngSeed() int64 {
	if x != nil {
		return x.RngSeed
	}
	return 0
}

var File_stable_diffusion_proto protoreflect.FileDescriptor

var file_stable_diffusion_proto_rawDesc = []byte{
	0x0a, 0x16, 0x73, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x5f, 0x64, 0x69, 0x66, 0x66, 0x75, 0x73, 0x69,
	0x6f, 0x6e, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x08, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63,
	0x6f, 0x6c, 0x22, 0x31, 0x0a, 0x0b, 0x53, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x49, 0x6e, 0x66,
	0x6f, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x02, 0x69,
	0x64, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x04, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x11, 0x0a, 0x0f, 0x4c, 0x69, 0x73, 0x74, 0x53, 0x61, 0x6d,
	0x70, 0x6c, 0x65, 0x72, 0x73, 0x52, 0x65, 0x71, 0x22, 0x45, 0x0a, 0x10, 0x4c, 0x69, 0x73, 0x74,
	0x53, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x73, 0x52, 0x65, 0x73, 0x70, 0x12, 0x31, 0x0a, 0x08,
	0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x15,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x53, 0x61, 0x6d, 0x70, 0x6c, 0x65,
	0x72, 0x49, 0x6e, 0x66, 0x6f, 0x52, 0x08, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x73, 0x22,
	0xce, 0x01, 0x0a, 0x09, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x49, 0x6e, 0x66, 0x6f, 0x12, 0x0e, 0x0a,
	0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x02, 0x69, 0x64, 0x12, 0x12, 0x0a,
	0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x12, 0x14, 0x0a, 0x05, 0x74, 0x69, 0x74, 0x6c, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x05, 0x74, 0x69, 0x74, 0x6c, 0x65, 0x12, 0x12, 0x0a, 0x04, 0x66, 0x69, 0x6c, 0x65, 0x18,
	0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x66, 0x69, 0x6c, 0x65, 0x12, 0x16, 0x0a, 0x06, 0x73,
	0x68, 0x61, 0x32, 0x35, 0x36, 0x18, 0x05, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x73, 0x68, 0x61,
	0x32, 0x35, 0x36, 0x12, 0x2c, 0x0a, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x18, 0x06, 0x20, 0x01, 0x28,
	0x0e, 0x32, 0x18, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x4d, 0x6f, 0x64,
	0x65, 0x6c, 0x49, 0x6e, 0x66, 0x6f, 0x2e, 0x4b, 0x69, 0x6e, 0x64, 0x52, 0x04, 0x6b, 0x69, 0x6e,
	0x64, 0x22, 0x2d, 0x0a, 0x04, 0x4b, 0x69, 0x6e, 0x64, 0x12, 0x0f, 0x0a, 0x0b, 0x55, 0x4e, 0x53,
	0x50, 0x45, 0x43, 0x49, 0x46, 0x49, 0x45, 0x44, 0x10, 0x00, 0x12, 0x08, 0x0a, 0x04, 0x53, 0x44,
	0x31, 0x35, 0x10, 0x01, 0x12, 0x0a, 0x0a, 0x06, 0x53, 0x44, 0x58, 0x4c, 0x31, 0x30, 0x10, 0x02,
	0x22, 0x0f, 0x0a, 0x0d, 0x4c, 0x69, 0x73, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x52, 0x65,
	0x71, 0x22, 0x3d, 0x0a, 0x0e, 0x4c, 0x69, 0x73, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x52,
	0x65, 0x73, 0x70, 0x12, 0x2b, 0x0a, 0x06, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x18, 0x01, 0x20,
	0x03, 0x28, 0x0b, 0x32, 0x13, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x4d,
	0x6f, 0x64, 0x65, 0x6c, 0x49, 0x6e, 0x66, 0x6f, 0x52, 0x06, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73,
	0x22, 0x96, 0x01, 0x0a, 0x05, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x12, 0x14, 0x0a, 0x05, 0x77, 0x69,
	0x64, 0x74, 0x68, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x05, 0x77, 0x69, 0x64, 0x74, 0x68,
	0x12, 0x16, 0x0a, 0x06, 0x68, 0x65, 0x69, 0x67, 0x68, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0d,
	0x52, 0x06, 0x68, 0x65, 0x69, 0x67, 0x68, 0x74, 0x12, 0x14, 0x0a, 0x04, 0x72, 0x67, 0x62, 0x38,
	0x18, 0x03, 0x20, 0x01, 0x28, 0x0c, 0x48, 0x00, 0x52, 0x04, 0x72, 0x67, 0x62, 0x38, 0x12, 0x16,
	0x0a, 0x05, 0x72, 0x67, 0x62, 0x61, 0x38, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0c, 0x48, 0x00, 0x52,
	0x05, 0x72, 0x67, 0x62, 0x61, 0x38, 0x12, 0x12, 0x0a, 0x03, 0x70, 0x6e, 0x67, 0x18, 0x05, 0x20,
	0x01, 0x28, 0x0c, 0x48, 0x00, 0x52, 0x03, 0x70, 0x6e, 0x67, 0x12, 0x14, 0x0a, 0x04, 0x6a, 0x70,
	0x65, 0x67, 0x18, 0x06, 0x20, 0x01, 0x28, 0x0c, 0x48, 0x00, 0x52, 0x04, 0x6a, 0x70, 0x65, 0x67,
	0x42, 0x07, 0x0a, 0x05, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x22, 0xda, 0x02, 0x0a, 0x0e, 0x54, 0x65,
	0x78, 0x74, 0x54, 0x6f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x52, 0x65, 0x71, 0x12, 0x19, 0x0a, 0x08,
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07,
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x49, 0x64, 0x12, 0x27, 0x0a, 0x0f, 0x70, 0x6f, 0x73, 0x69, 0x74,
	0x69, 0x76, 0x65, 0x5f, 0x70, 0x72, 0x6f, 0x6d, 0x70, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x0e, 0x70, 0x6f, 0x73, 0x69, 0x74, 0x69, 0x76, 0x65, 0x50, 0x72, 0x6f, 0x6d, 0x70, 0x74,
	0x12, 0x27, 0x0a, 0x0f, 0x6e, 0x65, 0x67, 0x61, 0x74, 0x69, 0x76, 0x65, 0x5f, 0x70, 0x72, 0x6f,
	0x6d, 0x70, 0x74, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0e, 0x6e, 0x65, 0x67, 0x61, 0x74,
	0x69, 0x76, 0x65, 0x50, 0x72, 0x6f, 0x6d, 0x70, 0x74, 0x12, 0x1b, 0x0a, 0x09, 0x63, 0x66, 0x67,
	0x5f, 0x73, 0x63, 0x61, 0x6c, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x01, 0x52, 0x08, 0x63, 0x66,
	0x67, 0x53, 0x63, 0x61, 0x6c, 0x65, 0x12, 0x19, 0x0a, 0x08, 0x72, 0x6e, 0x67, 0x5f, 0x73, 0x65,
	0x65, 0x64, 0x18, 0x05, 0x20, 0x01, 0x28, 0x03, 0x52, 0x07, 0x72, 0x6e, 0x67, 0x53, 0x65, 0x65,
	0x64, 0x12, 0x1d, 0x0a, 0x0a, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x5f, 0x69, 0x64, 0x18,
	0x06, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x49, 0x64,
	0x12, 0x23, 0x0a, 0x0d, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x5f, 0x73, 0x74, 0x65, 0x70,
	0x73, 0x18, 0x07, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x0c, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72,
	0x53, 0x74, 0x65, 0x70, 0x73, 0x12, 0x1f, 0x0a, 0x0b, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x5f, 0x77,
	0x69, 0x64, 0x74, 0x68, 0x18, 0x08, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x0a, 0x69, 0x6d, 0x61, 0x67,
	0x65, 0x57, 0x69, 0x64, 0x74, 0x68, 0x12, 0x21, 0x0a, 0x0c, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x5f,
	0x68, 0x65, 0x69, 0x67, 0x68, 0x74, 0x18, 0x09, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x0b, 0x69, 0x6d,
	0x61, 0x67, 0x65, 0x48, 0x65, 0x69, 0x67, 0x68, 0x74, 0x12, 0x1b, 0x0a, 0x09, 0x63, 0x6c, 0x69,
	0x70, 0x5f, 0x73, 0x6b, 0x69, 0x70, 0x18, 0x0a, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x08, 0x63, 0x6c,
	0x69, 0x70, 0x53, 0x6b, 0x69, 0x70, 0x22, 0x53, 0x0a, 0x0f, 0x54, 0x65, 0x78, 0x74, 0x54, 0x6f,
	0x49, 0x6d, 0x61, 0x67, 0x65, 0x52, 0x65, 0x73, 0x70, 0x12, 0x25, 0x0a, 0x05, 0x69, 0x6d, 0x61,
	0x67, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x0f, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x63, 0x6f, 0x6c, 0x2e, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x52, 0x05, 0x69, 0x6d, 0x61, 0x67, 0x65,
	0x12, 0x19, 0x0a, 0x08, 0x72, 0x6e, 0x67, 0x5f, 0x73, 0x65, 0x65, 0x64, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x03, 0x52, 0x07, 0x72, 0x6e, 0x67, 0x53, 0x65, 0x65, 0x64, 0x32, 0xdd, 0x01, 0x0a, 0x0f,
	0x53, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x44, 0x69, 0x66, 0x66, 0x75, 0x73, 0x69, 0x6f, 0x6e, 0x12,
	0x45, 0x0a, 0x0c, 0x4c, 0x69, 0x73, 0x74, 0x53, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x73, 0x12,
	0x19, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x4c, 0x69, 0x73, 0x74, 0x53,
	0x61, 0x6d, 0x70, 0x6c, 0x65, 0x72, 0x73, 0x52, 0x65, 0x71, 0x1a, 0x1a, 0x2e, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x4c, 0x69, 0x73, 0x74, 0x53, 0x61, 0x6d, 0x70, 0x6c, 0x65,
	0x72, 0x73, 0x52, 0x65, 0x73, 0x70, 0x12, 0x3f, 0x0a, 0x0a, 0x4c, 0x69, 0x73, 0x74, 0x4d, 0x6f,
	0x64, 0x65, 0x6c, 0x73, 0x12, 0x17, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e,
	0x4c, 0x69, 0x73, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x52, 0x65, 0x71, 0x1a, 0x18, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x4c, 0x69, 0x73, 0x74, 0x4d, 0x6f, 0x64,
	0x65, 0x6c, 0x73, 0x52, 0x65, 0x73, 0x70, 0x12, 0x42, 0x0a, 0x0b, 0x54, 0x65, 0x78, 0x74, 0x54,
	0x6f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x12, 0x18, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f,
	0x6c, 0x2e, 0x54, 0x65, 0x78, 0x74, 0x54, 0x6f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x52, 0x65, 0x71,
	0x1a, 0x19, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2e, 0x54, 0x65, 0x78, 0x74,
	0x54, 0x6f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x52, 0x65, 0x73, 0x70, 0x42, 0x29, 0x5a, 0x27, 0x67,
	0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x64, 0x65, 0x6e, 0x6e, 0x77, 0x63,
	0x2f, 0x67, 0x6f, 0x73, 0x64, 0x2f, 0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x2f, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_stable_diffusion_proto_rawDescOnce sync.Once
	file_stable_diffusion_proto_rawDescData = file_stable_diffusion_proto_rawDesc
)

func file_stable_diffusion_proto_rawDescGZIP() []byte {
	file_stable_diffusion_proto_rawDescOnce.Do(func() {
		file_stable_diffusion_proto_rawDescData = protoimpl.X.CompressGZIP(file_stable_diffusion_proto_rawDescData)
	})
	return file_stable_diffusion_proto_rawDescData
}

var file_stable_diffusion_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_stable_diffusion_proto_msgTypes = make([]protoimpl.MessageInfo, 9)
var file_stable_diffusion_proto_goTypes = []interface{}{
	(ModelInfo_Kind)(0),      // 0: protocol.ModelInfo.Kind
	(*SamplerInfo)(nil),      // 1: protocol.SamplerInfo
	(*ListSamplersReq)(nil),  // 2: protocol.ListSamplersReq
	(*ListSamplersResp)(nil), // 3: protocol.ListSamplersResp
	(*ModelInfo)(nil),        // 4: protocol.ModelInfo
	(*ListModelsReq)(nil),    // 5: protocol.ListModelsReq
	(*ListModelsResp)(nil),   // 6: protocol.ListModelsResp
	(*Image)(nil),            // 7: protocol.Image
	(*TextToImageReq)(nil),   // 8: protocol.TextToImageReq
	(*TextToImageResp)(nil),  // 9: protocol.TextToImageResp
}
var file_stable_diffusion_proto_depIdxs = []int32{
	1, // 0: protocol.ListSamplersResp.samplers:type_name -> protocol.SamplerInfo
	0, // 1: protocol.ModelInfo.kind:type_name -> protocol.ModelInfo.Kind
	4, // 2: protocol.ListModelsResp.models:type_name -> protocol.ModelInfo
	7, // 3: protocol.TextToImageResp.image:type_name -> protocol.Image
	2, // 4: protocol.StableDiffusion.ListSamplers:input_type -> protocol.ListSamplersReq
	5, // 5: protocol.StableDiffusion.ListModels:input_type -> protocol.ListModelsReq
	8, // 6: protocol.StableDiffusion.TextToImage:input_type -> protocol.TextToImageReq
	3, // 7: protocol.StableDiffusion.ListSamplers:output_type -> protocol.ListSamplersResp
	6, // 8: protocol.StableDiffusion.ListModels:output_type -> protocol.ListModelsResp
	9, // 9: protocol.StableDiffusion.TextToImage:output_type -> protocol.TextToImageResp
	7, // [7:10] is the sub-list for method output_type
	4, // [4:7] is the sub-list for method input_type
	4, // [4:4] is the sub-list for extension type_name
	4, // [4:4] is the sub-list for extension extendee
	0, // [0:4] is the sub-list for field type_name
}

func init() { file_stable_diffusion_proto_init() }
func file_stable_diffusion_proto_init() {
	if File_stable_diffusion_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_stable_diffusion_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SamplerInfo); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ListSamplersReq); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ListSamplersResp); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ModelInfo); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ListModelsReq); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ListModelsResp); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Image); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*TextToImageReq); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_stable_diffusion_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*TextToImageResp); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	file_stable_diffusion_proto_msgTypes[6].OneofWrappers = []interface{}{
		(*Image_Rgb8)(nil),
		(*Image_Rgba8)(nil),
		(*Image_Png)(nil),
		(*Image_Jpeg)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_stable_diffusion_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   9,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_stable_diffusion_proto_goTypes,
		DependencyIndexes: file_stable_diffusion_proto_depIdxs,
		EnumInfos:         file_stable_diffusion_proto_enumTypes,
		MessageInfos:      file_stable_diffusion_proto_msgTypes,
	}.Build()
	File_stable_diffusion_proto = out.File
	file_stable_diffusion_proto_rawDesc = nil
	file_stable_diffusion_proto_goTypes = nil
	file_stable_diffusion_proto_depIdxs = nil
}
