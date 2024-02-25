#ifndef CGO_SD_CPP
#define CGO_SD_CPP

#include "stable-diffusion.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

SD_API struct ggml_tensor* go_sd_sample(
   sd_ctx_t* sd_ctx, struct ggml_context* gctx,
   struct ggml_tensor* xt, struct ggml_tensor* noise,
   struct ggml_tensor* c, struct ggml_tensor* cvec,
   struct ggml_tensor* uc, struct ggml_tensor* ucvec,
   struct ggml_tensor* imageHint,
   int64_t seed, enum sample_method_t method, int steps, float cfgScale, float controlStrength
);

SD_API void go_sd_maybeFreeCond(sd_ctx_t* sd_ctx);
SD_API void go_sd_maybeFreeDiff(sd_ctx_t* sd_ctx);
SD_API void go_sd_maybeFreeFirst(sd_ctx_t* sd_ctx);
SD_API struct ggml_tensor* go_sd_decodeFirstStage(sd_ctx_t* sd_ctx, struct ggml_context* gctx, struct ggml_tensor* t);
SD_API void go_sd_getLearnedCondition(
    struct ggml_tensor** outC, struct ggml_tensor** outVec,
    sd_ctx_t* sd_ctx, struct ggml_context* gctx,
    const char* promptPtr, int clipSkip, int width, int height
);
SD_API void go_sd_getLearnedConditionNeg(
    struct ggml_tensor** outC, struct ggml_tensor** outVec,
    sd_ctx_t* sd_ctx, struct ggml_context* gctx,
    const char* promptPtr, int clipSkip, int width, int height
);

#ifdef __cplusplus
}
#endif

#endif // CGO_SD_CPP