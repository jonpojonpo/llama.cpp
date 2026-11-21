#include "common.cuh"

// fused Inkling 2 blocks, see GGML_OP_INKLING
void ggml_cuda_op_inkling(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_inkling_supported(const ggml_tensor * dst);
