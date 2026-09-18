from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor left, torch::Tensor right);

torch::Tensor vector_add(torch::Tensor left, torch::Tensor right) {
    TORCH_CHECK(left.is_cuda(), "left must be a CUDA tensor");
    TORCH_CHECK(right.is_cuda(), "right must be a CUDA tensor");
    TORCH_CHECK(left.sizes() == right.sizes(), "input shapes must match");
    TORCH_CHECK(left.scalar_type() == torch::kFloat32, "left must be float32");
    TORCH_CHECK(right.scalar_type() == torch::kFloat32, "right must be float32");
    return vector_add_cuda(left.contiguous(), right.contiguous());
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void vector_add_kernel(
    const float* left,
    const float* right,
    float* output,
    int64_t element_count
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = left[index] + right[index];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor left, torch::Tensor right) {
    auto output = torch::empty_like(left);
    const int64_t element_count = left.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((element_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    vector_add_kernel<<<blocks, threads, 0, stream>>>(
        left.data_ptr<float>(),
        right.data_ptr<float>(),
        output.data_ptr<float>(),
        element_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""


def main() -> None:
    os.environ.setdefault("MAX_JOBS", "2")
    build_directory = Path.home() / ".cache" / "torch_extensions" / "vector_add_smoke"
    build_directory.mkdir(parents=True, exist_ok=True)
    extension = load_inline(
        name="miniscale_vector_add_extension",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["vector_add"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        build_directory=str(build_directory),
        verbose=True,
    )

    left = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    right = torch.randn_like(left)
    expected = left + right
    actual = extension.vector_add(left, right)
    torch.cuda.synchronize()
    maximum_error = float((expected - actual).abs().max())
    print({"elements": left.numel(), "maximum_error": maximum_error})
    if maximum_error != 0.0:
        raise RuntimeError(f"extension result mismatch: {maximum_error}")


if __name__ == "__main__":
    main()
