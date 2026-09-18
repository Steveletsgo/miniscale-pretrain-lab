from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    left_pointer,
    right_pointer,
    output_pointer,
    element_count: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    left = tl.load(left_pointer + offsets, mask=mask)
    right = tl.load(right_pointer + offsets, mask=mask)
    tl.store(output_pointer + offsets, left + right, mask=mask)


def triton_vector_add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not left.is_cuda or not right.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if left.shape != right.shape:
        raise ValueError("input shapes must match")
    output = torch.empty_like(left)
    element_count = output.numel()
    grid = (triton.cdiv(element_count, 256),)
    vector_add_kernel[grid](left, right, output, element_count, block_size=256)
    return output


def main() -> None:
    element_count = 1 << 24
    left = torch.randn(element_count, device="cuda", dtype=torch.float32)
    right = torch.randn_like(left)
    expected = left + right
    actual = triton_vector_add(left, right)
    torch.cuda.synchronize()
    maximum_error = float((expected - actual).abs().max())

    triton_ms = triton.testing.do_bench(lambda: triton_vector_add(left, right))
    torch_ms = triton.testing.do_bench(lambda: left + right)
    transferred_bytes = 3 * element_count * left.element_size()
    triton_gbps = transferred_bytes / (triton_ms * 1e-3) / 1e9
    torch_gbps = transferred_bytes / (torch_ms * 1e-3) / 1e9
    print(
        {
            "elements": element_count,
            "maximum_error": maximum_error,
            "triton_ms": round(triton_ms, 4),
            "torch_ms": round(torch_ms, 4),
            "triton_gbps": round(triton_gbps, 2),
            "torch_gbps": round(torch_gbps, 2),
        }
    )
    if maximum_error != 0.0:
        raise RuntimeError(f"Triton result mismatch: {maximum_error}")


if __name__ == "__main__":
    main()
