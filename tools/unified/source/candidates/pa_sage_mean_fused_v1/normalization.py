"""FP32 first-order mean normalization; leave DGL sum/autograd untouched."""
import torch
from torch.autograd.function import once_differentiable
import DiGiTMeanNormalizeCUDA as native


def validate(x, degree, maximum):
    if (x.device.type != 'cuda' or x.dtype != torch.float32 or x.ndim != 2
            or degree.device != x.device or degree.ndim != 1
            or degree.dtype not in (torch.int32, torch.int64)
            or degree.numel() != x.shape[0] or not isinstance(maximum, int)
            or maximum < 1):
        raise ValueError('Expected CUDA FP32 matrix and matching integer degree vector')
    if degree.requires_grad:
        raise ValueError('Integer degrees cannot require gradients')


def divide(x, degree, maximum, backward=False):
    validate(x, degree, maximum)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    kernel = native.backward if backward else native.forward
    with torch.cuda.device(x.device):
        kernel(x.data_ptr(), degree.data_ptr(), out.data_ptr(), x.shape[0], x.shape[1],
               x.stride(0), x.stride(1), degree.stride(0), maximum,
               32 if degree.dtype == torch.int32 else 64,
               torch.cuda.current_stream(x.device).cuda_stream)
    return out


class MeanNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, summed, degree, maximum):
        ctx.save_for_backward(degree)
        ctx.maximum = maximum
        return divide(summed, degree, maximum)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (degree,) = ctx.saved_tensors
        return divide(grad_output, degree, ctx.maximum, backward=True), None, None


def fused_mean(block, features):
    import dgl.ops
    summed = dgl.ops.copy_u_sum(block, features)
    return MeanNormalize.apply(summed, block.in_degrees(), max(block.num_edges(), 1))
