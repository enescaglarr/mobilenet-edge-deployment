"""PyTorch integration for the custom MobileNetV2 CUDA kernels.

- load_ext()            : JIT-compiles kernels/mnv2_kernels.cu (needs nvcc + ninja)
- DepthwiseConv3x3      : nn.Module, impl in {"naive", "tiled", "cudnn"}
- PointwiseConv1x1      : nn.Module, impl in {"naive", "tiled", "cudnn"}
- convert_mobilenet_v2  : folds BatchNorm into convs and swaps dw/pw layers
- set_impl              : switches every custom layer's backend at runtime
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

_EXT = None
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
IMPLS = ("naive", "tiled", "cudnn")


def load_ext(verbose=False):
    """Compile (once) and return the extension module."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(
            name="mnv2_kernels",
            sources=[os.path.join(KERNEL_DIR, "mnv2_kernels.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=verbose,
        )
    return _EXT


class DepthwiseConv3x3(nn.Module):
    """Depthwise 3x3, padding 1, stride 1|2, optional fused bias + ReLU6."""

    def __init__(self, weight, bias, stride, relu6, impl="tiled"):
        super().__init__()
        C = weight.shape[0]
        assert weight.shape == (C, 1, 3, 3), weight.shape
        self.C, self.stride, self.relu6, self.impl = C, stride, relu6, impl
        # cuDNN path wants [C,1,3,3] half; custom path wants [C,9] half
        self.register_buffer("w4", weight.detach().clone().half())
        self.register_buffer("w2", weight.detach().reshape(C, 9).clone().half())
        if bias is not None:
            self.register_buffer("b32", bias.detach().clone().float())
            self.register_buffer("b16", bias.detach().clone().half())
        else:
            self.b32 = None
            self.b16 = None

    def forward(self, x):
        if self.impl == "cudnn":
            y = F.conv2d(x, self.w4, self.b16, stride=self.stride, padding=1, groups=self.C)
            return F.relu6(y) if self.relu6 else y
        b = self.b32 if self.b32 is not None else torch.empty(0, device=x.device)
        return load_ext().dw3x3_forward(x.contiguous(), self.w2, b,
                                        self.stride, self.relu6, self.impl == "tiled")

    def extra_repr(self):
        return f"C={self.C}, stride={self.stride}, relu6={self.relu6}, impl={self.impl}"


class PointwiseConv1x1(nn.Module):
    """1x1 conv, stride 1, groups 1, optional fused bias + ReLU6."""

    def __init__(self, weight, bias, relu6, impl="tiled"):
        super().__init__()
        Co, Ci = weight.shape[0], weight.shape[1]
        assert weight.shape == (Co, Ci, 1, 1), weight.shape
        self.Ci, self.Co, self.relu6, self.impl = Ci, Co, relu6, impl
        self.register_buffer("w4", weight.detach().clone().half())
        self.register_buffer("w2", weight.detach().reshape(Co, Ci).clone().half())
        if bias is not None:
            self.register_buffer("b32", bias.detach().clone().float())
            self.register_buffer("b16", bias.detach().clone().half())
        else:
            self.b32 = None
            self.b16 = None

    def forward(self, x):
        if self.impl == "cudnn":
            y = F.conv2d(x, self.w4, self.b16)
            return F.relu6(y) if self.relu6 else y
        b = self.b32 if self.b32 is not None else torch.empty(0, device=x.device)
        return load_ext().pw1x1_forward(x.contiguous(), self.w2, b,
                                        self.relu6, self.impl == "tiled")

    def extra_repr(self):
        return f"Ci={self.Ci}, Co={self.Co}, relu6={self.relu6}, impl={self.impl}"


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------
def _make_layer(conv: nn.Conv2d, relu6: bool, impl: str):
    k = conv.kernel_size
    if (k == (3, 3) and conv.groups == conv.in_channels == conv.out_channels
            and conv.padding == (1, 1) and conv.stride[0] in (1, 2)):
        return DepthwiseConv3x3(conv.weight, conv.bias, conv.stride[0], relu6, impl)
    if k == (1, 1) and conv.groups == 1 and conv.stride == (1, 1):
        return PointwiseConv1x1(conv.weight, conv.bias, relu6, impl)
    # anything else (the stem 3x3 conv): keep fused conv + activation
    return nn.Sequential(conv, nn.ReLU6(inplace=True)) if relu6 else conv


def _convert(module: nn.Module, impl: str) -> nn.Module:
    if isinstance(module, nn.Sequential):
        items, out, i = list(module.children()), [], 0
        while i < len(items):
            m = items[i]
            if isinstance(m, nn.Conv2d) and i + 1 < len(items) and isinstance(items[i + 1], nn.BatchNorm2d):
                fused = fuse_conv_bn_eval(m, items[i + 1])
                i += 2
                relu6 = i < len(items) and isinstance(items[i], nn.ReLU6)
                if relu6:
                    i += 1
                out.append(_make_layer(fused, relu6, impl))
            else:
                out.append(_convert(m, impl))
                i += 1
        return nn.Sequential(*out)
    for name, child in module.named_children():
        setattr(module, name, _convert(child, impl))
    return module


def convert_mobilenet_v2(model: nn.Module, impl: str = "tiled") -> nn.Module:
    """Return an FP16 copy of torchvision mobilenet_v2 with BN folded and
    depthwise/pointwise convs replaced by custom layers."""
    import copy
    model = copy.deepcopy(model).eval()
    model = _convert(model, impl)
    return model.half().eval()


def set_impl(model: nn.Module, dw: str = None, pw: str = None):
    for m in model.modules():
        if isinstance(m, DepthwiseConv3x3) and dw:
            m.impl = dw
        if isinstance(m, PointwiseConv1x1) and pw:
            m.impl = pw
    return model


def custom_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (DepthwiseConv3x3, PointwiseConv1x1))]
