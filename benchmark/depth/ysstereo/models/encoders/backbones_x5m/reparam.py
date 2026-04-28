import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OnlineReParamBlock"]


def pad_kernel(kernel, target_kernel_hw):
    h_to_pad = (target_kernel_hw - kernel.size(2)) // 2
    w_to_pad = (target_kernel_hw - kernel.size(3)) // 2
    return F.pad(kernel, [w_to_pad, w_to_pad, h_to_pad, h_to_pad], value=0)


def conv_fuse_bn(conv, bn):
    if isinstance(conv, nn.Conv2d):
        kernel = conv.weight
    else:
        assert isinstance(conv, (torch.Tensor, nn.Parameter))
        kernel = conv
    assert isinstance(bn, (nn.BatchNorm2d, nn.SyncBatchNorm)), type(bn)

    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    new_w = kernel * ((gamma / std).reshape(-1, 1, 1, 1))
    new_b = bn.bias - bn.running_mean * gamma / std
    return new_w, new_b


class OnlineReParamBlock(nn.Module):
    """A Online Re-parameter Convolution Block.

    For more details, please see https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3306 # noqa

    Args:
        in_channels : Same as nn.Conv2d.
        out_channels : Same as nn.Conv2d.
        kernel_args : A dict of `kernel_size: repeat_num` pair.
            kernel must be odd. eg: {5:3, 3:2, 1:1}.
        stride : Same as nn.Conv2d. Defaults to 1.
        groups : Same as nn.Conv2d.. Defaults to 1.
        dilation : Same as nn.Conv2d.. Defaults to 1.
        norm_layer: Activation layer. Defaults to None.
        act_layer : Normalization layer. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_args: Dict[int, int],
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        **kwargs,
    ) -> None:
        super().__init__()

        assert all(
            map(lambda i: i % 2 == 1, kernel_args.keys())
        ), "All kernel size must be odd number."
        if norm_layer:
            assert isinstance(
                norm_layer, (nn.BatchNorm2d, nn.SyncBatchNorm)
            ), f"{type(norm_layer)} is not supported."

        self.max_kernel = max(kernel_args.keys())
        kernel_args = sorted(
            kernel_args.items(), key=lambda i: i[0], reverse=True
        )

        if stride == 1:
            self.padding = math.ceil((self.max_kernel - 1) // 2)
        else:
            self.padding = math.ceil(self.max_kernel // stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_args = kernel_args
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        self.fused = False
        self.identity_tensor = None
        self.merge_conv = None

        self.build_block()

    def get_bn_kernel_bias(self, bn):
        if bn is None:
            return 0.0, 0.0
        assert isinstance(bn, (nn.BatchNorm2d, nn.SyncBatchNorm)), type(bn)

        if self.identity_tensor is None:
            input_dim = self.in_channels // self.groups
            kernel = torch.zeros(
                (
                    self.in_channels,
                    input_dim,
                    self.max_kernel,
                    self.max_kernel,
                ),
                dtype=torch.float32,
            ).to(bn.weight.device)
            mid = self.max_kernel // 2
            for i in range(self.in_channels):
                kernel[i, i % input_dim, mid, mid] = 1.0
            self.identity_tensor = kernel
        else:
            kernel = self.identity_tensor

        std = (bn.running_var + bn.eps).sqrt()
        kernel = (bn.weight / std).reshape(-1, 1, 1, 1) * kernel
        bias = bn.bias - bn.running_mean * bn.weight / std

        return kernel, bias

    def build_block(self):
        if self.in_channels == self.out_channels and self.stride == 1:
            self.bn_side = nn.BatchNorm2d(self.in_channels)
        else:
            self.bn_side = None

        icg = int(self.in_channels / self.groups)

        for k_size, k_num in self.kernel_args:
            _w = nn.ParameterList()
            _bn = nn.ModuleList()
            for _ in range(k_num):
                param = nn.Parameter(
                    torch.Tensor(self.out_channels, icg, k_size, k_size)
                )
                nn.init.normal_(param, mean=0.0, std=0.01)
                bn = nn.BatchNorm2d(self.out_channels)
                _w.append(param)
                _bn.append(bn)
            setattr(self, f"w_{k_size}", _w)
            setattr(self, f"bn_{k_size}", _bn)

    def get_weight_bias(self):
        weight, bias = 0.0, 0.0

        for k_size, k_num in self.kernel_args:
            w_list = getattr(self, f"w_{k_size}")
            bn_list = getattr(self, f"bn_{k_size}")
            for i in range(k_num):
                _w, _b = conv_fuse_bn(w_list[i], bn_list[i])
                weight += pad_kernel(_w, self.max_kernel)
                bias += _b
        w_bn, b_bn = self.get_bn_kernel_bias(self.bn_side)

        weight += w_bn
        bias += b_bn
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_conv:
            out = self.merge_conv(x)
            if self.fused:
                return out
            else:
                if self.norm_layer:
                    out = self.norm_layer(out)
                if self.act_layer:
                    out = self.act_layer(out)
                return out
        else:
            out = self.float_forward(x)
        return out

    def float_forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.get_weight_bias()
        out = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm_layer:
            out = self.norm_layer(out)
        if self.act_layer:
            out = self.act_layer(out)
        return out

    def create_merge_conv(self):
        if self.merge_conv:
            return
        weight, bias = self.get_weight_bias()
        merge_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.max_kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        merge_conv.weight.data = weight
        merge_conv.bias.data = bias
        self.merge_conv = merge_conv

    def _clean(self):
        for k_size, k_num in self.kernel_args:
            w_list = getattr(self, f"w_{k_size}")
            for i in range(k_num):
                w_list[i].detach_()
            self.__delattr__(f"w_{k_size}")
            self.__delattr__(f"bn_{k_size}")
        self.__delattr__("bn_side")
        self.__delattr__("norm_layer")
        self.__delattr__("act_layer")
