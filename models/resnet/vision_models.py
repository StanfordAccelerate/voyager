from typing import Callable, List, Optional, Type
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# This file contains a modified version of the resnet_reference.py model.
# It records the input and output activations, as well as the weight vectors.

_export = True  # Record weights and store them for export
_buffer = {}  # Dictionary buffer for weights
_layer_num = 1  # Layer counter
_block_num = 0  # Block counter
_verbose = False  # Print debug info
_no_intermediates = False  # Generate no intermediate buffers, only entry and exit
_no_weights = False  # Generate no weight/bias data

def reset():
    global _buffer
    global _layer_num
    global _block_num
    _buffer = {}
    _layer_num = 1
    _block_num = 0


def arrange_data(name: str, x: Tensor):
    if _verbose:
        print(name, '\t', x.size())

    # We need to store in float64
    x = x.type(torch.float64)

    # Special case for (last) FC layer, as we need to zero-pad to multiple of 8
    if name == 'fc.weight':
        x = torch.cat((x, torch.zeros(8, 512)), dim=0)
    elif name == 'fc.bias':
        x = torch.cat((x, torch.zeros(8)), dim=0)
    elif name == 'fc.comp':
        x = torch.cat((x, torch.zeros(1, 8)), dim=1)
    elif x.ndim < 4:
        pass
    else:
        x = torch.permute(x, (2, 3, 1, 0))

    # To make sure we copy/move tensor to CPU and convert to np
    return torch.flatten(x).cpu().detach().numpy()


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        global _buffer
        global _block_num

        identity = x

        _conv_num = 1
        _layer_name = "layer" + \
            str(_layer_num)+"."+str(_block_num)+".conv"+str(_conv_num)
        if _export and not _no_intermediates:
            _buffer[_layer_name +
                    ".input"] = arrange_data(_layer_name+".input", x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if _export and not _no_intermediates:
            _buffer[_layer_name +
                    ".comp"] = arrange_data(_layer_name+".comp", out)
        _conv_num += 1
        _layer_name = "layer"+str(_layer_num)+"." + \
            str(_block_num)+".conv"+str(_conv_num)
        if _export and not _no_intermediates:
            _buffer[_layer_name +
                    ".input"] = arrange_data(_layer_name+".input", out)

        out = self.conv2(out)
        out = self.bn2(out)

        _layer_name = "layer"+str(_layer_num)+".0.downsample.0"
        if self.downsample is not None:
            if _export and not _no_intermediates:
                _buffer[_layer_name +
                        ".input"] = arrange_data(_layer_name+".input", identity)
            identity = self.downsample(x)
            if _export and not _no_intermediates:
                _buffer[_layer_name +
                        ".comp"] = arrange_data(_layer_name+".comp", identity)
        _layer_name = "layer"+str(_layer_num)+"." + \
            str(_block_num)+".conv"+str(_conv_num)

        out += identity
        out = self.relu2(out)

        if _export and not _no_intermediates:
            _buffer[_layer_name +
                    ".comp"] = arrange_data(_layer_name+".comp", out)
        _block_num += 1

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        # We are only interested in small ResNets (18 and 34)
        block: Type[BasicBlock] = BasicBlock,
        # By default we want ResNet18, ResNet34 would be [3, 4, 6, 3]
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # MOD: Changed padding to zero to match accelerator
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        global _export
        global _buffer
        global _layer_num
        global _block_num
        global _verbose

        if _export and _verbose:
            print("--- ResNet weight tensors:")
        # As a first step we get the weights from the state_dict
        if _export and not _no_weights:
            for name, param in self.state_dict().items():
                _buffer[name] = arrange_data(name, param)

        if _export and _verbose:
            print("\n--- ResNet activation tensors:")
        # Then we go through all layers and record the input and ouput tensors
        # and store them as well
        if _export:
            _buffer["conv1.input"] = arrange_data("conv1.input", x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if _export and not _no_intermediates:
            _buffer["conv1.comp"] = arrange_data("conv1.comp", x)

        x = self.maxpool(x)
        if _export and not _no_intermediates:
            _buffer["maxpool2d.comp"] = arrange_data("maxpool2d.comp", x)


        x = self.layer1(x)
        _layer_num += 1
        _block_num = 0
        x = self.layer2(x)
        _layer_num += 1
        _block_num = 0
        x = self.layer3(x)
        _layer_num += 1
        _block_num = 0
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if _export and not _no_intermediates:
            _buffer["layer4.1.conv2.comp"] = arrange_data(
                "layer4.1.conv2.comp", x)
        if _export and not _no_intermediates:
            _buffer["fc.input"] = arrange_data("fc.input", x)

        x = self.fc(x)

        if _export:
            _buffer["fc.comp"] = arrange_data("fc.comp", x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
