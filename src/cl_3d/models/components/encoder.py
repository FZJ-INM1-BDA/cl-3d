from typing import Type, Union, List, Optional, Callable

import torch
from torch import Tensor
import torch.nn as nn

import torchvision
from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from cl_3d.models.components.layers import RunningNorm2D, PLIMods


class ProjectionHead(nn.Sequential):

    def __init__(
            self,
            features: List[int],
    ):
        assert len(features) >= 2
        
        self.features = features

        layers = [nn.Linear(features[0], features[1], bias=False)]
        for i in range(2, len(features)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(features[i-1], features[i], bias=False))

        super().__init__(*layers)


# Overwrite initialization from torchvision.models.resnet.ResNet
class ResNetEncoder(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            planes: List[int],
            channels: int,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = planes[0]
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
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=5, stride=4, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResNet._make_layer(self, block, planes[0], layers[0])
        self.layer2 = ResNet._make_layer(self, block, planes[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = ResNet._make_layer(self, block, planes[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = ResNet._make_layer(self, block, planes[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _forward_impl(
            self,
            x: Tensor
    ) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        out = self.avgpool(l4)
        out = torch.flatten(out, 1)

        return [l1, l2, l3, l4, out]

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)


class PLIResnetEncoder(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck, str]],
            layers: List[int],
            planes: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = nn.BatchNorm2d,
    ):
        super().__init__()

        if type(block) is str:
            block = getattr(torchvision.models.resnet, block)
        if type(norm_layer) is str:
            norm_layer = getattr(nn, norm_layer)

        self.input_layer = PLIMods()
        self.norm_layer = RunningNorm2D(3)
        self.encoder = ResNetEncoder(block, layers, planes, 3, zero_init_residual, groups, width_per_group,
                                     replace_stride_with_dilation, norm_layer)

    def forward(
            self,
            trans: torch.Tensor,
            dir: torch.Tensor,
            ret: torch.Tensor
    ):
        x = self.input_layer(trans, dir, ret)
        x = self.norm_layer(x)
        x = self.encoder(x)
        return x
