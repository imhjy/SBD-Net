from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_tuple_rep(value, dim: int) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = tuple(int(v) for v in value)
        if len(value) == dim:
            return value
        if len(value) == 1:
            return value * dim
        raise ValueError(f"Expected sequence length {dim}, but got {len(value)}.")
    return (int(value),) * dim


def _get_conv_cls(spatial_dims: int, is_transposed: bool = False):
    if spatial_dims == 1:
        return nn.ConvTranspose1d if is_transposed else nn.Conv1d
    if spatial_dims == 2:
        return nn.ConvTranspose2d if is_transposed else nn.Conv2d
    if spatial_dims == 3:
        return nn.ConvTranspose3d if is_transposed else nn.Conv3d
    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")


def _get_norm_cls(norm_name: str, spatial_dims: int):
    norm_name = norm_name.lower()
    if norm_name == "batch":
        return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[spatial_dims]
    if norm_name == "instance":
        return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[spatial_dims]
    raise ValueError(f"Unsupported norm type: {norm_name}")


def _get_dropout_cls(spatial_dims: int):
    return {1: nn.Dropout, 2: nn.Dropout2d, 3: nn.Dropout3d}[spatial_dims]


def _get_activation(act):
    if act is None:
        return None

    kwargs = {}
    if isinstance(act, (tuple, list)):
        act_name = act[0]
        if len(act) > 1 and isinstance(act[1], dict):
            kwargs = act[1]
    else:
        act_name = act

    act_name = str(act_name).lower()
    if act_name == "relu":
        return nn.ReLU(inplace=kwargs.get("inplace", True))
    if act_name == "leakyrelu":
        return nn.LeakyReLU(
            negative_slope=kwargs.get("negative_slope", 0.01),
            inplace=kwargs.get("inplace", True),
        )
    if act_name == "gelu":
        return nn.GELU()
    if act_name == "silu":
        return nn.SiLU(inplace=kwargs.get("inplace", True))
    if act_name == "sigmoid":
        return nn.Sigmoid()
    if act_name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation type: {act_name}")


def _normalize_norm_name(norm) -> str | None:
    if norm is None:
        return None
    if isinstance(norm, str):
        return norm
    return str(norm)


def _get_same_padding(kernel_size) -> tuple[int, ...]:
    return tuple(int((k - 1) / 2) for k in kernel_size)


def _get_transposed_padding(kernel_size, stride) -> tuple[int, ...]:
    return tuple(max((k - s + 1) // 2, 0) for k, s in zip(kernel_size, stride))


def _get_output_padding(kernel_size, stride, padding) -> tuple[int, ...]:
    return tuple(max(s + 2 * p - k, 0) for k, s, p in zip(kernel_size, stride, padding))


def _get_interp_mode(spatial_dims: int) -> str:
    return {1: "linear", 2: "bilinear", 3: "trilinear"}[spatial_dims]


def get_norm_layer(norm_name, spatial_dims: int, channels: int):
    norm_name = _normalize_norm_name(norm_name)
    if norm_name is None:
        return None
    norm_cls = _get_norm_cls(norm_name, spatial_dims)
    return norm_cls(channels)


class Norm:
    BATCH = "batch"
    INSTANCE = "instance"

    @classmethod
    def __class_getitem__(cls, item):
        norm_name, spatial_dims = item
        return _get_norm_cls(_normalize_norm_name(norm_name), spatial_dims)


class ADN(nn.Sequential):

    def __init__(self, ordering: str, in_channels: int, act="relu", norm=Norm.BATCH, dropout=0.0,
                 spatial_dims: int = 2):
        super().__init__()
        norm_name = _normalize_norm_name(norm)
        for op in ordering.upper():
            if op == "A" and act is not None:
                self.add_module("A", _get_activation(act))
            elif op == "D" and float(dropout) > 0:
                self.add_module("D", _get_dropout_cls(spatial_dims)(p=float(dropout)))
            elif op == "N" and norm_name is not None:
                self.add_module("N", get_norm_layer(norm_name, spatial_dims, in_channels))


class Convolution(nn.Sequential):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            strides: int | Sequence[int] = 1,
            kernel_size: int | Sequence[int] = 3,
            padding=None,
            adn_ordering: str = "NDA",
            act="relu",
            norm=Norm.BATCH,
            dropout=0.0,
            is_transposed: bool = False,
            conv_only: bool = False,
    ):
        super().__init__()
        kernel_size = _ensure_tuple_rep(kernel_size, spatial_dims)
        strides = _ensure_tuple_rep(strides, spatial_dims)

        conv_cls = _get_conv_cls(spatial_dims, is_transposed=is_transposed)
        if padding is None:
            padding = _get_transposed_padding(kernel_size, strides) if is_transposed else _get_same_padding(kernel_size)
        else:
            padding = _ensure_tuple_rep(padding, spatial_dims)

        conv_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            bias=True,
        )
        if is_transposed:
            conv_kwargs["output_padding"] = _get_output_padding(kernel_size, strides, padding)

        self.add_module("conv", conv_cls(**conv_kwargs))
        if not conv_only:
            self.add_module(
                "adn",
                ADN(adn_ordering, out_channels, act=act, norm=norm, dropout=dropout, spatial_dims=spatial_dims),
            )


def get_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        act=None,
        norm=None,
        dropout=0.0,
        bias: bool = True,
        conv_only: bool = True,
        is_transposed: bool = False,
):
    del bias
    return Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        conv_only=conv_only,
        is_transposed=is_transposed,
    )


class UnetBasicBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Sequence[int],
            stride: int | Sequence[int],
            norm_name="instance",
    ):
        super().__init__()
        act = ("leakyrelu", {"negative_slope": 0.01, "inplace": True})
        self.conv1 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
        )
        self.norm1 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.conv2 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
        )
        self.norm2 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.lrelu = _get_activation(act)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.lrelu(self.norm2(self.conv2(out)))
        return out


class UnetResBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Sequence[int],
            stride: int | Sequence[int],
            norm_name="instance",
    ):
        super().__init__()
        act = ("leakyrelu", {"negative_slope": 0.01, "inplace": True})
        stride = _ensure_tuple_rep(stride, spatial_dims)

        self.conv1 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
        )
        self.norm1 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.conv2 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
        )
        self.norm2 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.lrelu = _get_activation(act)

        if in_channels != out_channels or any(s != 1 for s in stride):
            self.conv3 = get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                conv_only=True,
            )
            self.norm3 = get_norm_layer(norm_name, spatial_dims, out_channels)
        else:
            self.conv3 = None
            self.norm3 = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        residual = inp
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.norm2(self.conv2(out))

        if self.conv3 is not None and self.norm3 is not None:
            residual = self.norm3(self.conv3(residual))

        out = self.lrelu(out + residual)
        return out


class UnetrBasicBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Sequence[int],
            stride: int | Sequence[int],
            norm_name="instance",
            res_block: bool = True,
    ):
        super().__init__()
        block_cls = UnetResBlock if res_block else UnetBasicBlock
        self.layer = block_cls(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.layer(inp)


class UnetrUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Sequence[int],
            upsample_kernel_size: int | Sequence[int],
            norm_name="instance",
            res_block: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.transp_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )
        block_cls = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = block_cls(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.transp_conv(inp)
        if out.shape[2:] != skip.shape[2:]:
            out = F.interpolate(out, size=skip.shape[2:], mode=_get_interp_mode(self.spatial_dims), align_corners=False)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class UnetOutBlock(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            conv_only=True,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.conv(inp)
