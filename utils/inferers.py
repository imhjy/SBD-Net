from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from metric.calculator import ConfusionMatrixMetric


def _to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def compute_sample_segmentation_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    sample_calculator = ConfusionMatrixMetric(num_classes=num_classes)
    sample_calculator.update(pred, target)
    metrics = sample_calculator.compute()

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(num_classes)]
    if len(class_names) != num_classes:
        class_names = list(class_names[:num_classes]) + [
            f"class_{idx}" for idx in range(len(class_names), num_classes)
        ]

    per_class = []
    for class_idx, class_name in enumerate(class_names):
        per_class.append({
            "class_index": class_idx,
            "class_name": str(class_name),
            "iou": _to_python_float(metrics["iou"][class_idx]),
            "dice": _to_python_float(metrics["dice"][class_idx]),
            "hd95": _to_python_float(metrics["hd95"][class_idx]),
        })

    return {
        "mean_iou": _to_python_float(metrics["mean_iou"]),
        "mean_dice": _to_python_float(metrics["mean_dice"]),
        "mean_hd95": _to_python_float(metrics["mean_hd95"]),
        "per_class": per_class,
    }


def save_sample_metrics_json(
        records: list[dict[str, Any]],
        output_path,
        class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda item: item.get("mean_iou", float("-inf")), reverse=True)

    payload = {
        "sort_key": "mean_iou",
        "order": "desc",
        "num_samples": len(sorted_records),
        "class_names": list(class_names) if class_names is not None else None,
        "samples": sorted_records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return sorted_records


def _ensure_tuple_rep(value, dim: int) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = tuple(int(v) for v in value)
        if len(value) == dim:
            return value
        if len(value) == 1:
            return value * dim
        raise ValueError(f"Expected sequence length {dim}, but got {len(value)}.")
    return (int(value),) * dim


def _get_scan_positions(image_size: int, roi_size: int, overlap: float) -> list[int]:
    if image_size <= roi_size:
        return [0]

    scan_interval = max(int(roi_size * (1.0 - overlap)), 1)
    positions = list(range(0, image_size - roi_size + 1, scan_interval))
    if positions[-1] != image_size - roi_size:
        positions.append(image_size - roi_size)
    return positions


def _pad_inputs(inputs: torch.Tensor, roi_size: tuple[int, ...], padding_mode: str, cval: float) -> torch.Tensor:
    spatial_size = inputs.shape[2:]
    pad_size = []
    for dim in reversed(range(len(spatial_size))):
        diff = max(roi_size[dim] - spatial_size[dim], 0)
        pad_size.extend([0, diff])

    if sum(pad_size) == 0:
        return inputs

    if padding_mode == "constant":
        return F.pad(inputs, pad_size, mode=padding_mode, value=cval)
    return F.pad(inputs, pad_size, mode=padding_mode)


def _compute_importance_map(
        roi_size: tuple[int, ...],
        mode: str,
        sigma_scale,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    if mode == "constant":
        return torch.ones((1, 1, *roi_size), device=device, dtype=dtype)

    if mode != "gaussian":
        raise ValueError(f"Unsupported blending mode: {mode}")

    sigma_scale = _ensure_tuple_rep(sigma_scale, len(roi_size))
    coords = [torch.arange(size, device=device, dtype=dtype) for size in roi_size]
    try:
        mesh = torch.meshgrid(*coords, indexing="ij")
    except TypeError:
        mesh = torch.meshgrid(*coords)

    importance_map = torch.ones(roi_size, device=device, dtype=dtype)
    for axis, grid in enumerate(mesh):
        center = (roi_size[axis] - 1) / 2.0
        sigma = max(roi_size[axis] * float(sigma_scale[axis]), 1e-6)
        importance_map = importance_map * torch.exp(-((grid - center) ** 2) / (2.0 * sigma ** 2))

    importance_map = importance_map / torch.clamp(importance_map.max(), min=1e-6)
    importance_map = torch.clamp(importance_map, min=1e-3)
    return importance_map.unsqueeze(0).unsqueeze(0)


def _resize_weight_map(weight_map: torch.Tensor, target_size: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if tuple(weight_map.shape[2:]) == tuple(target_size):
        return weight_map.to(dtype=dtype)

    interpolate_mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[len(target_size)]
    return F.interpolate(weight_map.float(), size=target_size, mode=interpolate_mode, align_corners=False).to(
        dtype=dtype)


def _allocate_output_buffer(
        output: Any,
        batch_size: int,
        image_size: tuple[int, ...],
        roi_size: tuple[int, ...],
        weight_map: torch.Tensor,
        device: torch.device,
):
    if torch.is_tensor(output):
        spatial_dims = len(roi_size)
        patch_size = tuple(int(v) for v in output.shape[-spatial_dims:])
        scale = tuple(patch_size[i] / float(roi_size[i]) for i in range(spatial_dims))
        full_spatial_size = tuple(int(round(image_size[i] * scale[i])) for i in range(spatial_dims))
        resized_weight = _resize_weight_map(weight_map.to(device=device), patch_size, output.dtype)
        return {
            "output": torch.zeros((batch_size, output.shape[1], *full_spatial_size), device=device, dtype=output.dtype),
            "count_map": torch.zeros((1, 1, *full_spatial_size), device=device, dtype=output.dtype),
            "scale": scale,
            "weight_map": resized_weight,
        }

    if isinstance(output, dict):
        return {key: _allocate_output_buffer(value, batch_size, image_size, roi_size, weight_map, device)
                for key, value in output.items()}

    if isinstance(output, list):
        return [_allocate_output_buffer(value, batch_size, image_size, roi_size, weight_map, device)
                for value in output]

    if isinstance(output, tuple):
        return tuple(_allocate_output_buffer(value, batch_size, image_size, roi_size, weight_map, device)
                     for value in output)

    raise TypeError(f"Unsupported predictor output type: {type(output)}")


def _accumulate_output(buffer, output: Any, batch_indices: list[int], window_starts: list[tuple[int, ...]]):
    if torch.is_tensor(output):
        weight_map = buffer["weight_map"]
        spatial_dims = len(buffer["scale"])
        for local_idx, (batch_idx, start_pos) in enumerate(zip(batch_indices, window_starts)):
            patch = output[local_idx:local_idx + 1].to(device=buffer["output"].device)
            out_slices = []
            for dim in range(spatial_dims):
                scaled_start = int(round(start_pos[dim] * buffer["scale"][dim]))
                scaled_end = scaled_start + patch.shape[2 + dim]
                out_slices.append(slice(scaled_start, scaled_end))

            target_slice = (slice(batch_idx, batch_idx + 1), slice(None), *out_slices)
            count_slice = (slice(None), slice(None), *out_slices)
            buffer["output"][target_slice] += patch * weight_map
            buffer["count_map"][count_slice] += weight_map
        return

    if isinstance(output, dict):
        for key, value in output.items():
            _accumulate_output(buffer[key], value, batch_indices, window_starts)
        return

    if isinstance(output, (list, tuple)):
        for sub_buffer, value in zip(buffer, output):
            _accumulate_output(sub_buffer, value, batch_indices, window_starts)
        return

    raise TypeError(f"Unsupported predictor output type: {type(output)}")


def _finalize_output(buffer, original_size: tuple[int, ...]):
    if isinstance(buffer, dict) and {"output", "count_map", "scale", "weight_map"} <= set(buffer.keys()):
        count_map = torch.clamp(buffer["count_map"], min=1e-6)
        output = buffer["output"] / count_map
        crop_slices = []
        for dim, scale in enumerate(buffer["scale"]):
            crop_slices.append(slice(0, int(round(original_size[dim] * scale))))
        return output[(slice(None), slice(None), *crop_slices)]

    if isinstance(buffer, dict):
        return {key: _finalize_output(value, original_size) for key, value in buffer.items()}

    if isinstance(buffer, list):
        return [_finalize_output(value, original_size) for value in buffer]

    if isinstance(buffer, tuple):
        return tuple(_finalize_output(value, original_size) for value in buffer)

    raise TypeError(f"Unsupported buffer type: {type(buffer)}")


def sliding_window_inference(
        inputs: torch.Tensor,
        roi_size,
        sw_batch_size: int,
        predictor,
        overlap: float = 0.25,
        mode: str = "constant",
        sigma_scale=0.125,
        padding_mode: str = "constant",
        cval: float = 0.0,
        sw_device=None,
        device=None,
        progress: bool = False,
        roi_weight_map: torch.Tensor | None = None,
        process_fn=None,
        buffer_steps=None,
        buffer_dim: int = -1,
        *args,
        **kwargs,
):
    del progress, buffer_steps, buffer_dim

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")

    spatial_dims = inputs.ndim - 2
    if spatial_dims <= 0:
        raise ValueError("inputs must contain batch, channel and spatial dimensions")

    roi_size = _ensure_tuple_rep(roi_size, spatial_dims)
    overlap = float(overlap)
    sw_batch_size = max(int(sw_batch_size), 1)

    sw_device = inputs.device if sw_device is None else torch.device(sw_device)
    device = inputs.device if device is None else torch.device(device)

    original_size = tuple(int(v) for v in inputs.shape[2:])
    padded_inputs = _pad_inputs(inputs, roi_size, padding_mode, cval)
    padded_size = tuple(int(v) for v in padded_inputs.shape[2:])

    if roi_weight_map is None:
        weight_map = _compute_importance_map(roi_size, mode, sigma_scale, sw_device, padded_inputs.dtype)
    else:
        if roi_weight_map.ndim == spatial_dims:
            roi_weight_map = roi_weight_map.unsqueeze(0).unsqueeze(0)
        elif roi_weight_map.ndim != spatial_dims + 2:
            raise ValueError("roi_weight_map shape is incompatible with roi_size")
        weight_map = roi_weight_map.to(device=sw_device, dtype=padded_inputs.dtype)

    scan_positions = [_get_scan_positions(padded_size[i], roi_size[i], overlap) for i in range(spatial_dims)]
    window_starts = list(product(*scan_positions))
    total_windows = [(batch_idx, start_pos) for batch_idx in range(padded_inputs.shape[0]) for start_pos in window_starts]

    output_buffer = None

    for index in range(0, len(total_windows), sw_batch_size):
        batch_windows = total_windows[index:index + sw_batch_size]
        window_data = []
        batch_indices = []
        start_positions = []

        for batch_idx, start_pos in batch_windows:
            slices = (slice(batch_idx, batch_idx + 1), slice(None)) + tuple(
                slice(start_pos[dim], start_pos[dim] + roi_size[dim]) for dim in range(spatial_dims)
            )
            window_data.append(padded_inputs[slices])
            batch_indices.append(batch_idx)
            start_positions.append(tuple(int(v) for v in start_pos))

        window_data = torch.cat(window_data, dim=0).to(sw_device)
        window_output = predictor(window_data, *args, **kwargs)

        if process_fn is not None:
            processed = process_fn(window_output, window_data, weight_map)
            window_output = processed[0] if isinstance(processed, tuple) else processed

        if output_buffer is None:
            output_buffer = _allocate_output_buffer(
                window_output,
                batch_size=padded_inputs.shape[0],
                image_size=padded_size,
                roi_size=roi_size,
                weight_map=weight_map,
                device=device,
            )

        _accumulate_output(output_buffer, window_output, batch_indices, start_positions)

    if output_buffer is None:
        raise RuntimeError("No output was produced during sliding window inference.")

    return _finalize_output(output_buffer, original_size)
