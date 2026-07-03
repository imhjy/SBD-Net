import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn


InputSize = Union[int, Sequence[int], torch.Size]


def _normalize_input_size(input_size: InputSize) -> Tuple[int, int, int]:
    if isinstance(input_size, int):
        return 3, input_size, input_size

    parts = tuple(int(value) for value in input_size)
    if len(parts) == 2:
        return 3, parts[0], parts[1]
    if len(parts) == 3:
        return parts
    if len(parts) == 4:
        return parts[1], parts[2], parts[3]

    raise ValueError(
        "input_size must be H, (H, W), (C, H, W), or (B, C, H, W), "
        f"got {input_size}"
    )


def _get_default_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        if "out" in output:
            return _extract_tensor(output["out"])
        for value in output.values():
            tensor = _extract_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(output, (list, tuple)):
        for value in output:
            tensor = _extract_tensor(value)
            if tensor is not None:
                return tensor
    return None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def _try_ptflops(model: nn.Module, input_shape: Tuple[int, int, int]) -> Optional[float]:
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None

    was_training = model.training
    model.eval()
    try:
        macs, _ = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return float(macs)
    except Exception:
        return None
    finally:
        model.train(was_training)


def _estimate_macs_with_hooks(model: nn.Module, dummy_input: torch.Tensor) -> float:
    macs = 0.0
    handles = []

    def add(value: float) -> None:
        nonlocal macs
        macs += float(value)

    def conv_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        output_tensor = _extract_tensor(output)
        if output_tensor is None:
            return
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups
        add(output_tensor.numel() * kernel_ops)

    def linear_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        output_tensor = _extract_tensor(output)
        if output_tensor is None:
            return
        add(output_tensor.numel() * module.in_features)

    def norm_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        output_tensor = _extract_tensor(output)
        if output_tensor is not None:
            add(2 * output_tensor.numel())

    def activation_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        output_tensor = _extract_tensor(output)
        if output_tensor is not None:
            add(output_tensor.numel())

    def pool_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        output_tensor = _extract_tensor(output)
        if output_tensor is None:
            return
        kernel_size = getattr(module, "kernel_size", 1)
        if isinstance(kernel_size, tuple):
            kernel_ops = 1
            for value in kernel_size:
                kernel_ops *= value
        else:
            kernel_ops = kernel_size
        add(output_tensor.numel() * kernel_ops)

    def upsample_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        output_tensor = _extract_tensor(output)
        if output_tensor is not None:
            add(output_tensor.numel())

    hook_map = {
        nn.Conv2d: conv_hook,
        nn.ConvTranspose2d: conv_hook,
        nn.Linear: linear_hook,
        nn.BatchNorm2d: norm_hook,
        nn.GroupNorm: norm_hook,
        nn.InstanceNorm2d: norm_hook,
        nn.LayerNorm: norm_hook,
        nn.ReLU: activation_hook,
        nn.ReLU6: activation_hook,
        nn.LeakyReLU: activation_hook,
        nn.SiLU: activation_hook,
        nn.GELU: activation_hook,
        nn.Sigmoid: activation_hook,
        nn.Softmax: activation_hook,
        nn.MaxPool2d: pool_hook,
        nn.AvgPool2d: pool_hook,
        nn.AdaptiveAvgPool2d: pool_hook,
        nn.Upsample: upsample_hook,
    }

    for module in model.modules():
        for module_type, hook in hook_map.items():
            if isinstance(module, module_type):
                handles.append(module.register_forward_hook(hook))
                break

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(dummy_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return macs


def profile_model(
    model: nn.Module,
    input_size: InputSize = (3, 256, 256),
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    warmup: int = 20,
    repeat: int = 100,
    flops_per_mac: float = 2.0,
    use_ptflops: bool = True,
    amp: bool = False,
    print_result: bool = True,
) -> Dict[str, Union[float, str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    channels, height, width = _normalize_input_size(input_size)
    target_device = torch.device(device) if device is not None else _get_default_device(model)
    model = model.to(target_device)

    total_params, trainable_params = count_parameters(model)
    dummy_input = torch.randn(batch_size, channels, height, width, device=target_device)
    flop_input = torch.randn(1, channels, height, width, device=target_device)

    macs = _try_ptflops(model, (channels, height, width)) if use_ptflops else None
    flop_backend = "ptflops" if macs is not None else "hooks"
    if macs is None:
        macs = _estimate_macs_with_hooks(model, flop_input)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                with torch.amp.autocast(target_device.type, enabled=amp and target_device.type == "cuda"):
                    model(dummy_input)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

            start_time = time.perf_counter()
            for _ in range(repeat):
                with torch.amp.autocast(target_device.type, enabled=amp and target_device.type == "cuda"):
                    model(dummy_input)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            elapsed = time.perf_counter() - start_time
    finally:
        model.train(was_training)

    result = {
        "params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "macs_g": macs / 1e9,
        "flops_g": macs * flops_per_mac / 1e9,
        "time_ms_per_img": elapsed * 1000.0 / (repeat * batch_size),
        "flop_backend": flop_backend,
    }

    if print_result:
        print(
            "Params(M): {params_m:.2f}\tFLOPs(G): {flops_g:.2f}\t"
            "Time(ms/img): {time_ms_per_img:.2f}".format(**result)
        )

    return result
