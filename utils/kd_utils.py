import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


_DISTILLER_REGISTRY = {}


def register_distiller(*names):
    def decorator(cls):
        for name in names:
            _DISTILLER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def build_distiller(kd_cfg, teacher_model, student_model):
    kd_cfg = kd_cfg or {}
    if not kd_cfg.get('enable', False):
        return NoDistiller(kd_cfg, teacher_model, student_model)

    method = str(_get_required(kd_cfg, 'method')).lower()
    if method not in _DISTILLER_REGISTRY:
        raise KeyError(f"Unsupported kd.method: {kd_cfg['method']}")

    return _DISTILLER_REGISTRY[method](kd_cfg, teacher_model, student_model)


def _get_required(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing kd config field: {key}")
    return cfg[key]


def _validate_common_kd_cfg(kd_cfg):
    required_keys = ['method', 'loss', 'loss_weight', 'task_weight', 'distill_weight']
    for key in required_keys:
        _get_required(kd_cfg, key)


def _compute_distill_loss(student_tensor: torch.Tensor,
                          teacher_tensor: torch.Tensor,
                          loss_name: str,
                          loss_cfg: Dict) -> torch.Tensor:
    if not torch.is_tensor(student_tensor):
        raise TypeError("student distillation target must be a tensor")
    if not torch.is_tensor(teacher_tensor):
        raise TypeError("teacher distillation target must be a tensor")

    teacher_tensor = teacher_tensor.detach().to(student_tensor.dtype)
    loss_name = str(loss_name).lower()

    if loss_name in ['kl', 'kl_div', 'soft_target', 'soft_ce', 'ce']:
        temperature = float(_get_required(loss_cfg, 'temperature'))
        log_p = F.log_softmax(student_tensor / temperature, dim=1)
        q = F.softmax(teacher_tensor / temperature, dim=1)
        kl_map = F.kl_div(log_p, q, reduction='none')
        kd_loss = kl_map.sum(dim=1).mean()
        return kd_loss * (temperature ** 2)
    if loss_name in ['mse', 'l2']:
        return F.mse_loss(student_tensor, teacher_tensor)
    if loss_name in ['l1', 'mae']:
        return F.l1_loss(student_tensor, teacher_tensor)
    if loss_name in ['smooth_l1', 'huber']:
        return F.smooth_l1_loss(student_tensor, teacher_tensor)
    if loss_name in ['cosine', 'cosine_embedding']:
        student_flat = student_tensor.flatten(1)
        teacher_flat = teacher_tensor.flatten(1)
        return 1 - F.cosine_similarity(student_flat, teacher_flat, dim=1).mean()

    raise KeyError(f"Unsupported kd.loss: {loss_name}")


def _reduce_loss_map(loss_map: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight_map is None:
        return loss_map.mean()

    if loss_map.shape != weight_map.shape:
        raise ValueError(f"loss map shape mismatch: {loss_map.shape} vs {weight_map.shape}")

    weight_map = weight_map.to(loss_map.dtype)
    eps = torch.finfo(loss_map.dtype).eps
    return (loss_map * weight_map).sum() / weight_map.sum().clamp_min(eps)


def _compute_weighted_distill_loss(student_tensor: torch.Tensor,
                                   teacher_tensor: torch.Tensor,
                                   loss_name: str,
                                   loss_cfg: Dict,
                                   weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight_map is None:
        return _compute_distill_loss(student_tensor, teacher_tensor, loss_name, loss_cfg)

    if not torch.is_tensor(student_tensor):
        raise TypeError("student distillation target must be a tensor")
    if not torch.is_tensor(teacher_tensor):
        raise TypeError("teacher distillation target must be a tensor")

    teacher_tensor = teacher_tensor.detach().to(student_tensor.dtype)
    loss_name = str(loss_name).lower()

    if loss_name in ['kl', 'kl_div', 'soft_target', 'soft_ce', 'ce']:
        if student_tensor.dim() != 4:
            return _compute_distill_loss(student_tensor, teacher_tensor, loss_name, loss_cfg)
        temperature = float(_get_required(loss_cfg, 'temperature'))
        log_p = F.log_softmax(student_tensor / temperature, dim=1)
        q = F.softmax(teacher_tensor / temperature, dim=1)
        kl_map = F.kl_div(log_p, q, reduction='none').sum(dim=1)
        return _reduce_loss_map(kl_map, weight_map) * (temperature ** 2)

    if loss_name in ['mse', 'l2']:
        loss_map = F.mse_loss(student_tensor, teacher_tensor, reduction='none')
    elif loss_name in ['l1', 'mae']:
        loss_map = F.l1_loss(student_tensor, teacher_tensor, reduction='none')
    elif loss_name in ['smooth_l1', 'huber']:
        loss_map = F.smooth_l1_loss(student_tensor, teacher_tensor, reduction='none')
    else:
        return _compute_distill_loss(student_tensor, teacher_tensor, loss_name, loss_cfg)

    if student_tensor.dim() != 4:
        return loss_map.mean()

    spatial_loss_map = loss_map.mean(dim=1)
    return _reduce_loss_map(spatial_loss_map, weight_map)


def _sanitize_kernel_size(kernel_size: int) -> int:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _build_boundary_map(target: torch.Tensor,
                        ignore_index: int = 255,
                        kernel_size: int = 5) -> torch.Tensor:
    if target.dim() != 3:
        raise ValueError(f"boundary target must be [B, H, W], got {target.shape}")

    valid_mask = target.ne(ignore_index)
    boundary_mask = torch.zeros_like(target, dtype=torch.bool)

    vertical_diff = valid_mask[:, 1:, :] & valid_mask[:, :-1, :] & target[:, 1:, :].ne(target[:, :-1, :])
    horizontal_diff = valid_mask[:, :, 1:] & valid_mask[:, :, :-1] & target[:, :, 1:].ne(target[:, :, :-1])

    boundary_mask[:, 1:, :] |= vertical_diff
    boundary_mask[:, :-1, :] |= vertical_diff
    boundary_mask[:, :, 1:] |= horizontal_diff
    boundary_mask[:, :, :-1] |= horizontal_diff

    boundary_map = boundary_mask.float().unsqueeze(1)
    kernel_size = _sanitize_kernel_size(kernel_size)
    if kernel_size > 1:
        padding = kernel_size // 2
        boundary_map = F.max_pool2d(boundary_map, kernel_size=kernel_size, stride=1, padding=padding)

    return boundary_map.squeeze(1)


def _build_entropy_map(logits: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    entropy_map = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)
    if logits.shape[1] > 1:
        entropy_map = entropy_map / math.log(logits.shape[1])
    return entropy_map


def _resize_weight_map(weight_map: torch.Tensor, spatial_size) -> torch.Tensor:
    if tuple(weight_map.shape[1:]) == tuple(spatial_size):
        return weight_map

    return F.interpolate(weight_map.unsqueeze(1),
                         size=spatial_size,
                         mode='bilinear',
                         align_corners=False).squeeze(1)


def _unique_items(items: Iterable[str]) -> List[str]:
    unique_values = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        unique_values.append(item)
        seen.add(item)
    return unique_values


def _extract_item(obj, key: Optional[str] = None, index: Optional[int] = None):
    if key is not None:
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict output when selecting key '{key}'")
        obj = obj[key]
    if index is not None:
        obj = obj[index]
    return obj


def _resolve_tensor(role: str, cfg: Dict, outputs, hook_manager):
    module_key = f'{role}_module'
    output_key = f'{role}_key'
    index_key = f'{role}_index'

    if module_key in cfg:
        value = hook_manager.get(cfg[module_key])
    elif output_key in cfg:
        value = _extract_item(outputs, key=cfg[output_key], index=cfg.get(index_key))
    else:
        raise KeyError(f"Missing {module_key} or {output_key} in feature pair config")

    if module_key in cfg and index_key in cfg:
        value = value[cfg[index_key]]

    if not torch.is_tensor(value):
        raise TypeError(f"Resolved {role} feature is not a tensor")

    return value


def _match_feature_shape(student_feature: torch.Tensor,
                         teacher_feature: torch.Tensor,
                         pair_cfg: Dict):
    if student_feature.dim() != teacher_feature.dim():
        raise ValueError("student feature and teacher feature must have the same dims")

    channel_mode = str(pair_cfg.get('channel_mode', 'strict')).lower()
    if student_feature.shape[1] != teacher_feature.shape[1]:
        if channel_mode == 'min':
            min_channel = min(student_feature.shape[1], teacher_feature.shape[1])
            student_feature = student_feature[:, :min_channel]
            teacher_feature = teacher_feature[:, :min_channel]
        else:
            raise ValueError(
                f"feature channels mismatch: {student_feature.shape[1]} vs {teacher_feature.shape[1]}"
            )

    if student_feature.shape[2:] != teacher_feature.shape[2:]:
        if student_feature.dim() < 4:
            raise ValueError("feature spatial size mismatch and interpolation is not supported for this tensor")

        default_mode = 'bilinear' if teacher_feature.dim() == 4 else 'trilinear'
        resize_mode = pair_cfg.get('resize_mode', default_mode)
        align_corners = pair_cfg.get('align_corners', False)
        interpolate_kwargs = dict(size=student_feature.shape[2:], mode=resize_mode)
        if resize_mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            interpolate_kwargs['align_corners'] = align_corners
        teacher_feature = F.interpolate(teacher_feature, **interpolate_kwargs)

    if student_feature.shape != teacher_feature.shape:
        raise ValueError(f"feature shape mismatch: {student_feature.shape} vs {teacher_feature.shape}")

    return student_feature, teacher_feature.to(student_feature.dtype)


class FeatureHookManager:
    def __init__(self, model, module_names: Optional[Iterable[str]] = None):
        self.model = model
        self.module_names = _unique_items(module_names or [])
        self.module_dict = dict(model.named_modules())
        self.handles = []
        self.cache = {}

        for name in self.module_names:
            if name not in self.module_dict:
                raise KeyError(f"Module '{name}' not found in model")
            handle = self.module_dict[name].register_forward_hook(self._build_hook(name))
            self.handles.append(handle)

    def _build_hook(self, name):
        def hook(module, inputs, outputs):
            self.cache[name] = outputs

        return hook

    def clear(self):
        self.cache.clear()

    def get(self, name):
        if name not in self.cache:
            raise KeyError(f"Feature '{name}' not found in hook cache")
        return self.cache[name]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.cache.clear()


class SpatialWeightingMixin:
    def _init_spatial_weighting(self, kd_cfg):
        spatial_cfg = kd_cfg.get('spatial_weighting', {})
        self.spatial_weighting_cfg = spatial_cfg
        self.use_boundary_weighting = bool(spatial_cfg.get('enable', True))
        self.boundary_boost = float(spatial_cfg.get('boundary_boost', 2.0))
        self.boundary_kernel_size = int(spatial_cfg.get('boundary_kernel_size', 5))
        self.ignore_index = int(spatial_cfg.get('ignore_index', 255))
        self.uncertainty_weight = float(spatial_cfg.get('uncertainty_weight', 0.0))
        self.max_weight = float(spatial_cfg.get('max_weight', 0.0))

    def build_weight_map(self, teacher_outputs, target):
        teacher_logits = _extract_item(teacher_outputs, key='out')
        if not torch.is_tensor(teacher_logits):
            raise TypeError("teacher logits must be a tensor")

        weight_map = teacher_logits.new_ones((teacher_logits.shape[0], *teacher_logits.shape[2:]))

        if self.use_boundary_weighting:
            if target is None:
                raise ValueError("boundary-aware distillation requires target labels")
            boundary_map = _build_boundary_map(target,
                                               ignore_index=self.ignore_index,
                                               kernel_size=self.boundary_kernel_size)
            weight_map = weight_map + self.boundary_boost * boundary_map.to(weight_map.dtype)

        if self.uncertainty_weight != 0:
            entropy_map = _build_entropy_map(teacher_logits).to(weight_map.dtype)
            weight_map = weight_map + self.uncertainty_weight * entropy_map

        if self.max_weight > 0:
            weight_map = weight_map.clamp(max=self.max_weight)

        return weight_map


class BaseDistiller:
    def __init__(self, kd_cfg, teacher_model, student_model):
        _validate_common_kd_cfg(kd_cfg)
        self.kd_cfg = kd_cfg
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_weight = float(kd_cfg['loss_weight'])

    def clear(self):
        return None

    def close(self):
        return None

    @staticmethod
    def zero(student_outputs):
        if not isinstance(student_outputs, dict) or 'out' not in student_outputs:
            raise KeyError("model output must be a dict and contain key 'out'")
        return student_outputs['out'].new_zeros(())

    @staticmethod
    def get_logits(student_outputs, teacher_outputs):
        student_logits = _extract_item(student_outputs, key='out')
        teacher_logits = _extract_item(teacher_outputs, key='out')
        if not torch.is_tensor(student_logits) or not torch.is_tensor(teacher_logits):
            raise TypeError("student/teacher logits must be tensors")
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(f"logits shape mismatch: {student_logits.shape} vs {teacher_logits.shape}")
        return student_logits, teacher_logits

    def compute(self, student_outputs, teacher_outputs, target=None):
        raise NotImplementedError


class NoDistiller:
    def __init__(self, kd_cfg, teacher_model, student_model):
        self.kd_cfg = kd_cfg or {}
        self.teacher_model = teacher_model
        self.student_model = student_model

    def clear(self):
        return None

    def close(self):
        return None

    def compute(self, student_outputs, teacher_outputs, target=None):
        if not isinstance(student_outputs, dict) or 'out' not in student_outputs:
            raise KeyError("model output must be a dict and contain key 'out'")
        return student_outputs['out'].new_zeros(())


@register_distiller('logits', 'kd', 'soft_target', 'soft_label', 'hinton')
class LogitsDistiller(BaseDistiller):
    def __init__(self, kd_cfg, teacher_model, student_model):
        super().__init__(kd_cfg, teacher_model, student_model)
        self.loss_name = str(kd_cfg['loss']).lower()

    def compute(self, student_outputs, teacher_outputs, target=None):
        student_logits, teacher_logits = self.get_logits(student_outputs, teacher_outputs)
        kd_loss = _compute_distill_loss(student_logits, teacher_logits, self.loss_name, self.kd_cfg)
        return self.loss_weight * kd_loss


class FeatureDistillationMixin:
    def _init_feature_distillation(self, kd_cfg, teacher_model, student_model):
        self.feature_pairs = kd_cfg.get('feature_pairs', [])
        if len(self.feature_pairs) == 0:
            raise KeyError("feature distillation requires kd.feature_pairs")

        teacher_module_names = []
        student_module_names = []
        for pair_cfg in self.feature_pairs:
            if 'teacher_module' in pair_cfg:
                teacher_module_names.append(pair_cfg['teacher_module'])
            if 'student_module' in pair_cfg:
                student_module_names.append(pair_cfg['student_module'])

        self.teacher_hook_manager = FeatureHookManager(teacher_model, teacher_module_names)
        self.student_hook_manager = FeatureHookManager(student_model, student_module_names)
        self.feature_loss_name = str(kd_cfg.get('feature_loss', kd_cfg['loss'])).lower()

    def clear(self):
        self.teacher_hook_manager.clear()
        self.student_hook_manager.clear()

    def close(self):
        self.teacher_hook_manager.close()
        self.student_hook_manager.close()

    def compute_feature_loss(self, student_outputs, teacher_outputs, weight_map=None):
        total_feature_loss = self.zero(student_outputs)
        for pair_cfg in self.feature_pairs:
            student_feature = _resolve_tensor('student', pair_cfg, student_outputs, self.student_hook_manager)
            teacher_feature = _resolve_tensor('teacher', pair_cfg, teacher_outputs, self.teacher_hook_manager)
            student_feature, teacher_feature = _match_feature_shape(student_feature, teacher_feature, pair_cfg)

            pair_loss_cfg = dict(self.kd_cfg)
            pair_loss_cfg.update(pair_cfg)
            pair_loss_name = str(pair_cfg.get('loss', self.feature_loss_name)).lower()
            pair_weight = float(pair_cfg.get('weight', 1.0))
            pair_weight_map = None
            if weight_map is not None and student_feature.dim() == 4 and pair_cfg.get('use_spatial_weighting', True):
                pair_weight_map = _resize_weight_map(weight_map, student_feature.shape[2:])

            total_feature_loss = total_feature_loss + pair_weight * _compute_weighted_distill_loss(
                student_feature,
                teacher_feature,
                pair_loss_name,
                pair_loss_cfg,
                weight_map=pair_weight_map
            )

        return total_feature_loss


@register_distiller('feature', 'feature_distill', 'feature_distillation')
class FeatureDistiller(FeatureDistillationMixin, BaseDistiller):
    def __init__(self, kd_cfg, teacher_model, student_model):
        BaseDistiller.__init__(self, kd_cfg, teacher_model, student_model)
        self._init_feature_distillation(kd_cfg, teacher_model, student_model)

    def compute(self, student_outputs, teacher_outputs, target=None):
        return self.loss_weight * self.compute_feature_loss(student_outputs, teacher_outputs)


@register_distiller('hybrid', 'logits_feature', 'feature_logits')
class HybridDistiller(FeatureDistillationMixin, BaseDistiller):
    def __init__(self, kd_cfg, teacher_model, student_model):
        BaseDistiller.__init__(self, kd_cfg, teacher_model, student_model)
        self.logits_loss_name = str(kd_cfg.get('logits_loss', kd_cfg['loss'])).lower()
        self.logits_weight = float(kd_cfg.get('logits_weight', 1.0))
        self.feature_weight = float(kd_cfg.get('feature_weight', 1.0))

        if self.feature_weight != 0:
            self._init_feature_distillation(kd_cfg, teacher_model, student_model)
        else:
            self.feature_pairs = []
            self.teacher_hook_manager = FeatureHookManager(teacher_model, [])
            self.student_hook_manager = FeatureHookManager(student_model, [])
            self.feature_loss_name = str(kd_cfg.get('feature_loss', kd_cfg['loss'])).lower()

    def compute(self, student_outputs, teacher_outputs, target=None):
        student_logits, teacher_logits = self.get_logits(student_outputs, teacher_outputs)
        logits_loss = _compute_distill_loss(student_logits, teacher_logits, self.logits_loss_name, self.kd_cfg)
        total_kd_loss = self.logits_weight * logits_loss

        if self.feature_weight != 0:
            total_kd_loss = total_kd_loss + self.feature_weight * self.compute_feature_loss(student_outputs,
                                                                                            teacher_outputs)

        return self.loss_weight * total_kd_loss


@register_distiller('boundary_hybrid', 'edge_aware_hybrid', 'boundary_aware_hybrid')
class BoundaryAwareHybridDistiller(SpatialWeightingMixin, FeatureDistillationMixin, BaseDistiller):
    def __init__(self, kd_cfg, teacher_model, student_model):
        BaseDistiller.__init__(self, kd_cfg, teacher_model, student_model)
        self.logits_loss_name = str(kd_cfg.get('logits_loss', kd_cfg['loss'])).lower()
        self.logits_weight = float(kd_cfg.get('logits_weight', 1.0))
        self.feature_weight = float(kd_cfg.get('feature_weight', 1.0))

        if self.feature_weight != 0:
            self._init_feature_distillation(kd_cfg, teacher_model, student_model)
        else:
            self.feature_pairs = []
            self.teacher_hook_manager = FeatureHookManager(teacher_model, [])
            self.student_hook_manager = FeatureHookManager(student_model, [])
            self.feature_loss_name = str(kd_cfg.get('feature_loss', kd_cfg['loss'])).lower()

        self._init_spatial_weighting(kd_cfg)

    def compute(self, student_outputs, teacher_outputs, target=None):
        student_logits, teacher_logits = self.get_logits(student_outputs, teacher_outputs)
        weight_map = self.build_weight_map(teacher_outputs, target)
        logits_loss = _compute_weighted_distill_loss(student_logits,
                                                     teacher_logits,
                                                     self.logits_loss_name,
                                                     self.kd_cfg,
                                                     weight_map=weight_map)
        total_kd_loss = self.logits_weight * logits_loss

        if self.feature_weight != 0:
            total_kd_loss = total_kd_loss + self.feature_weight * self.compute_feature_loss(student_outputs,
                                                                                            teacher_outputs,
                                                                                            weight_map=weight_map)

        return self.loss_weight * total_kd_loss
