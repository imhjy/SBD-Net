import torch
import torch.nn as nn
import torch.nn.functional as F


def make_one_hot(target, num_classes, ignore_index=-100):
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = torch.ne(target, ignore_index)

    safe_target = target.clone().long()
    safe_target[~valid_mask] = 0

    target_one_hot = F.one_hot(safe_target, num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
    valid_mask = valid_mask.unsqueeze(1)
    target_one_hot = target_one_hot * valid_mask
    return target_one_hot, valid_mask


class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1e-6, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, apply_class_weight=False, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.apply_class_weight = apply_class_weight

    def forward(self, predict, target, weight=None, ignore_index=None):
        if target.ndim == predict.ndim and target.shape[1] == 1:
            target = target.squeeze(1)

        num_classes = predict.shape[1]
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        class_weight = self.weight if weight is None else weight

        target_one_hot, valid_mask = make_one_hot(target.long(), num_classes, ignore_index)
        target_one_hot = target_one_hot.to(device=predict.device, dtype=predict.dtype)
        valid_mask = valid_mask.to(device=predict.device, dtype=predict.dtype)

        predict = F.softmax(predict, dim=1) * valid_mask
        dice = BinaryDiceLoss(**self.kwargs)

        class_losses = []
        for i in range(num_classes):
            class_losses.append(dice(predict[:, i], target_one_hot[:, i]))

        class_losses = torch.stack(class_losses)
        if self.apply_class_weight and class_weight is not None:
            class_weight = class_weight.to(device=predict.device, dtype=class_losses.dtype)
            assert class_weight.shape[0] == num_classes, \
                'Expect weight shape [{}], get[{}]'.format(num_classes, class_weight.shape[0])
            return (class_losses * class_weight).sum() / class_weight.sum().clamp_min(1e-6)

        return class_losses.mean()


class oct_cross_dice_loss(nn.Module):

    def __init__(self, loss_weight=None, ignore_index: int = -100, dice: bool = True,
                 dice_weight: bool = True, lambda_ce: float = 1.0,
                 lambda_dice: float = 1.0, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.dice = dice
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice

        self.dice_loss = DiceLoss(
            weight=loss_weight,
            ignore_index=ignore_index,
            apply_class_weight=dice_weight,
            **kwargs
        )

    def forward(self, input_data, target, weight=None, loss_weight=None, num_classes=None,
                ignore_index=None, **kwargs):
        if loss_weight is None:
            loss_weight = weight
        loss_weight = self.loss_weight if loss_weight is None else loss_weight
        ignore_index = self.ignore_index if ignore_index is None else ignore_index

        loss1 = F.cross_entropy(
            input_data,
            target.long(),
            weight=loss_weight,
            ignore_index=ignore_index
        )
        loss2 = self.dice_loss(
            input_data,
            target,
            weight=loss_weight,
            ignore_index=ignore_index
        )
        if not self.dice:
            return self.lambda_ce * loss1

        return self.lambda_ce * loss1 + self.lambda_dice * loss2


def compute_soft_class_centers_from_probs(probs, valid_mask=None):
    _, _, height, _ = probs.shape
    y_coords = torch.arange(height, device=probs.device, dtype=probs.dtype).view(1, 1, height, 1)
    if valid_mask is not None:
        probs = probs * valid_mask.to(device=probs.device, dtype=probs.dtype)

    mass = probs.sum(dim=2).clamp_min(1e-6)
    class_centers = (probs * y_coords).sum(dim=2) / mass
    return class_centers


def centers_to_boundaries(class_centers):
    return 0.5 * (class_centers[:, :-1, :] + class_centers[:, 1:, :])


def masked_mean(values, mask=None):
    if mask is None:
        return values.mean()

    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1)

    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class oct_topology_loss(nn.Module):

    def __init__(
            self,
            loss_weight=None,
            ignore_index: int = -100,
            dice: bool = True,
            dice_weight: bool = True,
            lambda_ce: float = 1.0,
            lambda_dice: float = 1.0,
            use_order: bool = True,
            use_smooth: bool = True,
            use_curvature: bool = True,
            lambda_order: float = 1.0,
            lambda_smooth: float = 0.1,
            lambda_curvature: float = 0.05,
            min_gap: float = 0.0,
            charbonnier_eps: float = 1e-3,
            **kwargs
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.use_order = use_order
        self.use_smooth = use_smooth
        self.use_curvature = use_curvature
        self.lambda_order = lambda_order
        self.lambda_smooth = lambda_smooth
        self.lambda_curvature = lambda_curvature
        self.min_gap = min_gap
        self.charbonnier_eps = charbonnier_eps
        self.loss_dict = {}

        self.seg_loss = oct_cross_dice_loss(
            loss_weight=loss_weight,
            ignore_index=ignore_index,
            dice=dice,
            dice_weight=dice_weight,
            lambda_ce=lambda_ce,
            lambda_dice=lambda_dice,
            **kwargs
        )

    @staticmethod
    def charbonnier(x, eps=1e-3):
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def softplus_barrier(x):
        return F.softplus(x)

    def extract_boundaries_from_logits(self, logits, target=None, ignore_index=None):
        valid_pixel_mask = None
        column_mask = None
        if target is not None and ignore_index is not None:
            valid_pixel_mask = torch.ne(target, ignore_index).unsqueeze(1)
            column_mask = valid_pixel_mask.any(dim=2)

        probs = F.softmax(logits, dim=1)
        class_centers = compute_soft_class_centers_from_probs(probs, valid_pixel_mask)
        pred_boundaries = centers_to_boundaries(class_centers)
        return pred_boundaries, column_mask

    def order_loss(self, pred_boundaries, column_mask=None):
        if pred_boundaries.shape[1] < 2:
            return pred_boundaries.new_tensor(0.0)

        diffs = pred_boundaries[:, 1:, :] - pred_boundaries[:, :-1, :]
        loss = self.softplus_barrier(self.min_gap - diffs)
        return masked_mean(loss, column_mask)

    def smooth_loss(self, pred_boundaries, column_mask=None):
        if pred_boundaries.shape[-1] < 2:
            return pred_boundaries.new_tensor(0.0)

        d1 = pred_boundaries[:, :, 1:] - pred_boundaries[:, :, :-1]
        smooth_mask = None
        if column_mask is not None:
            smooth_mask = column_mask[:, :, 1:] & column_mask[:, :, :-1]
        return masked_mean(self.charbonnier(d1, self.charbonnier_eps), smooth_mask)

    def curvature_loss(self, pred_boundaries, column_mask=None):
        if pred_boundaries.shape[-1] < 3:
            return pred_boundaries.new_tensor(0.0)

        d2 = pred_boundaries[:, :, 2:] - 2.0 * pred_boundaries[:, :, 1:-1] + pred_boundaries[:, :, :-2]
        curvature_mask = None
        if column_mask is not None:
            curvature_mask = column_mask[:, :, 2:] & column_mask[:, :, 1:-1] & column_mask[:, :, :-2]
        return masked_mean(self.charbonnier(d2, self.charbonnier_eps), curvature_mask)

    def forward(self, input_data, target, weight=None, loss_weight=None, num_classes=None,
                ignore_index=None, **kwargs):
        if loss_weight is None:
            loss_weight = weight
        ignore_index = self.ignore_index if ignore_index is None else ignore_index

        seg_loss = self.seg_loss(
            input_data,
            target,
            weight=loss_weight,
            loss_weight=loss_weight,
            num_classes=num_classes,
            ignore_index=ignore_index,
            **kwargs
        )
        pred_boundaries, column_mask = self.extract_boundaries_from_logits(
            input_data,
            target=target,
            ignore_index=ignore_index
        )

        ord_loss = self.order_loss(pred_boundaries, column_mask) if self.use_order else input_data.new_tensor(0.0)
        sm_loss = self.smooth_loss(pred_boundaries, column_mask) if self.use_smooth else input_data.new_tensor(0.0)
        curv_loss = self.curvature_loss(pred_boundaries, column_mask) if self.use_curvature else input_data.new_tensor(0.0)

        total_loss = (
                seg_loss
                + self.lambda_order * ord_loss
                + self.lambda_smooth * sm_loss
                + self.lambda_curvature * curv_loss
        )
        self.loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_seg": seg_loss.detach(),
            "loss_order": ord_loss.detach(),
            "loss_smooth": sm_loss.detach(),
            "loss_curvature": curv_loss.detach(),
        }
        return total_loss


def _set_topology_flags(kwargs, use_order, use_smooth, use_curvature):
    kwargs["use_order"] = use_order
    kwargs["use_smooth"] = use_smooth
    kwargs["use_curvature"] = use_curvature
    return kwargs


class oct_order_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, True, False, False))


class oct_smooth_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, False, True, False))


class oct_curvature_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, False, False, True))


class oct_order_smooth_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, True, True, False))


class oct_order_curvature_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, True, False, True))


class oct_smooth_curvature_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, False, True, True))


class oct_order_smooth_curvature_loss(oct_topology_loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_set_topology_flags(kwargs, True, True, True))

