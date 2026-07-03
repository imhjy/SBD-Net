import numpy as np
import torch
from torch import nn
import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)




def cross_dice_loss(input_data, target, loss_weight=None, num_classes: int = 2, dice: bool = True,
                    ignore_index: int = -100,
                    **kwargs):
    losses = {}

    loss = nn.functional.cross_entropy(input_data, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(input_data, dice_target, multiclass=True, ignore_index=ignore_index)

    return loss


def get_bce_target(target, num_classes=3):
    nhot_shape = [target.shape[0], num_classes, *list(target.shape)[1:]]
    if isinstance(target, torch.Tensor):
        mask_nhot = torch.zeros(nhot_shape, device=target.device)
    else:
        mask_nhot = np.zeros(nhot_shape)

    mask_nhot[:,0] = (target == 0)
    mask_nhot[:,1] = (target >= 1)
    mask_nhot[:,2] = (target == 2)
    return mask_nhot

def bce_dice_loss(input_data, target, loss_weight=None, num_classes: int = 3, dice: bool = True,
                  ignore_index: int = -100,
                  **kwargs):
    bce_loss_func = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    loss = bce_loss_func(input_data.permute([0, 2, 3, 1]), get_bce_target(target,num_classes).permute([0, 2, 3, 1]))
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(input_data, dice_target, multiclass=True, ignore_index=ignore_index)

    return loss

def cross_entropy_loss(input_data, target, loss_weight=None, num_classes: int = 3, dice: bool = True,
                  ignore_index: int = -100,
                  **kwargs):
    func = nn.CrossEntropyLoss()
    loss = func(input_data, target)

    return loss