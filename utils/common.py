import os
import platform
import random

import cv2
import numpy as np
import torch
from torch import Tensor

from metric.calculator import Calculator, CDRCalculator
from utils.post_process import inverse_polar_transform, inverse_polar_transform_batch


def str_dict_from_tensor(obj):
    if isinstance(obj, dict):
        return {k: str_dict_from_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, list):
        return [str_dict_from_tensor(item) for item in obj]
    else:
        return obj


def replace_system_separator(path: str):
    sys = platform.system()
    if sys == "Linux":
        return path.replace('\\', '/')
    elif sys == "Windows":
        return path.replace('/', '\\')
    return path


def remove_small_areas_based_threshold(img: Tensor, threshold=5):
    device = img.device
    img = img.cpu().numpy().astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    for i in range(1, stats.shape[0]):
        area = stats[i, 4]
        if area < threshold:
            labels[labels == i] = 0

    labels = labels.astype(np.uint8)
    ret, labels = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)
    labels[labels > 0] = 1
    return torch.tensor(labels, device=device)


def setup_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_loss(criterion, output, target, loss_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).cuda(), num_classes=5, ignore_idx=255):




    loss = criterion(output, target, loss_weight=loss_weight, weight=loss_weight,
                     num_classes=num_classes, ignore_index=ignore_idx)
    return loss


def get_disc_cup_calculator(hypes):
    disc_calculator = Calculator(num_classes=2)
    cup_calculator = Calculator(num_classes=2)
    cdr_calculator = CDRCalculator(is_polar='Polar' in hypes['augmentor']['core_method'])
    return disc_calculator, cup_calculator, cdr_calculator


def update_disc_cup_calculator(disc_calculator: Calculator, cup_calculator: Calculator, cdr_calculator: CDRCalculator,
                               output, target, is_polar=False):
    if is_polar:
        target = inverse_polar_transform(target)
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    output_argmax = output.argmax(1)
    disc_output = (output_argmax >= 1).long()
    cup_output = (output_argmax == 2).long()
    if is_polar:
        disc_output = inverse_polar_transform(disc_output)
        cup_output = inverse_polar_transform(cup_output)

    final_mask = torch.zeros_like(disc_output).to(disc_output.device)
    final_mask[disc_output == 1] = 1
    final_mask[cup_output == 1] = 2

    cdr_calculator.update(target, final_mask)
    disc_calculator.update(disc_target.flatten(), disc_output.flatten())
    cup_calculator.update(cup_target.flatten(), cup_output.flatten())


def update_disc_cup_calculator_post(disc_calculator: Calculator, cup_calculator: Calculator,
                                    cdr_calculator: CDRCalculator, output, target, is_polar=False):
    if isinstance(output, np.ndarray):
        output = torch.tensor(output).to(target.device)
    if is_polar:
        target = inverse_polar_transform(target)
    disc_target = ((target == 1) | (target == 2)).long()
    cup_target = (target == 2).long()

    disc_output = torch.logical_or(output == 128, output == 255).long()
    cup_output = (output == 255).long()

    final_mask = torch.zeros_like(disc_output).to(disc_output.device)
    final_mask[disc_output == 1] = 1
    final_mask[cup_output == 1] = 2

    cdr_calculator.update(target, final_mask)
    disc_calculator.update(disc_target.flatten(), disc_output.flatten())
    cup_calculator.update(cup_target.flatten(), cup_output.flatten())


class Predictor3Dto2DWrapper:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out_2d = self.model(x)
        out = out_2d['out']

        return out.unsqueeze(2)

class LogWriter(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == '__main__':
    tensor_dict = {
        'key1': torch.tensor(1),
        'key2': {
            'key3': torch.tensor(2),
            'key4': torch.tensor([3, 4, 5])
        },
        'key5': [torch.tensor(6), torch.tensor([7, 8])]
    }
    regular_dict = str_dict_from_tensor(tensor_dict)
    print(regular_dict)
