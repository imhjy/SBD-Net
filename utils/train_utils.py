

import glob
import importlib
import math

import yaml
import sys
import os
import re
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

from utils.ema import EMA


def _load_checkpoint(model_file, map_location='cpu'):
    try:
        return torch.load(model_file, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_file, map_location=map_location)


def load_saved_model(saved_path, model, optimizer=None, lr_scheduler=None, scaler=None,
                     device='cuda', ema=None, use_ema=False):
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                if "bestval" in file_:
                    result = re.findall("net_epoch_bestval_at(.*).pth.*", file_)
                    initial_epoch_ = int(result[0])
                    return initial_epoch_, True
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_, False

    initial_epoch, flag = findLastCheckpoint(saved_path)
    best_f1 = -1
    if initial_epoch > 0:
        model_file = os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % initial_epoch) \
            if flag else os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch)
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = _load_checkpoint(model_file, map_location='cpu')
        if use_ema and 'ema_model' in checkpoint:
            print('loading ema model for inference')
            model.load_state_dict(checkpoint['ema_model'], strict=False)
        else:
            if use_ema:
                print('ema model not found, loading raw model')
            model.load_state_dict(checkpoint['model'], strict=False)
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if "best_f1" in checkpoint:
            best_f1 = checkpoint["best_f1"]
        if ema is not None:
            if 'ema_model' in checkpoint:
                ema.load_state_dict(checkpoint['ema_model'])
            else:
                ema.register()

        del checkpoint

    return initial_epoch, model, optimizer, lr_scheduler, scaler, best_f1


def create_ema(hypes, model):
    if 'ema' not in hypes or not hypes['ema']['use']:
        return None

    ema_cfg = hypes['ema']
    decay = ema_cfg['args']['decay'] if 'args' in ema_cfg and 'decay' in ema_cfg['args'] else ema_cfg.get('decay', 0.999)
    print('EMA decay: {}'.format(decay))
    return EMA(model, decay)


def setup_train(hypes):
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes):
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name
    normalized_target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower() or name.replace('_', '').lower() == normalized_target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(**backbone_config)
    return instance


def create_loss(hypes):
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('优化器方法是: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch):
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit('not supported lr schedular')

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)


def kaiming_weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return net


def initialize_layers(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
