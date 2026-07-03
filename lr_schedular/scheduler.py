import math

import torch


def unet_scheduler(optimizer,
                   num_step: int,
                   epochs: int,
                   warmup=True,
                   warmup_epochs=1,
                   warmup_factor=1e-3, **kwargs):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def step(optimizer, step_size, gamma, **kwargs):
    from torch.optim.lr_scheduler import StepLR
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def multistep(optimizer, step_size, gamma, **kwargs):
    from torch.optim.lr_scheduler import MultiStepLR
    return MultiStepLR(optimizer, milestones=step_size, gamma=gamma)


def exponential(optimizer, gamma, **kwargs):
    from torch.optim.lr_scheduler import ExponentialLR
    return ExponentialLR(optimizer, gamma)


def cosine_anneal_warm(optimizer,
                       num_step: int,
                       epochs: int,
                       step_size,
                       warmup_lr,
                       warmup_epoches,
                       lr_min,
                       gamma, **kwargs):
    from timm.scheduler.cosine_lr import CosineLRScheduler
    num_steps = epochs * num_step
    warmup_lr = warmup_lr
    warmup_steps = warmup_epoches * num_step
    lr_min = lr_min

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=lr_min,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return scheduler


def cosine_annealing_lr_warm(optimizer, t_max=80):
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            t_cur = self.last_epoch - self.warmup_epochs
            T_max = self.T_max - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t_cur / T_max)) / 2
                    for base_lr in self.base_lrs]
