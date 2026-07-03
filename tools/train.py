import argparse
import csv
import gc
import inspect
import os
import sys


def resolve_config_path(args):
    if args.model_dir:
        return os.path.join(args.model_dir, 'config.yaml')
    return args.hypes_yaml


def require_config_file(args):
    config_path = resolve_config_path(args)
    if not config_path:
        sys.exit("Config file is required. Pass --hypes_yaml or --model_dir containing config.yaml.")
    if not os.path.isfile(config_path):
        sys.exit(f"Config file not found: {os.path.abspath(config_path)}")
    return config_path


def _early_require_config_file():
    if __name__ != '__main__' or any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hypes_yaml", type=str, default="")
    parser.add_argument('--model_dir', type=str, default="")
    args, _ = parser.parse_known_args()
    require_config_file(args)


_early_require_config_file()

import numpy as np
from torch import nn
from utils.common import setup_seed, get_loss, LogWriter
from utils.train_utils import initialize_layers
import causal_conv1d
import time
import datetime
import torch
from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils
from loss import build_loss
from metric.calculator import ConfusionMatrixMetric, MetricsCalculator
from utils import train_utils
from lr_schedular import build_lr_schedular
import utils.distributed_utils as utils
from utils.early_stopping import EarlyStopping
import torch.nn.functional as F
from inference import main as inference_main
from utils.inferers import sliding_window_inference

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument("--hypes_yaml", type=str,
                        default="",
                        help='配置文件路径')
    parser.add_argument('--model_dir', type=str,
                        help='训练路径,与hypes_yaml二选一')
    parser.add_argument('--save_vis', type=bool, default=True,
                        help='保存语义分割后的图像')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='加载哪个epoch的模型, 为None加载最好的模型')
    parser.add_argument('--immediately_inference', type=bool, default=False,
                        help='是否训练好后立即推理')
    args = parser.parse_args()

    return args


def main(args, hypes):
    global early_stopping
    device = torch.device(hypes['device'])

    print(f"模型名称: {hypes['name']}")
    print(f"设备: {device}")
    print(f"数据增强方法: {hypes['augmentor']['core_method']}")
    print(f"数据集: {hypes['dataset']['method']}")
    print(f"epoch数: {hypes['train_params']['epoches']}")

    batch_size = hypes['train_params']['batch_size']
    num_classes = hypes['num-classes'] + 1
    fold_num_list = hypes['train_params']['train_fold_list']
    epochs = hypes['train_params']['epoches']
    f1_value = 0

    is_use_grad_norm = 'grad_norm' in hypes and hypes['grad_norm']['use'] is True

    

    if 'seed' in hypes and hypes['seed'] != -1:
        setup_seed(hypes['seed'])
    origin_path = None
    for fold in fold_num_list:
        print(f"模型名称: {hypes['name']}")
        print('-----------------Dataset Building------------------')
        train_dataset = build_dataset(hypes, train=True, fold=fold)
        val_dataset = build_dataset(hypes, train=False, fold=fold)
        num_workers = hypes['num_workers']
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)

        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        model.to(device)
        model.apply(initialize_layers)
        optimizer = train_utils.setup_optimizer(hypes, model)

        scaler = torch.amp.GradScaler(hypes['device']) if hypes['amp'] else None

        num_steps = len(train_loader)

        lr_scheduler = build_lr_schedular(optimizer, hypes, num_step=num_steps, epochs=epochs,
                                          **hypes['lr_scheduler']['args'])


        criterion = build_loss(hypes)
        loss_args = hypes.get('loss', {}).get('args', {}) or {}
        if inspect.isclass(criterion):
            criterion = criterion(loss_weight=loss_weight, **loss_args)
        elif loss_args:
            raw_criterion = criterion

            def criterion(input_data, target, **runtime_args):
                merged_args = {**loss_args, **runtime_args}
                return raw_criterion(input_data, target, **merged_args)
        ema = train_utils.create_ema(hypes, model)

        lowest_val_epoch = -1

        if args.model_dir and hypes['train_params']['enable_resume']:
            print('-----------------Load Pretrained Model------------------')
            saved_path = os.path.join(args.model_dir, f'fold-{fold}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            init_epoch, model, optimizer, lr_scheduler, scaler, f1score = train_utils.load_saved_model(saved_path,
                                                                                                       model,
                                                                                                       optimizer,
                                                                                                       lr_scheduler,
                                                                                                       scaler,
                                                                                                       device=device,
                                                                                                       ema=ema)
            lowest_val_epoch = init_epoch
        else:
            init_epoch = 0
            f1score = -1
            if origin_path is None:
                origin_path = train_utils.setup_train(hypes)
            saved_path = os.path.join(origin_path, f'fold-{fold}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            if ema is not None:
                ema.register()

        log_file = open(os.path.join(saved_path, 'program_output.log'), "w", encoding="utf-8")
        sys.stdout = LogWriter(sys.__stdout__, log_file)

        if hypes['early_stop']['use']:
            early_stopping = EarlyStopping(**hypes['early_stop']['args'])

        best_f1 = f1score
        start_time = time.time()
        continue_train = True
        for epoch in range(init_epoch, max(epochs, init_epoch)):
            if not continue_train:
                break
            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}] Fold: [{}]'.format(epoch, fold)

            for images, target in metric_logger.log_every(train_loader, 1, header):
                with torch.amp.autocast(hypes['device'], enabled=scaler is not None):
                    image, target = images.to(device), target.to(device)

                    output = model(image)
                    loss = get_loss(criterion, output['out'], target, loss_weight, num_classes=num_classes,
                                    ignore_idx=255)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if is_use_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=hypes['grad_norm']['args']['max_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update()
                else:
                    loss.backward()
                    if is_use_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=hypes['grad_norm']['args']['max_norm'])
                    optimizer.step()
                    if ema is not None:
                        ema.update()
                if hypes['lr_scheduler']['step_per_batch']:
                    lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(loss=loss.item(), lr=lr)
            if not hypes['lr_scheduler']['step_per_batch']:
                lr_scheduler.step()
            torch.cuda.empty_cache()
            gc.collect()

            print("当前时间:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if epoch % hypes['train_params']['eval_freq'] == 0 and epoch != 0:
                if ema is not None:
                    ema.apply_shadow()
                model.eval()
                calculator = ConfusionMatrixMetric(num_classes=num_classes)
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = 'Test:'
                val_loss = []
                with torch.no_grad():
                    for images, target in metric_logger.log_every(val_loader, 100, header):

                        image, target = images.to(device), target.to(device)

                        output = sliding_window_inference(inputs=image, roi_size=(
                            hypes['augmentor']['args']['crop_size'][0],
                            hypes['augmentor']['args']['crop_size'][1]),
                                                          sw_batch_size=1,
                                                          predictor=model, overlap=0.25)
                        output = output['out']
                        loss = get_loss(criterion, output, target, loss_weight, num_classes=num_classes, ignore_idx=255)
                        val_loss.append(float(loss.detach().cpu()))

                        if torch.isnan(output).any() or torch.isinf(output).any():
                            print("输出数据包含 NaN 或无穷大")
                            output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
                        calculator.update(output.argmax(1).detach().cpu(), target.cpu().detach().cpu())
                del output, image, images, target
                torch.cuda.empty_cache()
                gc.collect()

                val_info = str(calculator)
                print(f'CAVF info: \n {val_info}')
                print(f"validate loss: {torch.tensor(val_loss, dtype=torch.float32).mean().item():.3f}")
                print('验证：\n')
                all_f1_scores = calculator.compute_dict['f1_score']

                if hypes['early_stop']['use']:
                    early_stopping(sum(all_f1_scores[1:]).item())
                    if early_stopping.early_stop:
                        print("Early stopping")
                        continue_train = False
                f1_value = sum(all_f1_scores[1:]).item()
                del calculator
                if ema is not None:
                    ema.restore()

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args,
                         "best_f1": best_f1}
            if hypes['amp']:
                save_file["scaler"] = scaler.state_dict()
            if ema is not None:
                save_file["ema_model"] = ema.state_dict()

            if best_f1 < f1_value and epoch % hypes['train_params']['eval_freq'] == 0 and epoch != 0:
                best_f1 = f1_value
                save_file['best_f1'] = best_f1
                torch.save(save_file, os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                                                          'net_epoch_bestval_at%d.pth' % (
                                                                                  lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                           'net_epoch_bestval_at%d.pth' % lowest_val_epoch))
                lowest_val_epoch = epoch + 1

            if epoch % hypes['train_params']['save_freq'] == 0:
                torch.save(save_file, os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
            del save_file
            torch.cuda.empty_cache()
            gc.collect()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))

        try:
            del model, optimizer, metric_logger
            del train_loader, val_loader, train_dataset, val_dataset, scaler, lr_scheduler, criterion
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(e)

    if args.immediately_inference:
        print(f'开始进行推理: {hypes["name"]} {hypes["dataset"]["method"]}')
        inference_main(args, hypes)



def modify_config(hypes):
    hypes['amp'] = True
    hypes['train_params']['train_fold_list'] = [1]
    hypes['train_params']['epoches'] = 1
    hypes['dataset']['train_expand_rate'] = 1
    hypes['num_workers'] = 0

    return hypes


if __name__ == '__main__':
    use_queue_train = False
    if not use_queue_train:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        config_path = require_config_file(args)
        hypes = yaml_utils.load_yaml(config_path, args)
        hypes['amp'] = True
        main(args, hypes)
    else:
        device = 'cuda:0'
        model_dir_path = get_model_dir_path(2)
        for path in model_dir_path:
            print('-----------------Analyze Config File------------------')
            args = parse_args()
            args.model_dir = path
            config_path = require_config_file(args)
            hypes = yaml_utils.load_yaml(config_path, args)
            print(f'当前训练模型路径: {os.path.abspath(path)}')
            if device is not None:
                hypes['device'] = device
            hypes['num_workers'] = 4
            hypes['amp'] = False
            main(args, hypes)
