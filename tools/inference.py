import csv
import json
import os
import sys
import argparse


root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


def resolve_config_path(args):
    if args.model_dir:
        return os.path.join(args.model_dir, 'config.yaml')
    return getattr(args, 'hypes_yaml', '')


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

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_utils.datasets import build_dataset
from hypes_yaml import yaml_utils
from metric.calculator import ConfusionMatrixMetric, CollectIOU
from utils import train_utils
import utils.distributed_utils as utils
from utils.inferers import (
    compute_sample_segmentation_metrics,
    save_sample_metrics_json,
    sliding_window_inference,
)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
DEFAULT_SOFT_PALETTE = [
    [0, 0, 0],
    [141, 211, 199],
    [255, 255, 179],
    [190, 186, 218],
    [251, 128, 114],
    [128, 177, 211],
    [253, 180, 98],
    [179, 222, 105],
    [252, 205, 229],
    [217, 217, 217],
]
IGNORE_INDEX_COLOR = [210, 210, 210]


def parse_args():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in ('yes', 'true', 't', '1', 'y'):
            return True
        if value in ('no', 'false', 'f', '0', 'n'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="推理参数")
    parser.add_argument("--hypes_yaml", type=str,
                        default="",
                        help='config file path')
    parser.add_argument('--model_dir', type=str,
                        default="",
                        help='模型路径')
    parser.add_argument('--save_vis', type=str2bool, nargs='?', const=True, default=True,
                        help='保存语义分割后的图像')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='加载哪个epoch的模型, 为None加载最好的模型')
    parser.add_argument('--tta', action='store_true',
                        default=True,
                        help='enable test time augmentation')
    parser.add_argument('--use_ema', action='store_true',
                        default=False,
                        help='enable ema model for inference')
    opt = parser.parse_args()
    return opt


def _ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def _format_float(x, fmt="{:.6f}"):
    try:
        return fmt.format(float(x))
    except Exception:
        return "NA"


def write_csv(saved_path: str,
              calculator,
              info: str,
              fold: Optional[int] = 0,
              class_names: Optional[List[str]] = None,
              csv_name: str = "validate_results.csv"):
    if not hasattr(calculator, "compute_dict") or not calculator.compute_dict:
        calculator.compute()

    num_classes = calculator.num_classes
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    assert len(class_names) == num_classes

    per_metric_keys = ["iou", "dice", "hd95", "f1_score", "precision", "recall", "accuracy", "sample_dice"]
    fieldnames = ["info", "fold"]
    for cname in class_names:
        for k in per_metric_keys:
            fieldnames.append(f"{cname}_{k}")
    summary_keys = ["mean_iou", "mean_dice", "mean_hd95", "mean_sample_dice", "mean_f1", "mean_accuracy",
                    "accuracy_global"]
    fieldnames.extend(summary_keys)

    csv_path = os.path.join(saved_path, csv_name)
    _ensure_dir(csv_path)
    write_header = not os.path.exists(csv_path)

    cd = calculator.compute_dict
    row = {"info": info, "fold": str(fold)}
    for i, cname in enumerate(class_names):
        row[f"{cname}_iou"] = _format_float(cd["iou"][i])
        row[f"{cname}_dice"] = _format_float(cd["dice"][i])
        row[f"{cname}_hd95"] = _format_float(cd["hd95"][i])
        row[f"{cname}_f1_score"] = _format_float(cd["f1_score"][i])
        row[f"{cname}_precision"] = _format_float(cd["precision"][i])
        row[f"{cname}_recall"] = _format_float(cd["recall"][i])
        row[f"{cname}_accuracy"] = _format_float(cd["accuracy"][i])
        row[f"{cname}_sample_dice"] = _format_float(0) if i == 0 else _format_float(cd["sample_mean_dice"][i - 1])

    row["mean_iou"] = _format_float(cd.get("mean_iou", np.mean(cd["iou"])))
    row["mean_dice"] = _format_float(cd.get("mean_dice", np.mean(cd["dice"])))
    row["mean_hd95"] = _format_float(cd.get("mean_hd95", np.mean(cd["hd95"][1:])))
    row["mean_sample_dice"] = _format_float(cd.get("mean_sample_dice", np.mean(cd["sample_mean_dice"])))
    row["mean_f1"] = _format_float(cd.get("mean_f1", np.mean(cd["f1_score"])))
    row["mean_accuracy"] = _format_float(cd.get("mean_accuracy", np.mean(cd["accuracy"])))
    row["accuracy_global"] = _format_float(cd.get("accuracy_global", np.mean(cd["accuracy"])))

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        ordered_row = {fn: row.get(fn, "NA") for fn in fieldnames}
        writer.writerow(ordered_row)


def write_summary_csv(saved_path: str,
                      calculator_list: List,
                      info: str,
                      class_names: Optional[List[str]] = None,
                      csv_name: str = "validate_results.csv"):
    assert len(calculator_list) > 0
    num_classes = calculator_list[0].num_classes
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    assert len(class_names) == num_classes

    per_metric_keys = ["iou", "dice", "hd95", "f1_score", "precision", "recall", "accuracy", "sample_dice"]
    fieldnames = ["info", "fold"]
    for cname in class_names:
        for k in per_metric_keys:
            fieldnames.append(f"{cname}_{k}")
    summary_keys = ["mean_iou", "mean_dice", "mean_hd95", "mean_sample_dice", "mean_f1", "mean_accuracy",
                    "accuracy_global"]
    fieldnames.extend(summary_keys)

    csv_path = os.path.join(saved_path, csv_name)
    _ensure_dir(csv_path)
    write_header = not os.path.exists(csv_path)

    rows = []
    for i, calc in enumerate(calculator_list):
        if not hasattr(calc, "compute_dict") or not calc.compute_dict:
            calc.compute()
        cd = calc.compute_dict
        row = {"info": info, "fold": str(i + 1)}
        for j, cname in enumerate(class_names):
            row[f"{cname}_iou"] = float(np.asarray(cd["iou"][j]))
            row[f"{cname}_dice"] = float(np.asarray(cd["dice"][j]))
            row[f"{cname}_hd95"] = float(np.asarray(cd["hd95"][j]))
            row[f"{cname}_f1_score"] = float(np.asarray(cd["f1_score"][j]))
            row[f"{cname}_precision"] = float(np.asarray(cd["precision"][j]))
            row[f"{cname}_recall"] = float(np.asarray(cd["recall"][j]))
            row[f"{cname}_accuracy"] = float(np.asarray(cd["accuracy"][j]))
            row[f"{cname}_sample_dice"] = float(np.asarray(0)) if j == 0 else float(
                np.asarray(cd["sample_mean_dice"][j - 1]))
        row["mean_iou"] = float(np.asarray(cd.get("mean_iou", np.mean(cd["iou"]))))
        row["mean_dice"] = float(np.asarray(cd.get("mean_dice", np.mean(cd["dice"]))))
        row["mean_hd95"] = float(np.asarray(cd.get("mean_hd95", np.mean(cd["hd95"][1:]))))
        row["mean_sample_dice"] = float(np.asarray(cd.get("mean_sample_dice", np.mean(cd["sample_mean_dice"]))))
        row["mean_f1"] = float(np.asarray(cd.get("mean_f1", np.mean(cd["f1_score"]))))
        row["mean_accuracy"] = float(np.asarray(cd.get("mean_accuracy", np.mean(cd["accuracy"]))))
        row["accuracy_global"] = float(np.asarray(cd.get("accuracy_global", np.mean(cd["accuracy"]))))
        rows.append(row)

    agg = {}
    numeric_fields = [fn for fn in fieldnames if fn not in ("info", "fold")]
    for fn in numeric_fields:
        vals = []
        for r in rows:
            v = r.get(fn, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        agg[fn] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            out = {}
            for fn in fieldnames:
                if fn in ("info", "fold"):
                    out[fn] = r.get(fn, "")
                else:
                    out[fn] = _format_float(r.get(fn, "NA"))
            writer.writerow(out)

        mean_row = {"info": info, "fold": "mean"}
        for fn in numeric_fields:
            mean_row[fn] = _format_float(agg[fn])
        ordered_mean = {fn: mean_row.get(fn, "NA") for fn in fieldnames}
        writer.writerow(ordered_mean)

    print("Summary written to:", csv_path)
    print(json.dumps({k: _format_float(v) for k, v in agg.items()}, indent=2))


def _segmentation_to_numpy(image):
    if torch.is_tensor(image):
        image_array = image.detach().cpu().numpy()
    else:
        image_array = np.asarray(image)

    image_array = np.squeeze(image_array)
    if image_array.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation map, got shape {image_array.shape}.")
    return image_array.astype(np.int64, copy=False)


def _get_color_map(num_classes=None, color_map=None):
    if color_map is not None:
        if isinstance(color_map, dict):
            return {int(class_idx): list(color) for class_idx, color in color_map.items()}
        return {class_idx: list(color) for class_idx, color in enumerate(color_map)}

    if num_classes is None:
        num_classes = len(DEFAULT_SOFT_PALETTE)
    return {
        class_idx: DEFAULT_SOFT_PALETTE[class_idx % len(DEFAULT_SOFT_PALETTE)]
        for class_idx in range(num_classes)
    }


def _resolve_class_names(hypes, num_classes):
    names = hypes.get('class_names') if isinstance(hypes, dict) else None
    if names:
        names = list(names)
    else:
        names = ['BG'] + [str(class_idx) for class_idx in range(1, num_classes)]

    if len(names) < num_classes:
        names.extend(str(class_idx) for class_idx in range(len(names), num_classes))
    return names[:num_classes]


def _get_sample_vis_name(val_dataset, idx):
    image_path = (
        val_dataset.img_list[idx]
        if hasattr(val_dataset, "img_list") and idx < len(val_dataset.img_list)
        else f"sample_{idx}"
    )
    file_name = os.path.basename(str(image_path))
    stem, _ = os.path.splitext(file_name)
    return f"{stem}.png"


def save_image(image, save_path, file_name, num_classes=None, color_map=None):
    image_array = _segmentation_to_numpy(image)
    resolved_color_map = _get_color_map(num_classes=num_classes, color_map=color_map)

    result_color = np.zeros((*image_array.shape, 3), dtype=np.uint8)
    for class_idx, color in resolved_color_map.items():
        result_color[image_array == class_idx] = color
    result_color[image_array == 255] = IGNORE_INDEX_COLOR

    safe_file_name = os.path.basename(str(file_name))
    output_path = os.path.join(save_path, safe_file_name)
    _ensure_dir(output_path)
    Image.fromarray(result_color).save(output_path)
    return result_color


def remove_all_csv(base_path, fold_num_list):
    for fold in fold_num_list:
        csv_path = os.path.join(base_path, f'fold-{fold}', 'validate_results.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
    csv_path = os.path.join(base_path, 'validate_results.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)


def _build_sample_metric_record(sample_metrics, val_dataset, fold: int, idx: int):
    image_path = (
        val_dataset.img_list[idx]
        if hasattr(val_dataset, "img_list") and idx < len(val_dataset.img_list)
        else f"sample_{idx}"
    )
    image_path = str(image_path)
    file_name = os.path.basename(image_path)

    record = {
        "fold": int(fold),
        "sample_index": int(idx),
        "file_name": file_name,
        "image_path": image_path,
    }
    record.update(sample_metrics)
    return record


def inference_once(model, image, hypes):
    output = sliding_window_inference(inputs=image, roi_size=(
        hypes['augmentor']['args']['crop_size'][0],
        hypes['augmentor']['args']['crop_size'][1]),
                                      sw_batch_size=1,
                                      predictor=model, overlap=0.25)
    return output['out']


def get_tta_transform():
    def _affine_transform_tensor(x, theta, mode='bilinear', padding_mode='zeros'):
        b = x.size(0)
        theta = x.new_tensor(theta).unsqueeze(0).repeat(b, 1, 1)
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    def _rotate_tensor(x, angle):
        rad = np.deg2rad(angle)
        cos_a = float(np.cos(rad))
        sin_a = float(np.sin(rad))
        return _affine_transform_tensor(
            x,
            [[cos_a, -sin_a, 0.0],
             [sin_a, cos_a, 0.0]],
            mode='bilinear',
            padding_mode='zeros'
        )

    def _identity_tensor(x):
        return x

    return [
        ('origin',
         _identity_tensor,
         _identity_tensor),
        ('hflip',
         lambda x: torch.flip(x, dims=[-1]),
         lambda x: torch.flip(x, dims=[-1])),
        ('rot_p5',
         lambda x: _rotate_tensor(x, 5.0),
         lambda x: _rotate_tensor(x, -5.0)),
        ('rot_m5',
         lambda x: _rotate_tensor(x, -5.0),
         lambda x: _rotate_tensor(x, 5.0)),
    ]


def tta_inference(model, image, hypes):
    output_list = []
    for _, transform_func, inverse_func in get_tta_transform():
        aug_image = transform_func(image)
        aug_output = inference_once(model, aug_image, hypes)
        output_list.append(inverse_func(aug_output))

    return torch.stack(output_list, dim=0).mean(dim=0)


def inference_with_tta(model, image, hypes, use_tta=False):
    if use_tta:
        return tta_inference(model, image, hypes)

    return inference_once(model, image, hypes)


def main(args, hypes):
    device = torch.device(hypes['device'])
    num_classes = hypes['num-classes'] + 1
    fold_num_list = hypes['train_params']['train_fold_list']
    use_tta = getattr(args, 'tta', False)
    use_ema = getattr(args, 'use_ema', False)
    info_list = []
    if use_tta:
        info_list.append('tta')
    if use_ema:
        info_list.append('ema')
    info = '_'.join(info_list) if len(info_list) > 0 else 'origin'
    class_names = _resolve_class_names(hypes, num_classes)
    calculator_list = []
    remove_all_csv(args.model_dir if args.model_dir else args.hypes_yaml, fold_num_list)
    global_calculator = ConfusionMatrixMetric(num_classes=num_classes)
    collectIOU = CollectIOU()
    sample_metric_records = []
    for fold in fold_num_list:
        print('-----------------Dataset Building------------------')
        val_dataset = build_dataset(hypes, train=False, fold=fold)
        num_workers = hypes['num_workers']
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)
        print('---------------Creating Model------------------')
        model = train_utils.create_model(hypes)
        print('-----------------Load Pretrained Model------------------')
        load_path = os.path.join(args.model_dir, f'fold-{fold}')
        _, model, _, _, _, _ = train_utils.load_saved_model(load_path, model, None, None, None,
                                                            use_ema=use_ema)
        model.to(device)
        print('-----------------Eval Step------------------')
        if use_tta:
            print('-----------------TTA Inference------------------')
        if use_ema:
            print('-----------------EMA Inference------------------')
        model.eval()
        calculator = ConfusionMatrixMetric(num_classes=num_classes)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test Fold [{fold}/{len(fold_num_list)}]:'

        with torch.no_grad():
            idx = 0
            for images, target in metric_logger.log_every(val_loader, 100, header):
                image, target = images.to(device), target.to(device)

                output = None if use_tta else sliding_window_inference(inputs=image, roi_size=(
                    hypes['augmentor']['args']['crop_size'][0],
                    hypes['augmentor']['args']['crop_size'][1]),
                                                                       sw_batch_size=1,
                                                                       predictor=model, overlap=0.25)
                if use_tta:
                    output = inference_with_tta(model, image, hypes, use_tta=use_tta)
                else:
                    output = output['out']
                pred = output.argmax(1)

                if getattr(args, 'save_vis', True):
                    file_name = _get_sample_vis_name(val_dataset, idx)
                    save_root = os.path.join(load_path, 'save_images')
                    save_image(
                        pred,
                        os.path.join(save_root, 'predictions'),
                        file_name,
                        num_classes=num_classes,
                    )
                    save_image(
                        target,
                        os.path.join(save_root, 'labels'),
                        file_name,
                        num_classes=num_classes,
                    )

                sample_metrics = compute_sample_segmentation_metrics(
                    pred,
                    target,
                    num_classes=num_classes,
                    class_names=class_names,
                )
                sample_metric_records.append(
                    _build_sample_metric_record(sample_metrics, val_dataset, fold, idx)
                )

                calculator.update(pred, target)
                collectIOU.update(pred, target)
                global_calculator.update(pred, target)
                idx += 1

        val_info = str(calculator)
        print(f'CAVF info: \n {val_info}')
        write_csv(load_path, calculator, info=info, class_names=class_names)
        calculator_list.append(calculator)
        del model
    write_summary_csv(args.model_dir, calculator_list, info=info, class_names=class_names)

    write_csv(args.model_dir, global_calculator, info=f'global_{info}', class_names=class_names)

    sample_metrics_path = os.path.join(args.model_dir, 'sample_metrics_by_iou.json')
    save_sample_metrics_json(
        sample_metric_records,
        sample_metrics_path,
        class_names=class_names,
    )
    print(f'Sample metrics written to: {sample_metrics_path}')

    with open(os.path.join(args.model_dir, 'IOU.json'), 'w') as f:
        data = collectIOU.get_metrics()
        json.dump(data, f)


if __name__ == '__main__':
    use_queue_train = False
    if use_queue_train:
        model_dir_path = [
        ]
        for path in model_dir_path:
            print('-----------------Analyze Config File------------------')
            args = parse_args()
            args.model_dir = path
            config_path = require_config_file(args)
            hypes = yaml_utils.load_yaml(config_path, args)
            device = 'cuda:0' 
            if device is not None:
                hypes['device'] = device
            hypes['dataset']['root_dir'] = ""
            hypes['num_workers'] = 4
            main(args, hypes)
    else:
        print('-----------------Analyze Config File------------------')
        args = parse_args()
        config_path = require_config_file(args)
        hypes = yaml_utils.load_yaml(config_path, args)
        main(args, hypes)
