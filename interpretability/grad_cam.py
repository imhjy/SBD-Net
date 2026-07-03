from __future__ import annotations
import sys

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interpretability.common import (
    build_sample_specs,
    colorize_index_mask,
    ensure_dir,
    get_model_output_dir,
    load_hypes,
    load_rgb_image,
    locate_sample_paths,
    parse_top_k,
    sample_dir_name,
    sample_stem,
    save_rgb_image,
    decode_mask,
)


CONFIG = {
    "model_dirs": ["../logs/unet_2026_04_17_17_55_24"],
    "model_names": None,
    "dataset_root": "D:\\F\\OCT datasets\\EYE-OCT\\EYE-OCT",
    "dataset_name": None,
    "output_dir": str(Path(__file__).resolve().parent / "generated" / "grad_cam"),
    "selection_json": None,
    "top_k": '4',
    "samples": None,
    "folds": None,
    "device": None,
    "target_mode": "prediction",
    "target_classes": None,
    "target_layer": None,
    "use_ema": False,
    "cam_mode": "full",
    "pad_multiple": 32,
    "cam_max_side": 768,
    "cam_percentile": 99.5,
    "cam_smooth_kernel": 3,
    "mask_cam_to_prediction": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for segmentation models.")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to a trained model folder.")
    parser.add_argument("--model-dirs", type=str, nargs="*", default=None, help="One or more trained model folders.")
    parser.add_argument("--model-names", type=str, nargs="*", default=None, help="Optional output names for model folders.")
    parser.add_argument("--dataset-root", type=str, default=CONFIG["dataset_root"], help="Dataset root containing images/ and masks/.")
    parser.add_argument("--dataset-name", type=str, default=CONFIG["dataset_name"], help="Optional dataset name used in output paths.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG["output_dir"],
        help="Directory used to save Grad-CAM outputs.",
    )
    parser.add_argument(
        "--selection-json",
        type=str,
        default=CONFIG["selection_json"],
        help="Optional sample_metrics_by_iou.json used to select ranked samples.",
    )
    parser.add_argument("--top-k", type=str, default=CONFIG["top_k"], help="Number of ranked samples to use. Use 'all' for every sample.")
    parser.add_argument("--samples", type=str, nargs="*", default=CONFIG["samples"], help="Explicit sample ids or file names.")
    parser.add_argument("--folds", type=int, nargs="*", default=CONFIG["folds"], help="Optional folds for explicit samples.")
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["prediction", "label", "class"],
        default=CONFIG["target_mode"],
        help="How to build the Grad-CAM target score.",
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="*",
        default=CONFIG["target_classes"],
        help="Used when --target-mode class is selected. Defaults to all foreground classes.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=CONFIG["target_layer"],
        help="Optional module name. If omitted, the script chooses the last suitable Conv2d layer.",
    )
    parser.add_argument(
        "--cam-mode",
        type=str,
        choices=["full"],
        default=CONFIG["cam_mode"],
        help="Use full-image Grad-CAM. The script pads incompatible sizes automatically.",
    )
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=CONFIG["pad_multiple"],
        help="Pad image height/width to this multiple before full-image Grad-CAM. Use 1 to disable.",
    )
    parser.add_argument(
        "--cam-max-side",
        type=int,
        default=CONFIG["cam_max_side"],
        help="Resize the full image for Grad-CAM when its longest side is larger than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--cam-percentile",
        type=float,
        default=CONFIG["cam_percentile"],
        help="Upper percentile used to normalize CAM values.",
    )
    parser.add_argument(
        "--cam-smooth-kernel",
        type=int,
        default=CONFIG["cam_smooth_kernel"],
        help="Odd Gaussian blur kernel for final CAM smoothing. Use 0 to disable.",
    )
    parser.add_argument(
        "--mask-cam-to-prediction",
        action="store_true",
        default=CONFIG["mask_cam_to_prediction"],
        help="Suppress CAM values far outside predicted foreground regions.",
    )
    parser.add_argument(
        "--no-mask-cam-to-prediction",
        action="store_false",
        dest="mask_cam_to_prediction",
        help="Do not suppress CAM values outside predicted foreground regions.",
    )
    parser.add_argument("--use-ema", action="store_true", default=CONFIG["use_ema"], help="Use EMA weights if available.")
    args = parser.parse_args()

    if args.model_dirs:
        args.model_dirs = args.model_dirs
    elif args.model_dir:
        args.model_dirs = [args.model_dir]
    else:
        args.model_dirs = list(CONFIG["model_dirs"])

    if args.model_names is None:
        args.model_names = CONFIG["model_names"]
    args.top_k = parse_top_k(args.top_k)
    if args.device is None:
        args.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return args


class GradCamExtractor:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def release(self):
        for handle in self.handles:
            handle.remove()
        self.activations = None
        self.gradients = None


def forward_logits(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    output = model(image_tensor)
    if isinstance(output, dict):
        return output["out"]
    return output


def pad_image_to_multiple(image_tensor: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    original_size = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
    multiple = max(int(multiple), 1)
    if multiple <= 1:
        return image_tensor, original_size

    pad_h = (multiple - original_size[0] % multiple) % multiple
    pad_w = (multiple - original_size[1] % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image_tensor, original_size
    return torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0), original_size


def resize_image_for_cam(image_tensor: torch.Tensor, max_side: int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    original_size = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
    max_side = int(max_side or 0)
    if max_side <= 0 or max(original_size) <= max_side:
        return image_tensor, original_size, original_size

    scale = float(max_side) / float(max(original_size))
    resized_size = (
        max(int(round(original_size[0] * scale)), 1),
        max(int(round(original_size[1] * scale)), 1),
    )
    resized = torch.nn.functional.interpolate(
        image_tensor,
        size=resized_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized, original_size, resized_size


def normalize_label_tensor(label_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if label_tensor is None:
        return None
    if label_tensor.ndim == 4 and label_tensor.shape[0] == 1 and label_tensor.shape[1] == 1:
        return label_tensor[0, 0]
    if label_tensor.ndim == 3 and label_tensor.shape[0] == 1:
        return label_tensor[0]
    return label_tensor


def pad_label_to_size(label_tensor: Optional[torch.Tensor], target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    label_tensor = normalize_label_tensor(label_tensor)
    if label_tensor is None:
        return None
    pad_h = max(int(target_size[0]) - int(label_tensor.shape[-2]), 0)
    pad_w = max(int(target_size[1]) - int(label_tensor.shape[-1]), 0)
    if pad_h == 0 and pad_w == 0:
        return label_tensor
    return torch.nn.functional.pad(label_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)


def resize_label_to_size(label_tensor: Optional[torch.Tensor], target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    label_tensor = normalize_label_tensor(label_tensor)
    if label_tensor is None:
        return None
    if tuple(label_tensor.shape[-2:]) == tuple(target_size):
        return label_tensor
    return torch.nn.functional.interpolate(
        label_tensor.unsqueeze(0).unsqueeze(0).float(),
        size=target_size,
        mode="nearest",
    ).squeeze(0).squeeze(0).to(dtype=torch.long)


def crop_cam_and_prediction(
    cam: np.ndarray,
    prediction_index: np.ndarray,
    original_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = int(original_size[0]), int(original_size[1])
    return cam[:height, :width], prediction_index[:height, :width]


def resize_cam_and_prediction(
    cam: np.ndarray,
    prediction_index: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = int(target_size[0]), int(target_size[1])
    if cam.shape[:2] == (height, width):
        return cam, prediction_index
    cam = cv2.resize(cam.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    prediction_index = cv2.resize(
        prediction_index.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return cam, prediction_index.astype(np.uint8)


def discover_target_layer(model: nn.Module, image_tensor: torch.Tensor, num_classes: int) -> Tuple[str, nn.Module]:
    candidates: List[Tuple[str, nn.Conv2d]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not candidates:
        raise ValueError("No Conv2d layers were found in the model.")

    shapes: Dict[str, Tuple[int, ...]] = {}
    handles = []

    def build_hook(name):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                shapes[name] = tuple(output.shape)
        return hook

    for name, module in candidates:
        handles.append(module.register_forward_hook(build_hook(name)))

    with torch.no_grad():
        _ = forward_logits(model, image_tensor)

    for handle in handles:
        handle.remove()

    valid = []
    for name, module in candidates:
        shape = shapes.get(name)
        if shape is None or len(shape) != 4:
            continue
        if shape[-1] <= 1 or shape[-2] <= 1:
            continue
        if module.kernel_size == (1, 1) and shape[1] <= num_classes:
            continue
        valid.append((name, module))

    if not valid:
        valid = candidates
    return valid[-1]


def build_target_score(
    logits: torch.Tensor,
    target_mode: str,
    target_classes: Optional[List[int]],
    label_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if target_mode == "class":
        class_ids = target_classes or list(range(1, logits.shape[1]))
        valid_scores = [logits[0, class_id].mean() for class_id in class_ids if 0 < class_id < logits.shape[1]]
        score = sum(valid_scores) if valid_scores else logits[0, 1:].mean()
        return score

    if target_mode == "label" and label_tensor is not None:
        label_tensor = normalize_label_tensor(label_tensor)
        if tuple(label_tensor.shape[-2:]) != tuple(logits.shape[-2:]):
            label_tensor = torch.nn.functional.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(0).squeeze(0).to(dtype=torch.long)
        score = 0.0
        for class_id in torch.unique(label_tensor):
            class_index = int(class_id.item())
            if class_index <= 0 or class_index >= logits.shape[1]:
                continue
            region = label_tensor == class_index
            if region.any():
                score = score + logits[0, class_index][region].mean()
        if isinstance(score, float):
            score = logits[0, 1:].mean()
        return score

    prediction = logits.argmax(dim=1)
    score = 0.0
    for class_id in torch.unique(prediction):
        class_index = int(class_id.item())
        if class_index <= 0 or class_index >= logits.shape[1]:
            continue
        region = prediction[0] == class_index
        if region.any():
            score = score + logits[0, class_index][region].mean()
    if isinstance(score, float):
        score = logits[0, 1:].mean()
    return score


def compute_grad_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_mode: str,
    target_classes: Optional[List[int]] = None,
    label_tensor: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    extractor = GradCamExtractor(model=model, target_layer=target_layer)
    try:
        model.zero_grad(set_to_none=True)
        logits = forward_logits(model, image_tensor)
        score = build_target_score(
            logits=logits,
            target_mode=target_mode,
            target_classes=target_classes,
            label_tensor=label_tensor,
        )
        score.backward(retain_graph=False)

        if extractor.activations is None or extractor.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = extractor.gradients.detach()
        activations = extractor.activations.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        cam = cam[0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        del logits, score, gradients, activations, weights
        model.zero_grad(set_to_none=True)
        return cam, prediction
    finally:
        extractor.release()


def overlay_cam(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = original_rgb.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def postprocess_cam(
    cam: np.ndarray,
    prediction_index: np.ndarray,
    percentile: float = 99.5,
    smooth_kernel: int = 7,
    mask_to_prediction: bool = True,
) -> np.ndarray:
    cam = np.nan_to_num(cam.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cam[cam < 0] = 0

    if mask_to_prediction:
        foreground = (prediction_index > 0).astype(np.uint8)
        if foreground.any():
            kernel = np.ones((21, 21), dtype=np.uint8)
            foreground = cv2.dilate(foreground, kernel, iterations=1)
            cam = cam * foreground.astype(np.float32)

    if smooth_kernel and smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        cam = cv2.GaussianBlur(cam, (smooth_kernel, smooth_kernel), sigmaX=0)

    positive = cam[cam > 0]
    if positive.size == 0:
        return np.zeros_like(cam, dtype=np.float32)

    upper = np.percentile(positive, float(percentile))
    lower = np.percentile(positive, 1.0)
    if upper <= lower:
        lower = float(positive.min())
        upper = float(positive.max())
    cam = (cam - lower) / (upper - lower + 1e-8)
    return np.clip(cam, 0, 1).astype(np.float32)


def load_model_for_fold(model_dir: Path, fold: int, dataset_root: Path, device: str, use_ema: bool):
    from utils import train_utils

    hypes = load_hypes(model_dir=model_dir, dataset_root=dataset_root, device=device)
    model = train_utils.create_model(hypes)
    saved_path = model_dir / f"fold-{fold}"
    initial_epoch, model, _, _, _, _ = train_utils.load_saved_model(
        str(saved_path),
        model,
        optimizer=None,
        lr_scheduler=None,
        scaler=None,
        device=device,
        use_ema=use_ema,
    )
    if initial_epoch <= 0:
        raise FileNotFoundError(
            f"No checkpoint was found under {saved_path}. "
            "Grad-CAM requires net_epoch*.pth files."
        )
    model.to(device)
    model.eval()
    return hypes, model


def resolve_num_classes(hypes: Dict) -> int:
    if "num-classes" in hypes:
        return int(hypes["num-classes"]) + 1
    model_args = hypes.get("model", {}).get("args", {})
    for key in ("num_classes", "n_classes", "classes"):
        if key in model_args:
            return int(model_args[key])
    raise KeyError("Could not infer num_classes from hypes.")


def release_cuda_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_for_model(
    model_dir: Path,
    dataset_root: Path,
    output_root: Path,
    args,
    model_name: Optional[str] = None,
    selection_json: Optional[Path] = None,
) -> Dict:
    from data_utils.datasets import build_dataset

    samples = build_sample_specs(
        model_dir=model_dir,
        top_k=args.top_k,
        selection_json=selection_json,
        samples=args.samples,
        folds=args.folds,
    )
    samples = sorted(samples, key=lambda sample: (sample.fold, sample.rank or 0, sample.sample_id))
    model_output_dir = ensure_dir(
        get_model_output_dir(
            output_root=output_root,
            model_dir=model_dir,
            model_name=model_name,
            dataset_root=dataset_root,
            dataset_name=args.dataset_name,
        )
    )

    summary = {
        "model_dir": str(model_dir),
        "dataset_root": str(dataset_root),
        "output_dir": str(model_output_dir),
        "model_name": model_output_dir.name,
        "dataset_name": model_output_dir.parent.name,
        "target_mode": args.target_mode,
        "target_classes": args.target_classes,
        "cam_mode": args.cam_mode,
        "pad_multiple": args.pad_multiple,
        "cam_max_side": args.cam_max_side,
        "cam_percentile": args.cam_percentile,
        "cam_smooth_kernel": args.cam_smooth_kernel,
        "mask_cam_to_prediction": args.mask_cam_to_prediction,
        "top_k": args.top_k,
        "manual_samples": args.samples,
        "manual_folds": args.folds,
        "samples": [],
    }

    current_fold = None
    current_hypes = None
    current_model = None
    current_val_dataset = None

    for sample in samples:
        if sample.fold != current_fold:
            if current_model is not None:
                del current_model
                del current_hypes
                del current_val_dataset
                release_cuda_memory()
            current_hypes, current_model = load_model_for_fold(
                model_dir=model_dir,
                fold=sample.fold,
                dataset_root=dataset_root,
                device=args.device,
                use_ema=args.use_ema,
            )
            current_val_dataset = build_dataset(current_hypes, train=False, fold=sample.fold)
            current_fold = sample.fold

        hypes, model, val_dataset = current_hypes, current_model, current_val_dataset
        target_index = None
        target_index = None
        target_stem = sample_stem(sample.file_name)
        for idx, img_path in enumerate(val_dataset.img_list):
            if sample_stem(str(img_path)) == target_stem:
                target_index = idx
                break
        if target_index is None:
            raise FileNotFoundError(f"Could not locate sample {sample.file_name} in fold-{sample.fold} dataset.")

        image_tensor, label_tensor = val_dataset[target_index]
        image_tensor = image_tensor.unsqueeze(0).to(args.device)
        image_tensor, original_image_size, cam_input_size = resize_image_for_cam(image_tensor, args.cam_max_side)
        image_for_cam, unpadded_cam_size = pad_image_to_multiple(image_tensor, args.pad_multiple)
        label_for_cam = None
        label_tensor_for_cam = None
        if args.target_mode == "label":
            label_for_cam = resize_label_to_size(label_tensor.to(args.device), cam_input_size)
            label_tensor_for_cam = pad_label_to_size(
                label_for_cam,
                target_size=tuple(int(v) for v in image_for_cam.shape[-2:]),
            )

        named_modules = None
        if args.target_layer:
            named_modules = dict(model.named_modules())
            if args.target_layer not in named_modules:
                raise KeyError(f"Target layer '{args.target_layer}' was not found in the model.")
            layer_name = args.target_layer
            target_layer = named_modules[layer_name]
        else:
            layer_name, target_layer = discover_target_layer(
                model=model,
                image_tensor=image_for_cam,
                num_classes=resolve_num_classes(hypes),
            )

        cam, prediction_index = compute_grad_cam(
            model=model,
            image_tensor=image_for_cam,
            target_layer=target_layer,
            target_mode=args.target_mode,
            target_classes=args.target_classes,
            label_tensor=label_tensor_for_cam,
        )
        cam, prediction_index = crop_cam_and_prediction(cam, prediction_index, unpadded_cam_size)
        cam, prediction_index = resize_cam_and_prediction(cam, prediction_index, original_image_size)

        paths = locate_sample_paths(model_dir=model_dir, dataset_root=dataset_root, sample=sample)
        original_rgb = load_rgb_image(paths["original"])
        gt_index = decode_mask(paths["gt_raw"] if paths["gt_raw"].exists() else paths["label_vis"])
        cam = postprocess_cam(
            cam=cam,
            prediction_index=prediction_index,
            percentile=args.cam_percentile,
            smooth_kernel=args.cam_smooth_kernel,
            mask_to_prediction=args.mask_cam_to_prediction,
        )
        prediction_color = colorize_index_mask(prediction_index)
        overlay = overlay_cam(original_rgb, cam)
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        sample_output_dir = ensure_dir(model_output_dir / sample_dir_name(sample))
        save_rgb_image(sample_output_dir / "original.png", original_rgb)
        save_rgb_image(sample_output_dir / "grad_cam_overlay.png", overlay)
        save_rgb_image(sample_output_dir / "grad_cam_heatmap.png", heatmap)
        save_rgb_image(sample_output_dir / "prediction_color.png", prediction_color)
        metadata = {
            "sample_id": sample.sample_id,
            "file_name": sample.file_name,
            "fold": sample.fold,
            "target_layer": layer_name,
            "target_mode": args.target_mode,
            "target_classes": args.target_classes,
            "cam_mode": args.cam_mode,
            "pad_multiple": int(args.pad_multiple),
            "cam_max_side": int(args.cam_max_side),
            "original_size": [int(original_image_size[0]), int(original_image_size[1])],
            "cam_input_size": [int(cam_input_size[0]), int(cam_input_size[1])],
            "padded_size": [int(image_for_cam.shape[-2]), int(image_for_cam.shape[-1])],
            "cam_percentile": float(args.cam_percentile),
            "cam_smooth_kernel": int(args.cam_smooth_kernel),
            "mask_cam_to_prediction": bool(args.mask_cam_to_prediction),
            "gt_classes": sorted(int(x) for x in np.unique(gt_index)),
            "pred_classes": sorted(int(x) for x in np.unique(prediction_index)),
        }
        (sample_output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["samples"].append(metadata)
        print(f"Saved Grad-CAM outputs to: {sample_output_dir}")
        model.zero_grad(set_to_none=True)
        del (
            hypes,
            model,
            val_dataset,
            named_modules,
            target_layer,
            image_tensor,
            image_for_cam,
            label_tensor,
            label_for_cam,
            label_tensor_for_cam,
            original_image_size,
            cam_input_size,
            unpadded_cam_size,
            cam,
            prediction_index,
            prediction_color,
            overlay,
            heatmap,
            original_rgb,
            gt_index,
        )
        release_cuda_memory()

    if current_model is not None:
        del current_model
        del current_hypes
        del current_val_dataset
        release_cuda_memory()

    (model_output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def resolve_model_names(model_dirs: List[Path], model_names: Optional[List[str]]) -> List[Optional[str]]:
    if not model_names:
        return [None] * len(model_dirs)
    if len(model_names) != len(model_dirs):
        raise ValueError("--model-names must have the same length as --model-dirs.")
    return model_names


def main():
    if TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Grad-CAM generation requires a Python environment with PyTorch installed."
        ) from TORCH_IMPORT_ERROR

    args = parse_args()
    model_dirs = [Path(path).resolve() for path in args.model_dirs]
    model_names = resolve_model_names(model_dirs, args.model_names)
    dataset_root = Path(args.dataset_root).resolve()
    output_root = ensure_dir(Path(args.output_dir).resolve())
    selection_json = Path(args.selection_json).resolve() if args.selection_json else None

    summaries = []
    for model_dir, model_name in zip(model_dirs, model_names):
        summaries.append(
            run_for_model(
                model_dir=model_dir,
                dataset_root=dataset_root,
                output_root=output_root,
                args=args,
                model_name=model_name,
                selection_json=selection_json,
            )
        )

    (output_root / "summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
