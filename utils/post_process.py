import numpy as np
import torch
from scipy.ndimage import rotate
from PIL import Image
import cv2
from torch import Tensor
from scipy.ndimage import binary_fill_holes
from skimage import measure

def inverse_polar_transform(image):
    device = None
    if isinstance(image, torch.Tensor):
        device = image.device
        image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

    img = rotate(image, 90, reshape=False)
    h, w = img.shape
    center = (w // 2, h // 2)
    max_radius = min(center[0], center[1])
    restored_img = cv2.warpPolar(
        src=img,
        dsize=[h, w],
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP
    )
    if device is not None:
        restored_img = torch.from_numpy(restored_img).to(device)
    return restored_img


def inverse_polar_transform_batch(image):
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        device = image.device
        image_np = image.cpu().numpy()
    else:
        image_np = np.asarray(image)

    batch_size = image_np.shape[0]
    restored_imgs = []

    for i in range(batch_size):
        img = image_np[i]
        img = rotate(img, 90, reshape=False)

        h, w = img.shape
        center = (w // 2, h // 2)
        max_radius = min(center[0], center[1])
        img = np.array(img, dtype=np.uint8)
        restored_img = cv2.warpPolar(
            src=img,
            dsize=(w, h),
            center=center,
            maxRadius=max_radius,
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )
        restored_imgs.append(restored_img)

    restored_imgs_np = np.stack(restored_imgs, axis=0)

    if is_tensor:
        return torch.from_numpy(restored_imgs_np).to(device)
    else:
        return restored_imgs_np


def keep_maximum_connectivity(disc_map: np.ndarray, cup_map: np.ndarray):
    def process(img):
        if not np.any(img):
            return img

        binary = np.zeros_like(img, dtype=np.uint8)
        binary[img != 0] = 255

        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if regions:
            max_region_idx = np.argmax([r.area for r in regions]) + 1

            processed = np.zeros_like(binary)
            processed[labeled == max_region_idx] = 255

            processed = binary_fill_holes(processed)

            result = np.zeros_like(img)
            result[processed != 0] = 128
            return result
        else:
            return img

    return process(disc_map), process(cup_map)


def check_cup_in_disc_area(disc_map: np.ndarray, cup_map: np.ndarray):
    disc_mask = disc_map.astype(bool)
    cup_mask = cup_map.astype(bool)

    overflow_mask = np.logical_and(cup_mask, ~disc_mask)

    corrected_cup = np.where(overflow_mask, 0, cup_map)

    if np.any(overflow_mask):
        print(f"发现{np.sum(overflow_mask)}个越界像素已清除")

    return disc_map, corrected_cup.astype(np.uint8)


def ellipse_fitting(disc_map: np.ndarray, cup_map: np.ndarray):

    def process_mask(mask):
        binary = np.where(mask == 128, 255, 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f'轮廓未找到.')
            return mask

        max_contour = max(contours, key=cv2.contourArea)

        output = np.zeros_like(mask, dtype=np.uint8)

        if len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            cv2.ellipse(output, ellipse, color=128, thickness=-1)

            return output
        return mask

    disc_ellipse = process_mask(disc_map)
    cup_ellipse = process_mask(cup_map)

    return disc_ellipse, cup_ellipse
