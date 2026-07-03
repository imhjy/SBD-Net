import random

import numpy as np
import torch
from torchvision.transforms import functional as F

import data_utils.augmentor.trans as T


def check_foreground_area(target, threshold=0.05):
    pixels_mask = ((target > 0) & (target < 255)).float()

    pixels_num = pixels_mask.sum()

    total_num = target.numel()

    return (pixels_num / total_num).item() >= threshold


class EnhanceGrayPresetTrain:
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5, gauss_prob=0.2, rotate_prob=0.5, angle=20,
                 mean=[0.38145922], std=[0.07928617], threshold=0.05, weights=[0.299, 0.587, 0.114], **kwargs):
        self.threshold = threshold
        trans = [T.RandomCrop2(crop_size),
                 T.ToNumpy()]
        trans.append(T.RandomReflectRotate(angle=angle, prob=rotate_prob))
        trans.extend([
            T.ConvertToGrayscale(weights=weights),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL()])
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        while True:
            trans_img, trans_target = self.transforms(img, target)
            if check_foreground_area(trans_target, self.threshold):
                break
        return trans_img, trans_target


class EnhanceGrayPresetEval:
    def __init__(self, mean=[0.38145922], std=[0.07928617], **kwargs):
        self.transforms = T.Compose([
            T.ToNumpy(),
            T.ConvertToGrayscale(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToPIL(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class PolarPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetTrain5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class PolarPresetEval5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.PolarTransform(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval2:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.RandomColorJitter(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval3:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomCropAndPad(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval4:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetTrain5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.RandomCrop2(calc_size),
            T.AddGaussianNoise(),
            T.RandomColorJitter(),
            T.RandomReflectRotate(angle=angle, prob=rotate_prob),
            T.Clahe(),
            T.GammaCorrection(),
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class CommonPresetEval5:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.Clahe(),
            T.GammaCorrection(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class nnUnetPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), calc_size=480, **kwargs):
        trans = [
            T.ToNumpy(),
            T.nnUnetTransformer(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class nnUnetPresetEval:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5, angle=20,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        trans = [
            T.ToNumpy(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        trans_img, trans_target = self.transforms(img, target)
        return trans_img, trans_target


class OCTA500PresetTrain:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.transforms = T.OCTA500Transformer()

    def __call__(self, modal_list, target):
        modal_list, target = self.transforms(modal_list, target)
        modal_list = [F.to_tensor(np.moveaxis(m, 0, -1)) for m in modal_list]
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return modal_list, target


class OCTA500PresetEval:
    def __init__(self, **kwargs):
        pass

    def __call__(self, modal_list, target):
        modal_list = [F.to_tensor(np.moveaxis(m, 0, -1)) for m in modal_list]
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return modal_list, target


class OCTA5002DPresetTrain:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.transforms = T.OCTA5002DTransformer()

    def __call__(self, image, target):
        image, target = self.transforms(image, target)
        image = torch.from_numpy(image)
        target = torch.from_numpy(np.array(target, dtype='long'))
        return image, target


class OCTA5002DPresetEval:
    def __init__(self, **kwargs):
        pass

    def __call__(self, image, target):
        image = torch.from_numpy(image)
        target = torch.from_numpy(np.array(target, dtype='long'))
        return image, target


def get_nnunet_transformer(train, hypes):

    if train:
        return nnUnetPresetTrain(**hypes['augmentor']['args'])
    else:
        return nnUnetPresetEval(**hypes['augmentor']['args'])


def get_enhance_gray_transformer(train, hypes):

    if train:
        return EnhanceGrayPresetTrain(**hypes['augmentor']['args'])
    else:
        return EnhanceGrayPresetEval(**hypes['augmentor']['args'])


def get_octa500_transformer(train, hypes):

    if train:
        return OCTA500PresetTrain(**hypes['augmentor']['args'])
    else:
        return OCTA500PresetEval(**hypes['augmentor']['args'])


def get_octa500_2d_transformer(train, hypes):

    if train:
        return OCTA5002DPresetTrain(**hypes['augmentor']['args'])
    else:
        return OCTA5002DPresetEval(**hypes['augmentor']['args'])


class OCTPresetTrain:
    def __init__(self,
                 crop_size,
                 hflip_prob=0.5,
                 translate_prob=0.5,
                 translate_px=10,
                 brightness_prob=0.5,
                 brightness_range=(0.8, 1.2),
                 scale_prob=0.5,
                 scale_range=(0.9, 1.1),
                 blur_prob=0.1,
                 noise_prob=0.1,
                 **kwargs):
        self.transforms = T.OCTTransformer(
            crop_size=crop_size,
            hflip_prob=hflip_prob,
            translate_prob=translate_prob,
            translate_px=translate_px,
            brightness_prob=brightness_prob,
            brightness_range=brightness_range,
            scale_prob=scale_prob,
            scale_range=scale_range,
            blur_prob=blur_prob,
            noise_prob=noise_prob
        )

    def __call__(self, image, target):
        image, target = self.transforms(image, target)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        target = torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64))
        return image, target


class OCTPresetEval:
    def __init__(self, **kwargs):
        pass

    def __call__(self, image, target):
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        target = torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64))
        return image, target


def get_oct_transformer(train, hypes):

    if train:
        return OCTPresetTrain(**hypes['augmentor']['args'])
    else:
        return OCTPresetEval(**hypes['augmentor']['args'])
