from data_utils.augmentor.augment import get_enhance_gray_transformer, get_nnunet_transformer, get_octa500_transformer, \
    get_octa500_2d_transformer, get_oct_transformer

__all__ = {
    'EnhanceGray': get_enhance_gray_transformer,
    'nnunet': get_nnunet_transformer,
    'OCTA500': get_octa500_transformer,
    'OCTA5002d': get_octa500_2d_transformer,
    'OCTTrans': get_oct_transformer
}


def build_dataset_transformer(hypes, train=True):

    dataset_augmentor = __all__[hypes['augmentor']['core_method']](
        train=train,
        hypes=hypes
    )

    return dataset_augmentor
