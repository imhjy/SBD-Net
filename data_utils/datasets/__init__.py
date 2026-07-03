from data_utils.augmentor import build_dataset_transformer
from data_utils.datasets.EyeOcdDataset import EyeOcdDataset
from data_utils.datasets.GoalsDataset import GoalsDataset
from data_utils.datasets.EyeOctDataset import EyeOctDataset

__all__ = {
    'EyeOcdDataset': EyeOcdDataset,
    'GoalsDataset': GoalsDataset,
    'EyeOctDataset': EyeOctDataset,

}


def build_dataset(hypes, train=True, fold=0):

    dataset_name = hypes['dataset']['method']
    error_message = f"{dataset_name} 没有找到. " \
                    f"请将数据集添加到: " \
                    f"data_utils/datasets/init.py"
    assert dataset_name in list(__all__), error_message

    dataset = __all__[dataset_name](
        hypes=hypes,
        train=train,
        transforms=build_dataset_transformer(hypes=hypes, train=train),
        fold=fold
    )

    return dataset
