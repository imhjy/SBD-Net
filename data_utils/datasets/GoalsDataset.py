import json
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.common import replace_system_separator


class GoalsDataset(Dataset):

    def __init__(self, hypes, train: bool, transforms=None, fold=0):
        super(GoalsDataset, self).__init__()

        self.transforms = transforms
        self.train = train

        root = hypes['dataset']['root_dir']
        self.train_expand_rate = hypes['dataset']['train_expand_rate']
        self.img_list, self.label_list = [], []

        if fold == 0:
            img_root = os.path.join(root, "images")
            lab_root = os.path.join(root, "masks")

            img_names = sorted(i for i in os.listdir(img_root) if i.lower().endswith(".png"))

            self.img_list = [os.path.join(img_root, i) for i in img_names]
            self.label_list = [os.path.join(lab_root, i) for i in img_names]

        else:
            json_path = os.path.join(root, "fold.json")

            with open(json_path, "r", encoding="utf-8") as f:
                json_dict = json.load(f)

            flag = "train" if train else "test"

            self.img_list = [
                os.path.join(root, p)
                for p in json_dict[f'fold-{fold}'][f'{flag}_images']
            ]

            self.label_list = [
                os.path.join(root, p)
                for p in json_dict[f'fold-{fold}'][f'{flag}_label']
            ]

        print(f"Dataset size: {len(self.img_list)}")

    def __getitem__(self, idx):
        if self.train:
            idx = idx % len(self.img_list)

        img_path = replace_system_separator(self.img_list[idx])
        label_path = replace_system_separator(self.label_list[idx])

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        img = np.array(img)
        label = np.array(label)

        mask = self.convert_label(label)


        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        if self.train:
            return len(self.img_list) * self.train_expand_rate
        return len(self.img_list)

    @staticmethod
    def convert_label(label):

        mask = np.zeros_like(label, dtype=np.uint8)

        mask[label == 0] = 0
        mask[label == 50] = 1
        mask[label == 100] = 2
        mask[label == 150] = 3
        mask[label == 200] = 4

        return mask

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    @staticmethod
    def fold_dataset_split(hypes):

        fold_num = hypes['dataset']['fold_num']
        dataset_path = hypes['dataset']['root_dir']

        img_root = os.path.join(dataset_path, "images")

        img_list = sorted(i for i in os.listdir(img_root) if i.lower().endswith(".png"))

        images_list = np.array(
            [os.path.join("images", f) for f in img_list]
        )

        label_list = np.array(
            [os.path.join("masks", f) for f in img_list]
        )

        idx_list = np.arange(len(images_list))

        kf = KFold(n_splits=fold_num, shuffle=True)

        json_dict = {}

        for index, (train_idx, test_idx) in enumerate(kf.split(idx_list)):
            json_dict[f'fold-{index + 1}'] = {}

            json_dict[f'fold-{index + 1}']['train_images'] = list(images_list[train_idx])
            json_dict[f'fold-{index + 1}']['train_label'] = list(label_list[train_idx])

            json_dict[f'fold-{index + 1}']['test_images'] = list(images_list[test_idx])
            json_dict[f'fold-{index + 1}']['test_label'] = list(label_list[test_idx])

        with open(os.path.join(dataset_path, "fold.json"), "w") as f:
            json.dump(json_dict, f)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
