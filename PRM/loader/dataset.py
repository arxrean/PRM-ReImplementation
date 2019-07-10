import os
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pdb
import matplotlib.pyplot as plt


def pascal_voc_object_categories(query: Optional[Union[int, str]] = None) -> Union[int, str, List[str]]:
    """PASCAL VOC dataset class names.
    """

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

    if query is None:
        return categories
    else:
        for idx, val in enumerate(categories):
            if isinstance(query, int) and idx == query:
                return val
            elif val == query:
                return idx


class VOC_Classification(Dataset):
    """Dataset for PASCAL VOC classification.
    """

    def __init__(self, data_dir, dataset, split, classes, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.dataset = dataset
        self.split = split
        self.image_dir = os.path.join(data_dir, dataset, 'JPEGImages')
        assert os.path.isdir(
            self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(
            self.data_dir, self.dataset, 'ImageSets', 'Main')
        assert os.path.isdir(
            self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.image_labels = self._read_annotations(self.split)

        res_dict = {}
        for label in self.image_labels:
            cls_num = len(np.where(label[1] == 1))
            if cls_num not in res_dict:
                res_dict[cls_num] = 0
            res_dict[cls_num] += 1

        obj_num = sorted(list(res_dict.keys()))
        plt.bar(range(len(obj_num)), [res_dict.get(xtick, 0) for xtick in obj_num], align='center',yerr=0.000001)
        plt.xticks(range(len(obj_num)), obj_num)
        plt.xlabel('obj num')
        plt.ylabel('sample num')
        plt.savefig('./save/imgs/statistic_voc2012_train.png')
        exit(0)

    def _read_annotations(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if name not in class_labels:
                            class_labels[name] = np.zeros(num_classes)
                        class_labels[name][class_idx] = int(label)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

        return list(class_labels.items())

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        target = torch.from_numpy(target).float()
        img = Image.open(os.path.join(
            self.image_dir, filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return filename, img, target

    def __len__(self):
        return len(self.image_labels)


def pascal_voc_classification(
        split: str,
        data_dir: str,
        year: int = 2007,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None) -> Dataset:
    """PASCAL VOC dataset.
    """

    object_categories = pascal_voc_object_categories()
    dataset = 'VOC' + str(year)
    return VOC_Classification(data_dir, dataset, split, object_categories, transform, target_transform)
