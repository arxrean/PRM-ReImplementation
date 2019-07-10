import os
from nest import modules, run_tasks
import argparse
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

from loader.det_dataset import PascalVOCCount


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="peak_cls_train", type=str)
    # data
    parser.add_argument(
        "--voc12_root", default='/mnt/lustre/jiangsu/dlar/home/zyk17/data/VOCdevkit/VOC2012/JPEGImages', type=str)
    parser.add_argument(
        "--json_to_pickle", default='./save/anno_dict.pkl', type=str)

    parser.add_argument("--crop_size", default=448, type=int)
    # config
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--cuda", default=False, type=bool)

    # save
    parser.add_argument("--save_weights", default='save/weights', type=str)

    args = parser.parse_args()

    return args

# count number of objects in each image in trainval dataset in voc2012


def voc12_train_countset_cnt(args):
    class_names = modules.pascal_voc_object_categories()

    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PascalVOCCount(
        json_to_pkl_file=args.json_to_pickle, transform=train_transform, args=args)
    train_loader = DataLoader(dataset, batch_size=1,
                              num_workers=0, shuffle=False)

    res_dict = dict()
    for iter, pack in enumerate(train_loader):
        imgs = pack[0]
        if args.cuda:
            imgs = imgs.cuda()
        labels = pack[1][0]
        cnt_labels = pack[1][1]

        if int(np.sum(cnt_labels.numpy()))==38:
            pass

        if int(np.sum(cnt_labels.numpy())) not in res_dict:
            res_dict[int(np.sum(cnt_labels.numpy()))] = 0
        res_dict[int(np.sum(cnt_labels.numpy()))] += 1

    print(res_dict)


if __name__ == '__main__':
    args = parse()
    voc12_train_countset_cnt(args)
