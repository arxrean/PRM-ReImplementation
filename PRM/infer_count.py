from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from loader.det_dataset import PascalVOCDetection, convert_json_labels_to_csv, PascalVOCCount
from model.backbone import fc_resnet50
from model.peak_net import peak_response_mapping
from loader.dataset import pascal_voc_classification
from nest import modules, run_tasks
from scipy.misc import imresize
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import json
import os
import json
import pickle
import warnings
import argparse
import pdb
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="peak_cls_train", type=str)
    # data
    parser.add_argument(
        "--voc12_root", default='/u/zkou2/Data/VOCdevkit/VOC2012/JPEGImages', type=str)
    parser.add_argument(
        "--json_to_pickle", default='/u/zkou2/Code/PRM-ReImplementation/PRM/save/anno_dict.pkl', type=str)

    parser.add_argument("--crop_size", default=448, type=int)
    # config
    parser.add_argument("--batch_size", default=16, type=int)

    # save
    parser.add_argument("--save_weights", default='save/weights', type=str)

    args = parser.parse_args()

    return args


def voc12_train_det(args):
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PascalVOCDetection(images_folder_path='/u/zkou2/Data/VOCdevkit/VOC2012/JPEGImages',
                                 annotation_json='/u/zkou2/Data/VOCdevkit/PASCAL_VOC_JSON/pascal_train2012.json',
                                 image_transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=0,
                              pin_memory=True, drop_last=False, shuffle=False)
    model = peak_response_mapping(
        backbone=fc_resnet50(), sub_pixel_locating_factor=8)
    model = model.cuda()
    model.load_state_dict(torch.load('./save/weights/peak_cls_train.pt'))

    results = []
    gt = []
    with torch.no_grad():
        for iter, pack in enumerate(tqdm(train_loader)):
            pass


def voc12_train_countset_cls(args):
    class_names = modules.pascal_voc_object_categories()

    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PascalVOCCount(
        json_to_pkl_file=args.json_to_pickle, transform=train_transform, args=args)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=0,
                              pin_memory=True, drop_last=False, shuffle=False)

    model = peak_response_mapping(
        backbone=fc_resnet50(), sub_pixel_locating_factor=8)
    model = model.cuda()
    model.load_state_dict(torch.load('./save/weights/peak_cls_train.pt'))

    results = []
    gt = []
    with torch.no_grad():
        for iter, pack in enumerate(tqdm(train_loader)):
            imgs = pack[0].cuda()
            labels = pack[1].cuda()

            aggregation = model.forward(imgs)
            results.append(aggregation.detach().cpu().numpy())
            gt.append(labels.cpu().numpy())

    results = np.concatenate(results, axis=0)
    gt = np.concatenate(gt, axis=0)
    results[results <= 0] = 0
    results[results > 0] = 1

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(
            results[:, i], gt[:, i])
        average_precision[i] = average_precision_score(
            results[:, i], gt[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(results.ravel(),
                                                                    gt.ravel())
    average_precision["micro"] = average_precision_score(results, gt,
                                                         average="micro")

    for i in range(len(class_names)):
        print('class:{} precision:{}'.format(class_names[i], precision[i]))
        print('class:{} recall:{}'.format(class_names[i], recall[i]))
        print('class:{} average_precision:{}'.format(
            class_names[i], average_precision[i]))
    print('avg precision:{} avg recall:{} avg average_precision:{}'.format(
        precision["micro"], recall["micro"], average_precision["micro"]))


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

    model = peak_response_mapping(
        backbone=fc_resnet50(), sub_pixel_locating_factor=8)
    model = model.cuda()
    model.load_state_dict(torch.load('./save/weights/peak_cls_train.pt'))
    model = model.inference()

    results = []
    gt = []
    for iter, pack in enumerate(tqdm(train_loader)):
        imgs = pack[0].cuda()
        labels = pack[1][0]
        cnt_labels = pack[1][1]

        aggregation, class_response_maps, valid_peak_list, peak_response_maps = model.forward(
            imgs)

        res = np.zeros(20)
        for l in valid_peak_list:
            res[l[1]] += 1
        results.append(res)
        gt.append(cnt_labels)

    with open('cnt.pkl', 'wb') as f:
        pickle.dump([results, gt], f)


if __name__ == '__main__':
    args = parse()
    # voc12_train_count(args)
    voc12_train_countset_cnt(args)
