import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from loader.dataset import pascal_voc_classification
from model.peak_net import peak_response_mapping
from model.backbone import fc_resnet50


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="peak_cls_train", type=str)
    # parser.add_argument("--session_name", default="train_with_center_loss", type=str)
    # data
    parser.add_argument(
        "--voc12_root", default='/mnt/lustre/jiangsu/dlar/home/zyk17/data/VOCdevkit', type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    # config
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    # train
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    # save
    parser.add_argument("--save_weights", default='save/weights', type=str)

    args = parser.parse_args()

    return args


def get_finetune_optimizer(args, model, epoch):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if 'classifier' in name:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr': lr/100},
                     {'params': bias_list, 'lr': lr/100},
                     {'params': last_weight_list, 'lr': lr},
                     {'params': last_bias_list, 'lr': lr}], momentum=0.9, lr=lr, weight_decay=1e-4, nesterov=False)

    return opt


def multilabel_soft_margin_loss(
        input: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        size_average: bool = True,
        reduce: bool = True,
        difficult_samples: bool = False) -> Tensor:
    """Multilabel soft margin loss.
    """

    if difficult_samples:
        # label 1: positive samples
        # label 0: difficult samples
        # label -1: negative samples
        gt_label = target.clone()
        gt_label[gt_label == 0] = 1
        gt_label[gt_label == -1] = 0
    else:
        gt_label = target

    return F.multilabel_soft_margin_loss(input, gt_label, weight, size_average, reduce)


def train(args):
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = pascal_voc_classification(
        split='trainval', data_dir=args.voc12_root, year=2012, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False, shuffle=True)

    model = peak_response_mapping(
        backbone=fc_resnet50(), sub_pixel_locating_factor=8)
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        for iter, pack in enumerate(train_loader):
            imgs = pack[1].cuda()
            labels = pack[2].cuda()

            aggregation = model.forward(imgs)
            loss = multilabel_soft_margin_loss(
                input=aggregation, target=labels, difficult_samples=True)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if iter % 50 == 0:
                print('epoch:{} iter:{} loss:{}'.format(epoch, iter, loss))

    torch.save(model.module.state_dict(), os.path.join(
        args.save_weights, args.session_name+'40.pt'))

def train_with_center_loss(args):
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = pascal_voc_classification(
        split='trainval', data_dir=args.voc12_root, year=2012, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False, shuffle=True)

    model = peak_response_mapping(
        backbone=fc_resnet50(), sub_pixel_locating_factor=8)
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        for iter, pack in enumerate(train_loader):
            imgs = pack[1].cuda()
            labels = pack[2].cuda()

            aggregation = model.forward(imgs)
            loss = multilabel_soft_margin_loss(
                input=aggregation, target=labels, difficult_samples=True)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if iter % 50 == 0:
                print('epoch:{} iter:{} loss:{}'.format(epoch, iter, loss))

    torch.save(model.module.state_dict(), os.path.join(
        args.save_weights, args.session_name+'.pt'))


if __name__ == '__main__':
    args = parse()
    train(args)
