import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch

from loader.dataset import pascal_voc_classification
from model.peak_net import peak_response_mapping
from model.backbone import fc_resnet50


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="peak_cls_train", type=str)
    # data
    parser.add_argument(
        "--voc12_root", default='/u/zkou2/Data/VOCdevkit', type=str)
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

    opt = optim.SGD([{'params': weight_list, 'lr': lr},
                     {'params': bias_list, 'lr': lr},
                     {'params': last_weight_list, 'lr': lr, 'weight_decay': 1e-4},
                     {'params': last_bias_list, 'lr': lr, 'weight_decay': 1e-4}], momentum=0.9)

    return opt


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

    loss_func = torch.nn.MultiLabelSoftMarginLoss()

    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        for iter, pack in enumerate(train_loader):
            imgs = pack[1].cuda()
            labels = pack[2].cuda()

            aggregation = model.forward(imgs)
            loss = loss_func(aggregation, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            


if __name__ == '__main__':
    args = parse()
    train(args)
