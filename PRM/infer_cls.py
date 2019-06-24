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
import warnings
import argparse
import pdb
from torchvision import datasets, transforms
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="peak_cls_train", type=str)
    # data
    parser.add_argument(
        "--voc12_root", default='/u/zkou2/Data/VOCdevkit', type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    # config
    parser.add_argument("--batch_size", default=16, type=int)

    # save
    parser.add_argument("--save_weights", default='save/weights', type=str)

    args = parser.parse_args()

    return args


def random_color():
    _HEX = '0123456789ABCDEF'

    return '#' + ''.join(random.choice(_HEX) for _ in range(6))


def plot_points(raw_img, valid_peak_list, class_names):
    plt.imshow(raw_img)
    plt.savefig('raw_img.png')
    plt.close()

    plt.figure(figsize=(10, 10))

    peaks = valid_peak_list.cpu().numpy()
    classes = list(set([x[1] for x in peaks]))
    class2color = dict()
    for c in classes:
        class2color[c] = random_color()

    for p in peaks:
        plt.scatter([p[3]], [p[2]], s=75, c=class2color[p[1]],
                    label='{}'.format(class_names[p[1]]))

    plt.xticks(np.arange(0, 14, 1))
    plt.yticks(np.arange(0, 14, 1))

    plt.savefig('tmp_img.png')
    plt.close()


def sample():
    # run_tasks('./config.yml')
    class_names = modules.pascal_voc_object_categories()

    image_size = 448
    # image pre-processor
    transformer = modules.image_transform(

        image_size=[image_size, image_size],
        augmentation=dict(),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    backbone = modules.fc_resnet50(num_classes=20, pretrained=False)
    model = modules.peak_response_mapping(backbone)

    state = torch.load('./save/weights/peak_cls_train.pt')
    # new_dict=dict()
    # for k,v in state['model'].items():
    #   new_dict[k[7:]]=v
    model.load_state_dict(state)
    model = model.cuda()

    idx = 0
    raw_img = PIL.Image.open('./data/sample%d.jpg' % idx).convert('RGB')
    input_var = transformer(raw_img).unsqueeze(0).cuda().requires_grad_()
    with open('./data/sample%d.json' % idx, 'r') as f:
        proposals = list(map(modules.rle_decode, json.load(f)))

    model = model.eval()
    confidence = model(input_var)
    for idx in range(len(class_names)):
        if confidence.data[0, idx] > 0:
            print('    [class_idx: %d] %s (%.2f)' %
                  (idx, class_names[idx], confidence[0, idx]))

    model = model.inference()
    aggregation, class_response_maps, valid_peak_list, peak_response_maps = model(
        input_var)
    plot_points(raw_img, valid_peak_list, class_names)


def train_infer(args):
	class_names = modules.pascal_voc_object_categories()

    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = pascal_voc_classification(
        split='trainval', data_dir=args.voc12_root, year=2012, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0,
                              pin_memory=True, drop_last=False, shuffle=True)

    model = peak_response_mapping(
        backbone=fc_resnet50(), sub_pixel_locating_factor=8)
    model = model.cuda()
    model.load_state_dict(torch.load('./save/weights/peak_cls_train.pt'))

    for iter, pack in enumerate(train_loader):
        imgs = pack[1].cuda()
        labels = pack[2].cuda()

        aggregation = model.forward(imgs)
        pass


# CUDA_VISIBLE_DEVICES=0,1 python -m pdb main.py
# scipy 1.2.0 python 3.6.6
if __name__ == '__main__':
    args = parse()
    train_infer(args)
