from loader.det_dataset import PascalVOCDetection
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings("ignore")


def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument("--session_name", default="peak_cls_train", type=str)
	# data
	parser.add_argument(
		"--voc12_root", default='/u/zkou2/Data/VOCdevkit/VOC2012', type=str)
	parser.add_argument("--crop_size", default=448, type=int)
	# config
	parser.add_argument("--batch_size", default=16, type=int)

	# save
	parser.add_argument("--save_weights", default='save/weights', type=str)

	args = parser.parse_args()

	return args


def voc12_train_count(args):
	train_transform = transforms.Compose([
		transforms.Resize((448, 448)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	dataset = PascalVOCDetection(images_folder_path='/u/zkou2/Data/VOCdevkit/VOC2012',
								 annotation_json='/u/zkou2/Data/VOCdevkit/PASCAL_VOC_JSON/pascal_train2012.json',
								 image_transform=train_transform)
	train_loader = DataLoader(dataset, batch_size=16, num_workers=0,
							  pin_memory=True, drop_last=False, shuffle=True)
	model = peak_response_mapping(
		backbone=fc_resnet50(), sub_pixel_locating_factor=8)
	model = model.cuda()
	model.load_state_dict(torch.load('./save/weights/peak_cls_train.pt'))

	results = []
	gt = []
	with torch.no_grad():
		for iter, pack in enumerate(tqdm(train_loader)):
			pass


if __name__ == '__main__':
	args = parse()
	voc12_train_count(args)
