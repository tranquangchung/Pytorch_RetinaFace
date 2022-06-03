import time
import os
import copy
import argparse
import pdb
import collections
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

# import model_level_attention
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import utils_visual
import cv2
assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

ckpt =  False
def main(args=None):

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)

    parser.add_argument('--model_name', help='name of the model to save')
    parser.add_argument('--pretrained', help='pretrained model name')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Resizer(), Normalizer()]))
    
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model_pose_level_attention
    # if parser.depth == 18:
    #     retinanet = model_level_attention.resnet18(num_classes=1)
    # elif parser.depth == 34:
    #     retinanet = model_level_attention.resnet34(num_classes=1)
    # elif parser.depth == 50:
    #     retinanet = model_level_attention.resnet50(num_classes=1)
    # elif parser.depth == 101:
    #     retinanet = model_level_attention.resnet101(num_classes=1)
    # elif parser.depth == 152:
    #     retinanet = model_level_attention.resnet152(num_classes=1)
    # else:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # checkpoint = torch.load(parser.pretrained).state_dict()
    # retinanet.load_state_dict(checkpoint, strict=True)
    # retinanet = torch.nn.DataParallel(retinanet, device_ids=[0])
    # retinanet.eval()
    # if parser.csv_val is not None:
    #     print('Evaluating dataset')
    #     mAP = utils_visual.evaluate(dataset_val, retinanet)


if __name__ == '__main__':
    main()
