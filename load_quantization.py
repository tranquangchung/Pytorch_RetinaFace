from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from data.data_augment_1 import preproc1
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace, QuantizedRetinaFace 
import copy
import pdb

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=15, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()
print(args)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
print(cfg)

# rgb_mean = (104, 117, 123) # bgr order
rgb_mean = (0, 0, 0) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = 5
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
# print(net)
# for name, param in net.named_parameters(): 
#     if 'body' in name:
#         param.requires_grad = False
pytorch_total_params = sum(p.numel() for p in net.parameters())
pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("pytorch_total_params", pytorch_total_params)
print("pytorch_total_params_trainable", pytorch_total_params_trainable)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=False)
    print("Load: ", args.resume_net)
    print("Load finish")


if __name__ == '__main__':
    # path_save = save_folder + cfg['name'] + '_Final_quantized_jit.pth'
    path_save = "./weights/lr_1e3_resize_image_rgb_relu/mobilenet0.25_Final_quantized_jit.pth"
    print(path_save)
    # path_save = "weights/lr_1e3_resize_image_rgb_relu/mobilenet0.25_Final_quantized_.pth"
    print("DONE")
    device='cpu'
    quantized_model_load = torch.jit.load(path_save, map_location=device)
    # print(quantized_model_load)
    print("LOAD DONE")
