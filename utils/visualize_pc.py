from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer, PointNetInpainting, PointNetDiscriminator
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PyTorchEMD.emd import earth_mover_distance
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from chamferdist import ChamferDistance
#from pytorch3d.loss import chamfer_distance
chamfer_dist = ChamferDistance()
# from show3d_balls import showpoints
writer = SummaryWriter('tensorboard/inpainting_GAN')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='../../ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0/', help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        inpainting=True,
        class_choice = ['Chair'],
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        inpainting=True,
        class_choice = ['Chair'],
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# exit()

# classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
classifier = PointNetInpainting(output=256, feature_transform=False)
epoch_r = 71
path = '%s/denseCls_feat_model_249.pth' % (opt.outf)
inp_path = '%s/inpaint_model_%d.pth' % (opt.outf, epoch_r)
# classifier.load_state_dict(torch.load(inp_path))
discriminator = PointNetDiscriminator()
# discriminator.loadFeat(path)


# if opt.model != '':
#     classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.RMSprop(classifier.parameters(), lr=0.0001)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=15, gamma=0.5)
# classifier.cuda()
# discriminator.cuda()



num_batch = len(dataset) / opt.batchSize
counter = 0;

i, data = next(enumerate(testdataloader))
points, target = data
target = target.transpose(2, 1)
points = points.transpose(2, 1)
# points, target = points.cuda(), target.cuda()
writer.add_graph(classifier, points, verbose=True)
writer.close()