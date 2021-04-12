from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset, ModelNetDataset
from model import PointNetCls, feature_transform_regularizer, PointNetInpainting
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import open3d


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
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

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        print("pc type:: ", points.type())
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred,fine, trans, trans_feat = classifier(points)
        
        loss = classifier.create_loss(pred,fine,target,0.5)
        loss.backward()
        optimizer.step()
        
        #loss = nn.MSELoss(pred, target)
        #if opt.feature_transform:
            #loss += feature_transform_regularizer(trans_feat) * 0.001
        #loss.backward()
        #optimizer.step()
        #pred_choice = pred.data.max(1)[1]
        #correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred,fine, _, _ = classifier(points)
            
            loss = classifier.create_loss(pred,fine,target,0.5)
            
            #loss = nn.MSELoss(pred, target)
            #pred_choice = pred.data.max(1)[1]
            #correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    coarse, fine, _, _ = classifier(points)
    
    print(coarse.shape,fine.shape,target.shape)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(fine.data.cpu().numpy()[0]+np.array([1.0,0.0,0.0]))
    pcd.colors = open3d.Vector3dVector(np.ones((fine.shape[1],3))* [0.76,0.23,0.14])

    pcd2 = open3d.PointCloud()
    pcd2.points = open3d.Vector3dVector(target.data.cpu().numpy()[0]+np.array([-1.0,0.0,0.0]))
    pcd2.colors = open3d.Vector3dVector(np.ones((target.shape[1],3))* [0.16,0.23,0.14])

    pcd3 = open3d.PointCloud()
    pcd3.points = open3d.Vector3dVector(points.data.cpu().numpy()[0])
    pcd3.colors = open3d.Vector3dVector(np.ones((points.shape[1],3))* [0.16,0.23,0.14])
    
    open3d.draw_geometries([pcd,pcd2,pcd3])
    
    
    #pred_choice = pred.data.max(1)[1]
    #correct = pred_choice.eq(target.data).cpu().sum()
    #total_correct += correct.item()
    #total_testset += points.size()[0]

#print("final accuracy {}".format(total_correct / float(total_testset)))