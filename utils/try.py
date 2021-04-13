import torch
from chamferdist import ChamferDistance

source_cloud = torch.randn(5, 10, 3).cuda()
target_cloud = torch.randn(5, 10, 3).cuda()

chamferDist = ChamferDistance()


dist = []


for i in range(source_cloud.shape[0]):
    
    #dist_i = chamferDist(source_cloud[i,:,:], target_cloud[i,:,:])
    #mean_dist = torch.mean(dist)
    #num_256 = source_cloud.shape[1]
    #num_3 = source_cloud.shape[2]
    
    #torch.reshape(cloud,(1,num_256,num_3))
    #cloud.unsqueeze(0)
    #cloud = torch.randn(1, 10, 3)
    #cloud.view(1,10,3)
    
    cloud = source_cloud[i,:,:]
    cloud.resize_(1,10,3)
    tar = target_cloud[i,:,:]
    tar.resize_(1,10,3)
    dist_i = chamferDist(cloud,tar)
    
    
    #print(cloud.shape)
    #print(source_cloud.shape)
    print(dist_i)
    dist.append(dist_i)
loss = sum(dist)/len(dist)
print(loss)