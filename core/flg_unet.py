import pandas as pd
import numpy as np
import scipy as sp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields
import enum
import typing
import pathlib
import flg_support as fls
import flg_numerics
import sklearn.neighbors
import cupy as cp
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import cupyx.scipy.ndimage
import functools
import hashlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import time
import cc3d
import flg_unet_resnet18_3layer
import flg_unet_resnet34_3layer
import itertools
import monai

@dataclass(slots=True)
class DatasetTrain(torch.utils.data.IterableDataset):
    seed: object = field(init=True, default=None)

    # Selection settings
    n_positive: int = field(init=True, default=1)
    n_random: int = field(init=True, default=1)
    size: tuple = field(init=True, default=(128,128*3//2,128*3//2))    
    offset_range_for_pos: tuple = field(init=True, default=(32,64,64))

    # Transforms
    normalize: bool = field(init=True, default=True)

    # Target settings
    radius: float = field(init=True, default=200.) # in angstrom, not pixels!

    data_list: list = field(init=True, default_factory=list)
    

    #@fls.profile_each_line
    def __iter__(self): 
        rng = np.random.default_rng(seed=self.seed)
        for d in self.data_list:
            d.load_to_h5py()
        volumes = np.array([np.prod(d.data.shape) for d in self.data_list]).astype(np.float64)
        names = [d.name for d in self.data_list]     
        tl = []
        for d in self.data_list:
            tl.append(copy.deepcopy(d.labels))
            tl[-1]['tomo_id'] = d.name
        all_train_labels = pd.concat(tl, axis=0).reset_index()
        
        for i_set in itertools.count():     
            #t=time.time()
            # Determine which location in which tomogram we will use
            if i_set%(self.n_positive+self.n_random)<self.n_positive:
                # Center around a motor
                while True:
                    row = rng.integers(0,len(all_train_labels))
                    if not all_train_labels['z'][row]==-1:
                        break
                loc = np.argwhere([x==all_train_labels['tomo_id'][row] for x in names])
                assert(loc.shape == (1,1))
                dataset = self.data_list[loc[0,0]]
                coords = [all_train_labels['z'][row], all_train_labels['y'][row], all_train_labels['x'][row]]                
                for i in range(3):
                    coords[i] = coords[i] + rng.integers(-self.offset_range_for_pos[i], self.offset_range_for_pos[i])
                    if coords[i]<self.size[i]//2: coords[i] = self.size[i]//2                    
                    if coords[i]>dataset.data.shape[i]-self.size[i]//2-1: coords[i] = dataset.data.shape[i]-self.size[i]//2-1
                #print(dataset.name,dataset.data.shape,coords)
            else:
                # Pick at random
                #print('rand')
                index = rng.choice(range(len(volumes)), p=volumes/np.sum(volumes))
                dataset = self.data_list[index]
                coords = []
                for i in range(3):
                    coords.append(rng.integers(self.size[i]//2, dataset.data.shape[i]-self.size[i]//2-1))
                #print(dataset.name,dataset.data.shape,coords)

            #print('1', t-time.time())
            # Pick the tomogram data            
            slices = []
            for i in range(3):
                slices.append(slice(coords[i]-self.size[i]//2, coords[i]+self.size[i]//2))
            image = dataset.data[tuple(slices)][...].astype(np.float32)
            if self.normalize:
                mean_list = dataset.mean_per_slice[slices[0]]
                std_list = dataset.std_per_slice[slices[0]]
                for ii in range(image.shape[0]):
                    image[ii,:,:,] = (image[ii,:,:,]-mean_list[i])/std_list[i]
            assert image.shape == self.size

            #print('2', t-time.time())

            # Construct target
            target = np.zeros_like(image, dtype=np.float32)
            radius_pix = self.radius/dataset.voxel_spacing
            mask_size = np.ceil(radius_pix).astype(int)+2            
            inds = np.arange(-mask_size, mask_size+1)
            xx,yy,zz = np.meshgrid(inds, inds, inds, indexing="ij")             
            mask = np.zeros_like(zz, dtype=np.float32)
            mask[np.sqrt(zz**2+yy**2+xx**2) < radius_pix] = 1.
            for row in range(len(dataset.labels)):
                offset = np.array([dataset.labels['z'][row], dataset.labels['y'][row], dataset.labels['x'][row]]) - coords + np.array(self.size)//2
                flg_numerics.add_matrix_with_offset(target,mask,offset)               
            target[target>1]=1

            #print('3', t-time.time())

            image = torch.tensor(image, dtype=torch.float16)
            target = torch.tensor(target, dtype=torch.float16)

            #print('4', t-time.time())
            yield image, target


@dataclass
class UNetModel(fls.BaseClass):
    # Data management
    dataset: object = field(init=True, default_factory=DatasetTrain)

    # Learning rate
    learning_rate = 1e-3
    n_images_per_update = 10
    n_epochs = 100

    # Loss
    tversky_alpha = 1.
    tversky_beta = 1.

    # Other
    seed = None
    verbose = False
    deterministic_train = False

    # Trained
    model = 0

    # Diagnostics
    train_loss_list1: list = field(init=True, default_factory=list)
    train_loss_list2: list = field(init=True, default_factory=list)
    
    

    def train(self,train_data):
        #TODO: half precision, mix losses, scheduler, augments, ensemble
        cpu,device = fls.prep_pytorch(self.seed, self.deterministic_train, True)  

        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = monai.losses.TverskyLoss(smooth_nr=1e-05, smooth_dr=1e-05, batch=True, to_onehot_y=False, sigmoid=True, \
                                                 alpha=self.tversky_alpha, beta=self.tversky_beta)

        model = flg_unet_resnet18_3layer.UNetResNet18_3D(num_classes=1)

        for module in model.modules():  # Recursively iterate through all submodules
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = 0.01
                module.affine = False

        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)   

        self.dataset.data_list = copy.deepcopy(train_data)
        if not self.seed is None:
            self.dataset.seed = self.seed+1
        data_loader = iter(torch.utils.data.DataLoader(self.dataset,batch_size=self.n_images_per_update,num_workers=0,pin_memory=True,persistent_workers=True))

        scaler = torch.amp.GradScaler('cuda')

        for i_epoch in range(self.n_epochs):
            print(i_epoch)
            running_loss1 = 0.0
            running_loss2 = 0.0
            images, targets = next(data_loader)
            with torch.amp.autocast('cuda'):
                N=4
                for i_image in range(images.shape[0]//N):
                    image_device = images[N*i_image:N*i_image+N,np.newaxis,:,:,:].to(device, dtype=torch.float16, non_blocking=True)
                    target_device = targets[N*i_image:N*i_image+N,np.newaxis,:,:,:].to(device, dtype=torch.float16, non_blocking=True)
                    output = model(image_device)  
                    # for ii in range(N):
                    #     loss1 = criterion1(output[ii,:,:,:,:], target_device[ii,:,:,:,:])
                    #     loss2 = criterion2(output[ii,:,:,:,:], target_device[ii,:,:,:,:])   
                    #     print(loss1.item(),loss2.item())
                    #     #print(loss1.item(), loss2.item())
                    #     running_loss1 += loss1.detach()/N
                    #     running_loss2 += loss2.detach()/N
                    loss1 = criterion1(output, target_device)
                    loss2 = criterion2(output, target_device)  
                    running_loss1 += loss1.detach()
                    running_loss2 += loss2.detach()

                    # alt_loss = 0
                    # for ii in range(N):
                    #     #loss2x = criterion1(output[ii,:,:,:,:], target_device[ii,:,:,:,:])
                    #     loss2x = criterion2(output[ii,:,:,:,:], target_device[ii,:,:,:,:])   
                    #     #print(loss1.item(),loss2.item())
                    #     #print(loss1.item(), loss2.item())
                    #     alt_loss += loss2x.detach()/N
                    # print(loss2.item(),alt_loss.item())

                    

                    
    
                    loss = 0.004*loss1 + loss2
    
                    scaler.scale(N*loss/images.shape[0]).backward()

            epoch_loss1 = N*running_loss1.item() / images.shape[0]
            epoch_loss2 = N*running_loss2.item() / images.shape[0]            
            print(epoch_loss1, epoch_loss2)
            #loss.backward()
            #optimizer.step()            
            scaler.step(optimizer)
            scaler.update()
            
            self.train_loss_list1.append(epoch_loss1)
            self.train_loss_list2.append(epoch_loss2)

            optimizer.zero_grad()

        model.to(cpu)
        model.eval()
        self.model = model
                

        

                         