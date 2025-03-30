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
import gc
import h5py
import contextlib

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
    reproduce_voxel_bug: bool = field(init=True, default=False)

    # Augments
    rotate_xy_90 = 0.5
    rotate_xy_180 = 0.5
    rotate_xz_180 = 0.5
    flip_x = 0.5

    # Other
    return_float32 = False
    

    data_list: list = field(init=True, default_factory=list)

    def _preprocess(self, image, mean_list, std_list, percentile_list):
        if self.normalize:
            for ii in range(image.shape[0]):
                image[ii,:,:,] = (image[ii,:,:,]-mean_list[ii])/std_list[ii]
    

    #@fls.profile_each_line
    def __iter__(self): 
        @fls.profile_each_line
        def get_next():
            #t=time.time()
            # Determine which location in which tomogram we will use
            if i_set%(self.n_positive+self.n_random)<self.n_positive:
                # Center around a motor
                while True:
                    row = rng.integers(0,len(all_train_labels))
                    if not all_train_labels['z'][row]==-1:
                        break
                loc = all_train_labels['ind'][row]
                #assert(loc.shape == (1,1))
                dataset = self.data_list[loc]
                coords = [int(all_train_labels['z'][row]*dataset.resize_factor), int(all_train_labels['y'][row]*dataset.resize_factor), int(all_train_labels['x'][row]*dataset.resize_factor)]                
                for i in range(3):
                    coords[i] = coords[i] + rng.integers(-self.offset_range_for_pos[i], self.offset_range_for_pos[i])
                    if coords[i]<self.size[i]//2: coords[i] = self.size[i]//2                    
                    if coords[i]>dataset.data_shape[i]-self.size[i]//2-1: coords[i] = dataset.data_shape[i]-self.size[i]//2-1
                #print(dataset.name,dataset.data_shape,coords)
            else:
                # Pick at random
                #print('rand')
                index = rng.choice(range(len(volumes)), p=volumes/np.sum(volumes))
                dataset = self.data_list[index]
                coords = []
                for i in range(3):
                    coords.append(rng.integers(self.size[i]//2, dataset.data_shape[i]-self.size[i]//2-1))
                #print(dataset.name,dataset.data_shape,coords)

            #print(dataset.name)
            #print('1', t-time.time())
            # Pick the tomogram data            
            slices = []
            for i in range(3):
                slices.append(slice(coords[i]-self.size[i]//2, coords[i]+self.size[i]//2))
            with h5py.File(dataset.data) as f:
                image = f['data'][tuple(slices)][...].astype(np.float32)
            if self.normalize:
                mean_list = dataset.mean_per_slice[slices[0]]
                std_list = dataset.std_per_slice[slices[0]]
                for ii in range(image.shape[0]):
                    image[ii,:,:,] = (image[ii,:,:,]-mean_list[ii])/std_list[ii]
            assert image.shape == self.size

            #print('2', t-time.time())

            # Construct target
            target = np.zeros_like(image, dtype=bool)
            if self.reproduce_voxel_bug:
                voxel_spacing = 6.5
            else:
                voxel_spacing = dataset.voxel_spacing
            radius_pix = self.radius/voxel_spacing*dataset.resize_factor
            mask_size = np.ceil(radius_pix).astype(int)+2            
            inds = np.arange(-mask_size, mask_size+1)
            xx,yy,zz = np.meshgrid(inds, inds, inds, indexing="ij")             
            mask = np.zeros_like(zz, dtype=bool)
            mask[np.sqrt(zz**2+yy**2+xx**2) < radius_pix] = True
            for row in range(len(dataset.labels)):
                offset = np.array([int(dataset.labels['z'][row]*dataset.resize_factor), int(dataset.labels['y'][row]*dataset.resize_factor), int(dataset.labels['x'][row]*dataset.resize_factor)]) - coords + np.array(self.size)//2
                flg_numerics.or_matrix_with_offset(target,mask,offset)               
            #target[target>1]=1

            # Augment
            augment_functions = []
            if rng.uniform()<self.rotate_xy_90:
                augment_functions.append(lambda x:np.rot90(x, axes=(1,2)))
            if rng.uniform()<self.rotate_xy_180:
                augment_functions.append(lambda x:np.rot90(x, k=2, axes=(1,2)))
            if rng.uniform()<self.rotate_xz_180:
                augment_functions.append(lambda x:np.rot90(x, k=2, axes=(0,2)))
            if rng.uniform()<self.flip_x:
                augment_functions.append(lambda x:np.flip(x, axis=2))
            for f in augment_functions:
                image = f(image)
                target = f(target)
            

            #print('3', t-time.time())

            if self.return_float32:
                image = torch.tensor(image.copy(), dtype=torch.float32)
            else:
                image = torch.tensor(image.copy(), dtype=torch.float16)
            target = torch.tensor(target.copy(), dtype=torch.bool)

            #print('4', t-time.time())
            return image, target
                
        rng = np.random.default_rng(seed=self.seed)
        for d in self.data_list:
            d.load_to_h5py()
        volumes = np.array([np.prod(d.data_shape) for d in self.data_list]).astype(np.float64)
        names = [d.name for d in self.data_list]     
        tl = []
        for (i,d) in enumerate(self.data_list):
            tl.append(copy.deepcopy(d.labels))
            tl[-1]['tomo_id'] = d.name
            tl[-1]['ind'] = i
        all_train_labels = pd.concat(tl, axis=0).reset_index()        
        
        for i_set in itertools.count():     
            yield get_next()


@dataclass
class UNetModel(fls.BaseClass):
    # Data management
    dataset: object = field(init=True, default_factory=DatasetTrain)

    # Learning rate
    learning_rate = 1e-3
    n_images_per_update = 20
    n_epochs = 2000

    # Loss
    tversky_alpha = 1.
    tversky_beta = 1.
    entropy_weight = 0.

    # Inference
    infer_size: tuple = field(init=True, default=(256,256,256))
    infer_overlap = 64

    # Other
    seed = None
    verbose = False
    deterministic_train = False
    save_model_every = 100
    plot_every = 500
    n_images_test = 100
    test_loss_every = 10

    # Trained
    model = 0

    # Diagnostics
    train_loss_list1: list = field(init=True, default_factory=list)
    train_loss_list2: list = field(init=True, default_factory=list)
    test_loss_epochs: list = field(init=True, default_factory=list)
    test_loss_list2: list = field(init=True, default_factory=list)
    
    def train(self,train_data,validation_data):
        #TODO: scheduler, ensemble, entropy weighting
        print(self.seed)
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

        dataset_test = copy.deepcopy(self.dataset)
        dataset_test.return_float32 = self.deterministic_train
        dataset_test.data_list = copy.deepcopy(validation_data)
        data_loader_test = iter(torch.utils.data.DataLoader(dataset_test,batch_size=self.n_images_test,num_workers=1,pin_memory=True,persistent_workers=True))

        self.dataset.data_list = copy.deepcopy(train_data)
        self.dataset.return_float32 = self.deterministic_train
        if not self.seed is None:
            self.dataset.seed = self.seed+1
        data_loader = iter(torch.utils.data.DataLoader(self.dataset,batch_size=self.n_images_per_update,num_workers=1,pin_memory=True,persistent_workers=True))

        scaler = torch.amp.GradScaler('cuda')

        print('alter')
        device_type = torch.float32 if self.deterministic_train else torch.float16
        mixed_precision_context = contextlib.nullcontext() if self.deterministic_train else torch.amp.autocast('cuda')

        for i_epoch in range(self.n_epochs):
            print(i_epoch, end=' ')
            running_loss1 = 0.0
            running_loss2 = 0.0
            images, targets = next(data_loader)
            with mixed_precision_context:
                N=self.dataset.n_positive + self.dataset.n_random
                for i_image in range(images.shape[0]//N):
                    image_device = images[N*i_image:N*i_image+N,np.newaxis,:,:,:].to(device, dtype=device_type, non_blocking=True)
                    target_device = targets[N*i_image:N*i_image+N,np.newaxis,:,:,:].to(device, dtype=device_type, non_blocking=True)
                    output = model(image_device)  
                    loss1 = criterion1(output, target_device)
                    loss2 = criterion2(output, target_device)  
                    running_loss1 += loss1.detach()
                    running_loss2 += loss2.detach()
                    loss = self.entropy_weight*loss1 + loss2
                    scaler.scale(N*loss/images.shape[0]).backward()


            epoch_loss1 = N*running_loss1.item() / images.shape[0]
            epoch_loss2 = N*running_loss2.item() / images.shape[0]            
            del images
            del targets
            #print(epoch_loss1, epoch_loss2)
            #loss.backward()
            #optimizer.step()            
            scaler.step(optimizer)
            scaler.update()
            
            self.train_loss_list1.append(epoch_loss1)
            self.train_loss_list2.append(epoch_loss2)

            optimizer.zero_grad()

            if (i_epoch+1)%self.test_loss_every==0:
                running_loss1 = 0.0
                running_loss2 = 0.0
                images, targets = next(data_loader_test)                
                with torch.no_grad(), mixed_precision_context:
                    N=self.dataset.n_positive + self.dataset.n_random
                    for i_image in range(images.shape[0]//N):
                        image_device = images[N*i_image:N*i_image+N,np.newaxis,:,:,:].to(device, dtype=torch.float16, non_blocking=True)
                        target_device = targets[N*i_image:N*i_image+N,np.newaxis,:,:,:].to(device, dtype=torch.float16, non_blocking=True)
                        output = model(image_device)  
                        loss1 = criterion1(output, target_device)
                        loss2 = criterion2(output, target_device)  
                        running_loss1 += loss1.detach()
                        running_loss2 += loss2.detach()              

                
                epoch_loss1 = N*running_loss1.item() / images.shape[0]
                epoch_loss2 = N*running_loss2.item() / images.shape[0]  
                self.test_loss_epochs.append(i_epoch)
                self.test_loss_list2.append(epoch_loss2)
                del images
                del targets

            if not self.save_model_every is None and (i_epoch+1)%self.save_model_every==0:
                model_save = copy.deepcopy(model)
                model_save.to(cpu)
                model_save.eval()
                fls.dill_save(fls.temp_dir + 'intermediate_model_' + str(i_epoch+1) + '.pickle', model_save)

            if (i_epoch+1)%self.plot_every==0 or i_epoch+1 == self.n_epochs:
                plt.figure()
                plt.plot(self.train_loss_list1)
                plt.plot(self.train_loss_list2)
                plt.plot(self.test_loss_epochs, self.test_loss_list2)                
                plt.grid(True)
                plt.pause(0.1)

        model.to(cpu)
        model.eval()
        self.model = model

        

    @fls.profile_each_line
    def infer(self, data):
        #TODO: TTA

        cpu,device = fls.prep_pytorch(self.seed, True, False)

        # Prepare data and output
        image = torch.tensor(data.data, dtype=torch.float16).to(device)
        image = image[None,None,:,:,:]
        pad_list = []
        for dim in [2,1,0]:
            pad_list.append(0)
            pad_list.append(max(0,self.infer_size[dim]-image.shape[dim+2]))
        image = F.pad(image, pad_list, mode="constant", value=0)
        if self.dataset.normalize:
            for ii in range(image.shape[0]):
                image[ii,:,:] = (image[ii,:,:]-data.mean_per_slice[ii])/data.std_per_slice[ii]        
        combined_probablity_map = torch.zeros((image.shape[2], image.shape[3], image.shape[4]),dtype=torch.float16).to(device)
        self.model.to(device)
        self.model.eval()

        def find_ranges(total_size, batch_size_infer, infer_edge_size):
            assert total_size>=batch_size_infer
            ranges = [((0,batch_size_infer), (0,batch_size_infer-infer_edge_size), (0,batch_size_infer-infer_edge_size))]
            # part of input to use - part of result to use - where to insert in probablity_map
            cur_start = 0
            while True:
                cur_start = cur_start + batch_size_infer - 2*infer_edge_size
                cur_end = cur_start + batch_size_infer
                if cur_end>=total_size:
                    break
                ranges.append( ((cur_start, cur_end),  (infer_edge_size,batch_size_infer-infer_edge_size), (cur_start+infer_edge_size, cur_end-infer_edge_size)) )                
            ranges.append( ((total_size-batch_size_infer, total_size), (infer_edge_size,batch_size_infer), (total_size-batch_size_infer+infer_edge_size,total_size)) )
            return ranges

        ranges_z = find_ranges(image.shape[2], self.infer_size[0], self.infer_overlap)
        ranges_y = find_ranges(image.shape[3], self.infer_size[1], self.infer_overlap)
        ranges_x = find_ranges(image.shape[4], self.infer_size[2], self.infer_overlap)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for rx in ranges_x:
                for ry in ranges_y:
                    for rz in ranges_z:
                        #print(rx,ry,rz)
                        res = self.model(image[:1,:1,rz[0][0]:rz[0][1],ry[0][0]:ry[0][1],rx[0][0]:rx[0][1]])                        
                        combined_probablity_map[rz[2][0]:rz[2][1],ry[2][0]:ry[2][1],rx[2][0]:rx[2][1]] = \
                            res[0,0,rz[1][0]:rz[1][1],ry[1][0]:ry[1][1],rx[1][0]:rx[1][1]]

        self.model.to(cpu)

        del image
        gc.collect()
        return combined_probablity_map.to(torch.float32).detach().cpu().numpy()

        
                

        

                         