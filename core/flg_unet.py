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

@dataclass(slots=True)
class DatasetTrain(torch.utils.data.IterableDataset):
    seed: int = field(init=True, default=0)

    # Selection settings
    n_positive: int = field(init=True, default=1)
    n_random: int = field(init=True, default=1)
    size: tuple = field(init=True, default=(128,128,128))    

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
        
        for i_set in itertools.count():     
            #t=time.time()
            # Determine which location in which tomogram we will use
            if i_set%(self.n_positive+self.n_random)<self.n_positive:
                # Center around a motor
                while True:
                    row = rng.integers(0,len(fls.all_train_labels))
                    if not fls.all_train_labels['z'][row]==-1:
                        break
                loc = np.argwhere([x==fls.all_train_labels['tomo_id'][row] for x in names])
                assert(loc.shape == (1,1))
                dataset = self.data_list[loc[0,0]]
                coords = [fls.all_train_labels['z'][row], fls.all_train_labels['y'][row], fls.all_train_labels['x'][row]]
                for i,c in enumerate(coords):
                    if c<self.size[i]//2: coords[i] = self.size[i]//2                    
                    if c>dataset.data.shape[i]-self.size[i]//2-1: coords[i] = dataset.data.shape[i]-self.size[i]//2-1
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
           
    

                         