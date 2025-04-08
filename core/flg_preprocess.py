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
import sklearn.neighbors
import sklearn.mixture
import sklearn.gaussian_process
import h5py
import cupy as cp
import time

@dataclass
class Preprocessor(fls.BaseClass):

    # Loading
    pad_to_original_size = False

    # Resizing
    resize = False
    resize_target = 640

    # Scaling
    scale_percentile = False
    scale_percentile_value = 2.
    scale_percentile_clip = True

    scale_std = True

    return_uint8 = False

    @fls.profile_each_line
    def load_and_preprocess(self, data, desired_original_slices = None):

        data.load_to_memory(desired_slices = desired_original_slices, pad_to_original_size = self.pad_to_original_size)

        img_list = []
        for i in range(data.data.shape[0]):
            img_list.append(cp.array(data.data[i,:,:]).astype(cp.float16))
        img = cp.stack(img_list)
        print(img.shape)

        print('starting sync')
        cp.cuda.Device().synchronize()
        print('sync done')

        # Resize
        if self.resize:
            print('reconsider')
            import cupyx.scipy.ndimage
            #print(img.shape)
            data.resize_factor = min(self.resize_target/img.shape[1], self.resize_target/img.shape[2])
            #test_data = cupyx.scipy.ndimage.zoom(img[0,:,:], data.resize_factor)
            # data_new = cp.zeros((img.shape[0], test_data.shape[0], test_data.shape[1]), dtype=img.dtype)
            # for ii in range(img.shape[0]):
            #     data_new[ii,:,:] = cupyx.scipy.ndimage.zoom(img[ii,:,:], data.resize_factor)
            # img = data_new
            # test_data = cupyx.scipy.ndimage.zoom(img[:,0,0], data.resize_factor)
            # data_new = cp.zeros((test_data.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            # for ii in range(img.shape[1]):
            #     data_new[:,ii,:] = cupyx.scipy.ndimage.zoom(img[:,ii,:], (data.resize_factor,1.))
            img = cupyx.scipy.ndimage.zoom(img, (data.resize_factor,data.resize_factor,data.resize_factor))
            #print(img.shape)

            data.data_shape = img.shape
            data.voxel_spacing = data.voxel_spacing/data.resize_factor
        else:
            data.resize_factor = 1.

        # Scale percentile
        if self.scale_percentile:
            for ii in range(img.shape[0]):
                perc_low = cp.percentile(img[ii,:,:], self.scale_percentile_value)
                perc_high = cp.percentile(img[ii,:,:], 100-self.scale_percentile_value)
                img[ii,:,:] = (img[ii,:,:]-perc_low)/(perc_high-perc_low)
            if self.scale_percentile_clip:
                img[img>1.] = 1.
                img[img<0.] = 0.

        # Scale STD
        if self.scale_std:
            mean_per_slice = cp.mean(img,axis=(1,2))
            std_per_slice = cp.std(img.astype(cp.float32),axis=(1,2)).astype(cp.float16)
            img = (img - mean_per_slice[:,None,None]) / std_per_slice[:,None,None]
            #for ii in range(img.shape[0]):
            #    img[ii,:,:,] = (img[ii,:,:,]-mean_list[ii])/std_list[ii]

        # Cast to uint8
        if self.return_uint8:
            if self.scale_percentile:
                img = (255*img).astype(cp.uint8)
            else:
                img = (img).astype(cp.uint8)
            assert not self.scale_std

        print('starting sync')
        cp.cuda.Device().synchronize()
        print('sync done')
        print('back')
        data.data = np.empty(img.shape)
        for i in range(data.data.shape[0]):
            print(cp.mean(img[i,:,:]))
        for i in range(data.data.shape[0]):
            print(i)
            data.data[i,:,: ] =  cp.asnumpy(img_list[i])
        raise 'wrong'
    