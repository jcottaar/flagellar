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
    scale_std_clip_value = 3.

    scale_moving_average = False
    scale_also_moving_std = False
    moving_ratio = 0.2

    # Blurring
    blur_z = 1

    # Other
    invert_sign = False

    return_uint8 = False

    #@fls.profile_each_line
    def load_and_preprocess(self, data, desired_original_slices = None):

        data.load_to_memory(desired_slices = desired_original_slices, pad_to_original_size = self.pad_to_original_size)

        fls.claim_gpu('cupy')
        while True:
            try:
                img = cp.array(data.data).astype(cp.float32)
                break
            except:
                fls.claim_gpu('')
                time.sleep(1)
                fls.claim_gpu('cupy')
                print('failed cupy')
                pass

        # Scale percentile
        if self.scale_percentile:
            for ii in range(img.shape[0]):
                perc_low = cp.percentile(img[ii,:,:], self.scale_percentile_value)
                perc_high = cp.percentile(img[ii,:,:], 100-self.scale_percentile_value)
                img[ii,:,:] = (img[ii,:,:]-perc_low)/(perc_high-perc_low)
                if self.scale_percentile_clip:
                    img[ii,:,:] = cp.clip(img[ii,:,:], 0., 1.)

        blur_matrix = cp.ones((self.blur_z,1), dtype=cp.float16)/self.blur_z
        import cupyx.scipy.signal
        for ii in range(img.shape[2]):
            img[:,:,ii] = cupyx.scipy.signal.fftconvolve(img[:,:,ii], blur_matrix, mode='same')


        # Moving average/STD scaling
        if self.scale_moving_average:
            moving_size = np.round(np.sqrt(img.shape[1]*img.shape[2])*self.moving_ratio).astype(int)
            pad_size = moving_size//2
            #print(moving_size)
            conv_matrix = cp.ones((moving_size,moving_size), dtype=cp.float32)
            conv_matrix = conv_matrix/np.sum(conv_matrix)
            for ii in range(img.shape[0]):
                arr = img[ii,...]
                moving_mean = cupyx.scipy.signal.fftconvolve(arr, conv_matrix, mode='same')  
                assert cp.max(moving_mean)>0
                moving_mean = moving_mean[pad_size:-pad_size,pad_size:-pad_size]
                moving_mean = cp.pad(moving_mean, ((pad_size,pad_size),(pad_size,pad_size)), mode='edge')
                if not self.scale_also_moving_std:
                    img[ii,...] = (arr - moving_mean)
                else:
                    moving_mean_of_squared = cupyx.scipy.signal.fftconvolve(arr**2, conv_matrix, mode='same')            
                    assert cp.max(moving_mean_of_squared)>0
                    moving_mean_of_squared = moving_mean_of_squared[pad_size:-pad_size,pad_size:-pad_size]
                    moving_mean_of_squared = cp.pad(moving_mean_of_squared, ((pad_size,pad_size),(pad_size,pad_size)), mode='edge')
                    moving_std = moving_mean_of_squared - moving_mean**2
                    img[ii,...] = ((arr - moving_mean)/moving_std)
                    
                    

        # Scale STD
        if self.scale_std:
            mean_per_slice = cp.mean(img,axis=(1,2))
            std_per_slice = cp.std(img,axis=(1,2)).astype(cp.float16)
            for ii in range(img.shape[0]):
                img[ii,...] = (img[ii,...] - mean_per_slice[ii,None,None]) / std_per_slice[ii,None,None]
                if not np.isnan(self.scale_std_clip_value):
                    img[ii,...] = (img[ii,...]+self.scale_std_clip_value)/(2*self.scale_std_clip_value)
                    img[ii,...] = cp.clip(img[ii,...], 0., 1.)     
            #for ii in range(img.shape[0]):
            #    img[ii,:,:,] = (img[ii,:,:,]-mean_list[ii])/std_list[ii]
                
        # Resize
        if self.resize:
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
        
        # Cast to uint8
        if self.return_uint8:
            if self.scale_percentile:
                img = (255*img).astype(cp.uint8)
            else:
                img = (img).astype(cp.uint8)

        if self.invert_sign:
            img = cp.max(img)-img

        data.data = cp.asnumpy(img)

    