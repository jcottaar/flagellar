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
import gc
import flg_numerics

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
    blur_xy = 0.
    blur_z = 1

    # Other
    invert_sign = False

    return_uint8 = False

    #@fls.profile_each_line
    def load_and_preprocess(self, data, desired_original_slices = None, allow_missing=False):

        data.load_to_memory(desired_slices = desired_original_slices, pad_to_original_size = self.pad_to_original_size, allow_missing=allow_missing)

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

        # Blur
        blur_matrix = cp.ones((self.blur_z,1), dtype=cp.float16)/self.blur_z
        import cupyx.scipy.signal
        import cupyx.scipy.ndimage
        for ii in range(img.shape[2]):
            img[:,:,ii] = cupyx.scipy.signal.fftconvolve(img[:,:,ii], blur_matrix, mode='same')
        for ii in range(img.shape[0]):
            img[ii,:,:] = cupyx.scipy.ndimage.gaussian_filter(img[ii,:,:], sigma = self.blur_xy)


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
                    moving_std = np.sqrt(moving_mean_of_squared - moving_mean**2)
                    img[ii,...] = ((arr - moving_mean)/moving_std)
                    
                    

        # Scale STD
        if self.scale_std:
            mean_per_slice = cp.mean(img,axis=(1,2))
            std_per_slice = cp.std(img,axis=(1,2)).astype(cp.float16)
            for ii in range(img.shape[0]):
                img[ii,...] = (img[ii,...] - mean_per_slice[ii,None,None]) / std_per_slice[ii,None,None]                
                if not np.isnan(self.scale_std_clip_value):
                    #print(cp.min(img[ii,:,:]), cp.max(img[ii,:,:]))
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
                img *= 255
                img = img.astype(cp.uint8)
            else:
                img = (img).astype(cp.uint8)

        if self.invert_sign:
            img = cp.max(img)-img

        data.data = cp.asnumpy(img)

class Preprocessor2(fls.BaseClass):

    # Loading
    pad_to_original_size = False
    voxel_scale = 1.

    # Percentile scaling    
    scale_percentile_value = 3.

    # Resizing
    target_voxel_spacing = 20. # Angstrom

    # Blurring
    blur_xy = 30. # Angstrom
    blur_z = 0. # Angstrom

    # Average/STD scaling
    scale_moving_average = True
    scale_moving_average_size = 3000.
    
    scale_moving_std = True
    scale_moving_std_size = 3000.
    blur_xy_moving_std = 60. # Angstrom

    # Augmentation
    apply_transpose = False
    apply_flipud = False
    apply_fliplr = False

    clip_value = 3.

    #@fls.profile_each_line
    def load_and_preprocess(self, data, desired_original_slices = None, allow_missing=False):

        data.load_to_memory(desired_slices = desired_original_slices, pad_to_original_size = self.pad_to_original_size, allow_missing=allow_missing)

        # Guess voxel spacing if not provided
        if np.isnan(data.voxel_spacing):
            xy_size = np.sqrt(data.data.shape[1]*data.data.shape[2])
            data.voxel_spacing = ((-8.71223429e-03)*xy_size + 2.33859781e+01)
            print('Guessed voxel spacing: ', data.voxel_spacing)
        data.voxel_spacing *= self.voxel_scale

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
        for ii in range(img.shape[0]):
            perc_low = cp.percentile(img[ii,:,:], self.scale_percentile_value)
            perc_high = cp.percentile(img[ii,:,:], 100-self.scale_percentile_value)
            img[ii,:,:] = (img[ii,:,:]-perc_low)/(perc_high-perc_low)
            img[ii,:,:] = cp.clip(img[ii,:,:], 0., 1.)

        # Resize
        # print(img.shape, data.voxel_spacing)
        # plt.figure()
        # plt.imshow(cp.asnumpy(img[6,:,:]), cmap='bone')
        # plt.colorbar()
        data.resize_factor = data.voxel_spacing/self.target_voxel_spacing
        target_shape = tuple(np.round(np.array( (img.shape[1], img.shape[2]) )*data.resize_factor/2 ).astype(int)*2)
        old_img_list = [img[ii,:,:] for ii in range(img.shape[0])]
        del img        
        #gc.collect()
        img = cp.zeros( (len(old_img_list), target_shape[0], target_shape[1]), dtype=cp.float32 )
        for ii in range(img.shape[0]):
            n_y = old_img_list[ii].shape[0]
            if n_y%2 == 1:
                n_y = n_y-1
            n_x = old_img_list[ii].shape[1]
            if n_x%2 == 1:
                n_x = n_x-1
            img[ii,:,:] = flg_numerics.fourier_resample_nd(old_img_list[ii][:n_y,:n_x], target_shape)
            img[ii,:,:] = cp.clip(img[ii,:,:], 0., 1.)
        if len(data.labels)>0:
            data.labels['x'] *= data.resize_factor
            data.labels['y'] *= data.resize_factor
        if len(data.negative_labels)>0:
            data.negative_labels['x'] *= data.resize_factor
            data.negative_labels['y'] *= data.resize_factor
        # plt.figure()
        # plt.imshow(cp.asnumpy(img[6,:,:]), cmap='bone')
        # plt.colorbar()
        # print(target_shape, img.shape)

        # print(img.shape)

        import cupyx.scipy.signal
        import cupyx.scipy.ndimage

        # Blur
        for ii in range(img.shape[0]):
            img[ii,:,:] = cupyx.scipy.ndimage.gaussian_filter(img[ii,:,:], sigma = self.blur_xy/self.target_voxel_spacing)        
        for ii in range(img.shape[2]):
            img[:,:,ii] = cupyx.scipy.ndimage.gaussian_filter(img[:,:,ii], sigma = (self.blur_z/data.voxel_spacing,0))
            
        # Moving average scaling
        if self.scale_moving_average:
            moving_size = min(np.round(self.scale_moving_average_size / self.target_voxel_spacing).astype(int), min(img.shape[1]-4, img.shape[2]-4))
            pad_size = moving_size//2
            conv_matrix = cp.ones((moving_size,moving_size), dtype=cp.float32)
            conv_matrix = conv_matrix/np.sum(conv_matrix)
            for ii in range(img.shape[0]):
                arr = img[ii,...]
                #print(arr.shape, conv_matrix.shape)
                moving_mean = cupyx.scipy.signal.fftconvolve(arr, conv_matrix, mode='same')  
                #print(moving_mean.shape)
                assert cp.max(moving_mean)>0
                #print(pad_size)
                moving_mean = moving_mean[pad_size:-pad_size,pad_size:-pad_size]
                #print(moving_mean.shape)
                moving_mean = cp.pad(moving_mean, ((pad_size,pad_size),(pad_size,pad_size)), mode='edge')
                img[ii,...] = (img[ii,...] - moving_mean)                
        else:
            for ii in range(img.shape[0]):
                img[ii,:,:] = img[ii,:,:] - np.mean(img[ii,:,:])

        # plt.figure()
        # plt.imshow(cp.asnumpy(moving_mean), cmap='bone')
        # plt.colorbar()
        # #print(cp.min(img), cp.max(img))
        # plt.figure()
        # plt.imshow(cp.asnumpy(img[6,:,:]), cmap='bone')
        # plt.colorbar()
        
        # Moving STD scaling
        if self.scale_moving_std:
            moving_size = min(np.round(self.scale_moving_std_size / self.target_voxel_spacing).astype(int), min(img.shape[1]-4, img.shape[2]-4))
            #print(moving_size)
            pad_size = moving_size//2
            conv_matrix = cp.ones((moving_size,moving_size), dtype=cp.float32)
            conv_matrix = conv_matrix/np.sum(conv_matrix)
            for ii in range(img.shape[0]):
                arr = cupyx.scipy.ndimage.gaussian_filter(img[ii,...], sigma = self.blur_xy_moving_std/self.target_voxel_spacing)        
                moving_mean = cupyx.scipy.signal.fftconvolve(arr, conv_matrix, mode='same')  
                assert cp.max(moving_mean)>0
                moving_mean = moving_mean[pad_size:-pad_size,pad_size:-pad_size]
                moving_mean = cp.pad(moving_mean, ((pad_size,pad_size),(pad_size,pad_size)), mode='edge')

                moving_mean_of_squared = cupyx.scipy.signal.fftconvolve(arr**2, conv_matrix, mode='same')            
                assert cp.max(moving_mean_of_squared)>0
                moving_mean_of_squared = moving_mean_of_squared[pad_size:-pad_size,pad_size:-pad_size]
                moving_mean_of_squared = cp.pad(moving_mean_of_squared, ((pad_size,pad_size),(pad_size,pad_size)), mode='edge')
                moving_std = np.sqrt(moving_mean_of_squared - moving_mean**2)
                
                img[ii,...] = img[ii,...]/moving_std
                #print(np.mean(moving_std), np.std(arr))
        else:
            for ii in range(img.shape[0]):
                img[ii,:,:] = img[ii,:,:]/np.std(img[ii,:,:])

        # plt.figure()
        # plt.imshow(cp.asnumpy(img[6,:,:]), cmap='bone')
        # plt.colorbar()
       
        # print(target_shape, img.shape)
        for ii in range(img.shape[0]):
            img[ii,...] = (img[ii,...]+self.clip_value)/(2*self.clip_value)
            img[ii,...] = cp.clip(img[ii,...], 0., 1.)     

        if self.apply_transpose:
            img = cp.transpose(img, axes=(0,2,1))
            if len(data.labels)>0:
                tmp = data.labels['x'].to_numpy()
                data.labels['x'] = data.labels['y'].to_numpy()
                data.labels['y'] = tmp
            if len(data.negative_labels)>0:
                tmp = data.negative_labels['x'].to_numpy()
                data.negative_labels['x'] = data.negative_labels['y'].to_numpy()
                data.negative_labels['y'] = tmp            
        if self.apply_flipud:
            img = cp.flip(img, axis=1)
            if len(data.labels)>0:
                data.labels['y'] = img.shape[1]-data.labels['y']
            if len(data.negative_labels)>0:
                data.negative_labels['y'] = img.shape[1]-data.negative_labels['y']
        if self.apply_fliplr:
            img = cp.flip(img, axis=2)
            if len(data.labels)>0:
                data.labels['x'] = img.shape[2]-data.labels['x']
            if len(data.negative_labels)>0:
                data.negative_labels['x'] = img.shape[2]-data.negative_labels['x']

        # plt.figure()
        # plt.imshow(cp.asnumpy(img[6,:,:]), cmap='bone')
        # plt.colorbar()
        
        img = img*255
        img = img.astype(cp.uint8)

        data.data = cp.asnumpy(img)
