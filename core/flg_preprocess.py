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

@dataclass
class Preprocessor(fls.BaseClass):

    # Loading
    pad_to_original_size = False

    # Resizing
    resize = False
    resize_value = 640

    # Scaling
    scale_percentile = False
    scale_percentile_value = 2.
    scale_percentile_clip = True

    scale_std = True

    return_uint8 = False

    def load_and_preprocess(self, data, desired_original_slices = slice(None,None,None)):

        data = copy.deepcopy(data)
        data.load_to_memory(desired_slices = desired_original_slices, pad_to_original_size = self.pad_to_original_size)

        fls.claim_gpu('cupy')
        img = cp.array(data.data, dtype=cp.float16)

        # Resize, also adjust voxel spacing!
        assert not self.resize
        data.resize_factor = 1.

        # Scale percentile
        assert not self.scale_percentile

        # Scale STD
        if self.scale_std:
            mean_per_slice = cp.mean(img,axis=(1,2))
            std_per_slice = cp.std(img.astype(cp.float32),axis=(1,2)).astype(cp.float16)
            img = (img - mean_per_slice[:,None,None]) / std_per_slice[:,None,None]
            #for ii in range(img.shape[0]):
            #    img[ii,:,:,] = (img[ii,:,:,]-mean_list[ii])/std_list[ii]
        
        # Cast to uint8
        assert not self.return_uint8

        data.data = cp.asnumpy(img)

        return data

    