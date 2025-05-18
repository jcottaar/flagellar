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
import math

def extract_patch(matrix, center, patch_size, constant_value=0):
    """
    Extracts a patch of size `patch_size` centered at `center` from a nD numpy array `matrix`.
    If the patch extends beyond the boundaries of `matrix`, the missing values are filled with
    `constant_value`.

    Parameters:
      matrix         : nD numpy array from which to extract the patch.
      center         : Tuple of n integers (z, y, x) specifying the center of the patch.
      patch_size     : Tuple of n integers specifying the desired patch size.
      constant_value : Value to fill in for out-of-bound regions (default is 0).

    Returns:
      patch          : A new 3D numpy array of shape `patch_size` containing the extracted patch.
    """
    # Create an output patch filled with the constant value.
    patch = np.full(2*patch_size+1, constant_value, dtype=np.float64)
    
    # Prepare slices for each dimension.
    matrix_slices = []
    patch_slices = []
    
    for i in range(len(matrix.shape)):
        # Calculate where the patch would start and end in the coordinate system of matrix.
        start = center[i] - patch_size[i]
        end = center[i] + patch_size[i] + 1
        
        # Determine the valid indices that overlap with the matrix.
        m_start = max(start, 0)
        m_end = min(end, matrix.shape[i])
        
        # Calculate corresponding indices in the patch array.
        p_start = m_start - start
        p_end = p_start + (m_end - m_start)
        
        matrix_slices.append(slice(m_start, m_end))
        patch_slices.append(slice(p_start, p_end))
    
    # Copy the overlapping region from the matrix to the patch.
    patch[tuple(patch_slices)] = matrix[tuple(matrix_slices)].astype(np.float64)
    
    return patch

def collect_patches_neg(data, sizes, preprocessor, n_collect):
    neg_labels = fls.dill_load(fls.code_dir + '/neg_labels.pickle')
    data = fls.load_all_train_data()
    collected = []
    is_edge = []
    for i_row in range(n_collect):
        name = neg_labels['name'][i_row]
        for d in data:
            if d.name == name:
                break
        else:
            raise 'Not found'
        if len(d.labels)>0:
            continue
        dd = copy.deepcopy(d)
        coords = np.array((np.round(neg_labels['z'][i_row]).astype(int), np.round(neg_labels['y'][i_row]).astype(int), np.round(neg_labels['x'][i_row]).astype(int)))
        print(coords)
        desired_slices = np.arange(coords[0]-sizes[0], coords[0]+sizes[0]+1)
        desired_slices = desired_slices[desired_slices>=0]
        desired_slices = desired_slices[desired_slices<dd.data_shape[0]]
        preprocessor.load_and_preprocess(dd, desired_original_slices=list(desired_slices))
        coords[0] = len(dd.slices_present)//2
        coords[1] = np.round(coords[1]*dd.resize_factor).astype(int)
        coords[2] = np.round(coords[2]*dd.resize_factor).astype(int)
        is_edge = np.nan
        to_append = copy.deepcopy(extract_patch(dd.data, coords, sizes, constant_value=np.nan))        
        collected.append( to_append )  
        # print('x')
        # print(neg_labels[i_row:i_row+1])
        # print(d.labels)

    collected = np.stack(collected)
    is_edge = np.array(is_edge)
    return collected, is_edge

def collect_patches_xz_transposed(data, sizes, preprocessor):    
    collected = []
    is_edge = []
    
    for d in data:   
        dd = copy.deepcopy(d)
        if len(d.labels)>0:
            preprocessor.apply_transpose_xz = True
            preprocessor.load_and_preprocess(dd)
            print(dd.data.shape)
        for index,row in d.labels.iterrows():
            
            coords = np.array((np.round(row['x']).astype(int), np.round(row['y']).astype(int), np.round(row['z']).astype(int)))
            print(coords)
            #desired_slices = np.arange(coords[0]-sizes[0], coords[0]+sizes[0]+1)
            #desired_slices = desired_slices[desired_slices>=0]
            #desired_slices = desired_slices[desired_slices<dd.data_shape[0]]
            #dd.load_to_memory(desired_slices=list(desired_slices))    
            
            coords[0] = np.round(coords[0]).astype(int)
            coords[1] = np.round(coords[1]*dd.resize_factor).astype(int)
            coords[2] = np.round(coords[2]*dd.resize_factor).astype(int)
            #is_edge.append(not np.all(np.logical_and(coords >= sizes, coords <= np.array(np.shape(f['data'])) - sizes - 1)))
            is_edge = np.nan # todo
            #to_append = copy.deepcopy(data_ext[coords[0]+3:coords[0]+2*sizes[0]+4,coords[1]+3:coords[1]+2*sizes[1]+4,coords[2]+3:coords[2]+2*sizes[2]+4])
            #to_append = to_append - np.mean(to_append)
               
            to_append = copy.deepcopy(extract_patch(dd.data, coords, sizes, constant_value=np.nan))            
            # if normalize_slices:
            #      mean_list = np.nanmean(to_append, axis=(1,2))#extract_patch(d.mean_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
            #      std_list = np.nanstd(to_append, axis=(1,2))#extract_patch(d.std_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
            #      to_append = (to_append-mean_list[:,np.newaxis,np.newaxis])/std_list[:,np.newaxis,np.newaxis]
            collected.append( to_append )                     

    collected = np.stack(collected)
    is_edge = np.array(is_edge)
    return collected, is_edge

def collect_patches_yz_transposed(data, sizes, preprocessor):    
    collected = []
    is_edge = []
    
    for d in data:   
        dd = copy.deepcopy(d)
        if len(d.labels)>0:
            preprocessor.apply_transpose_yz = True
            preprocessor.load_and_preprocess(dd)
            print(dd.data.shape)
        for index,row in d.labels.iterrows():
            
            coords = np.array((np.round(row['y']).astype(int), np.round(row['z']).astype(int), np.round(row['x']).astype(int)))
            print(coords)
            #desired_slices = np.arange(coords[0]-sizes[0], coords[0]+sizes[0]+1)
            #desired_slices = desired_slices[desired_slices>=0]
            #desired_slices = desired_slices[desired_slices<dd.data_shape[0]]
            #dd.load_to_memory(desired_slices=list(desired_slices))    
            
            coords[0] = np.round(coords[0]).astype(int)
            coords[1] = np.round(coords[1]*dd.resize_factor).astype(int)
            coords[2] = np.round(coords[2]*dd.resize_factor).astype(int)
            #is_edge.append(not np.all(np.logical_and(coords >= sizes, coords <= np.array(np.shape(f['data'])) - sizes - 1)))
            is_edge = np.nan # todo
            #to_append = copy.deepcopy(data_ext[coords[0]+3:coords[0]+2*sizes[0]+4,coords[1]+3:coords[1]+2*sizes[1]+4,coords[2]+3:coords[2]+2*sizes[2]+4])
            #to_append = to_append - np.mean(to_append)
               
            to_append = copy.deepcopy(extract_patch(dd.data, coords, sizes, constant_value=np.nan))            
            # if normalize_slices:
            #      mean_list = np.nanmean(to_append, axis=(1,2))#extract_patch(d.mean_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
            #      std_list = np.nanstd(to_append, axis=(1,2))#extract_patch(d.std_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
            #      to_append = (to_append-mean_list[:,np.newaxis,np.newaxis])/std_list[:,np.newaxis,np.newaxis]
            collected.append( to_append )                     

    collected = np.stack(collected)
    is_edge = np.array(is_edge)
    return collected, is_edge

def collect_patches(data, sizes, preprocessor):    
    collected = []
    is_edge = []
    
    for d in data:   
        for index,row in d.labels.iterrows():
            dd = copy.deepcopy(d)
            coords = np.array((np.round(row['z']).astype(int), np.round(row['y']).astype(int), np.round(row['x']).astype(int)))
            print(coords)
            desired_slices = np.arange(coords[0]-sizes[0], coords[0]+sizes[0]+1)
            desired_slices = desired_slices[desired_slices>=0]
            desired_slices = desired_slices[desired_slices<dd.data_shape[0]]
            #dd.load_to_memory(desired_slices=list(desired_slices))            
            preprocessor.load_and_preprocess(dd, desired_original_slices=list(desired_slices), allow_missing = True)
            coords[0] = len(dd.slices_present)//2
            coords[1] = np.round(coords[1]*dd.resize_factor).astype(int)
            coords[2] = np.round(coords[2]*dd.resize_factor).astype(int)
            #is_edge.append(not np.all(np.logical_and(coords >= sizes, coords <= np.array(np.shape(f['data'])) - sizes - 1)))
            is_edge = np.nan # todo
            #to_append = copy.deepcopy(data_ext[coords[0]+3:coords[0]+2*sizes[0]+4,coords[1]+3:coords[1]+2*sizes[1]+4,coords[2]+3:coords[2]+2*sizes[2]+4])
            #to_append = to_append - np.mean(to_append)
               
            to_append = copy.deepcopy(extract_patch(dd.data, coords, sizes, constant_value=np.nan))            
            # if normalize_slices:
            #      mean_list = np.nanmean(to_append, axis=(1,2))#extract_patch(d.mean_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
            #      std_list = np.nanstd(to_append, axis=(1,2))#extract_patch(d.std_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
            #      to_append = (to_append-mean_list[:,np.newaxis,np.newaxis])/std_list[:,np.newaxis,np.newaxis]
            collected.append( to_append )                     

    collected = np.stack(collected)
    is_edge = np.array(is_edge)
    return collected, is_edge


def or_matrix_with_offset(A,B,offset):
    # Adds B to A, with the center of B at index offset
    # 3D matrices
    # Elements of B that end up outside A are ignored
    # Works in place

    # Calculate the start and end indices for A
    z_start = max(0, offset[0] - B.shape[0] // 2)
    y_start = max(0, offset[1] - B.shape[1] // 2)
    x_start = max(0, offset[2] - B.shape[2] // 2)
    
    z_end = min(A.shape[0], offset[0] + B.shape[0] // 2 + 1)
    y_end = min(A.shape[1], offset[1] + B.shape[1] // 2 + 1)
    x_end = min(A.shape[2], offset[2] + B.shape[2] // 2 + 1)

    if z_end<z_start or y_end<y_start or x_end<x_start:
        return
    
    # Calculate the corresponding start and end indices for B
    bz_start = max(0, B.shape[0] // 2 - offset[0])
    by_start = max(0, B.shape[1] // 2 - offset[1])
    bx_start = max(0, B.shape[2] // 2 - offset[2])
    
    bz_end = bz_start + (z_end - z_start)
    by_end = by_start + (y_end - y_start)
    bx_end = bx_start + (x_end - x_start)

    # Add B to A at the specified offset
    A[z_start:z_end, y_start:y_end, x_start:x_end] = np.logical_or(A[z_start:z_end, y_start:y_end, x_start:x_end],B[bz_start:bz_end, by_start:by_end, bx_start:bx_end])

import cupy as cp
from cupyx.scipy.fft import fftn, ifftn, fftshift, ifftshift

def fourier_resample_nd(x: cp.ndarray, new_shape: tuple) -> cp.ndarray:
    """
    Fourier-based N-D resampling of a real or complex CuPy array x to shape `new_shape`.
    
    Parameters
    ----------
    x : cp.ndarray
        Input array of shape old_shape.
    new_shape : tuple of ints
        Desired output shape, same number of dims as x.ndim.
    
    Returns
    -------
    y : cp.ndarray
        Resampled array of shape new_shape, same dtype as x (real outputs if x was real).
    """
    old_shape = x.shape
    if len(old_shape) != len(new_shape):
        raise ValueError("new_shape must have same number of dimensions as x")

    # 1) Forward FFT
    Xf = fftn(x)
    del x
    # 2) Shift zero-frequency to center
    Xf_shift = fftshift(Xf)
    del Xf

    # 3) Prepare target freq array
    Yf_shift = cp.zeros(new_shape, dtype=Xf_shift.dtype)

    # 4) Compute slices for each axis and copy overlap
    src_slices = []
    dst_slices = []
    for old_n, new_n in zip(old_shape, new_shape):
        min_n = min(old_n, new_n)
        # source start/end
        src_start = (old_n - min_n) // 2
        src_end   = src_start + min_n
        # dest start/end
        dst_start = (new_n - min_n) // 2
        dst_end   = dst_start + min_n
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    # Copy the low-frequency region
    Yf_shift[tuple(dst_slices)] = Xf_shift[tuple(src_slices)]
    del Xf_shift

    # 5) Inverse shift and inverse FFT
    Yf = ifftshift(Yf_shift)
    del Yf_shift
    y  = ifftn(Yf)
    del Yf

    # Normalize amplitude
    norm_factor = math.prod(new_shape) / math.prod(old_shape)
    y = y * norm_factor

    # 6) Return real part if input was real
    return cp.real(y)