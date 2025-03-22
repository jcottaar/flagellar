import pandas as pd
import numpy as np
import scipy as sp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import ndjson
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields
import enum
import typing
import zarr
import pathlib
import flg_support as fls
import sklearn.neighbors
import sklearn.mixture
import sklearn.gaussian_process

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

def collect_patches(data, sizes, normalize_slices = True):    
    collected = []
    is_edge = []
    
    for d in data:        
        for index,row in d.labels.iterrows():
            coords = np.array((row['z'], row['y'], row['x']))
            is_edge.append(not np.all(np.logical_and(coords >= sizes, coords <= np.array(np.shape(d.data)) - sizes - 1)))
            #to_append = copy.deepcopy(data_ext[coords[0]+3:coords[0]+2*sizes[0]+4,coords[1]+3:coords[1]+2*sizes[1]+4,coords[2]+3:coords[2]+2*sizes[2]+4])
            #to_append = to_append - np.mean(to_append)
            to_append = copy.deepcopy(extract_patch(d.data, coords, sizes, constant_value=np.nan))
            if normalize_slices:
                mean_list = extract_patch(d.mean_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
                std_list = extract_patch(d.mean_per_slice, coords[:1], sizes[:1], constant_value=np.nan)
                to_append = (to_append-mean_list[:,np.newaxis,np.newaxis])/std_list[:,np.newaxis,np.newaxis]
            collected.append( to_append )                     

    collected = np.stack(collected)
    is_edge = np.array(is_edge)
    return collected, is_edge
