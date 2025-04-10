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
import flg_unet
import flg_yolo2
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
import itertools
import monai

@dataclass
class HeatMapToLocations(fls.BaseClass):
    threshold = 0.
    submission_logit_offset = 0.

    # Trained
    state = 0 # 0 is uncalibrated, 1 is calibrated

    #@fls.profile_each_line
    def make_labels(self, heatmap, data):
        rows_list = []
        
        mask = heatmap>self.threshold
        connected = cc3d.connected_components(mask, connectivity=26)           
        del mask
        stats = cc3d.statistics(connected) 
        for cluster_ind in np.arange(1,stats['voxel_counts'].shape[0]): # skip 0 = background cluster 
            # Construct a local map
            bounding_box = stats['bounding_boxes'][cluster_ind]
            local_connected =connected[bounding_box]
            local_heatmap =  copy.copy(heatmap[bounding_box])
            to_keep = local_connected==cluster_ind
            assert np.sum(to_keep)>0
            local_heatmap[np.logical_not(to_keep)]=-np.inf

            centroid = stats['centroids'][cluster_ind]
            to_add = dict()
            to_add['z'] = np.round(centroid[0])/data.resize_factor
            to_add['y'] = np.round(centroid[1])/data.resize_factor
            to_add['x'] = np.round(centroid[2])/data.resize_factor
            to_add['size'] = stats['voxel_counts'][cluster_ind]
            to_add['max_logit'] = np.max(local_heatmap)
            if fls.is_submission:
                to_add['max_logit'] = to_add['max_logit'] + self.submission_logit_offset
            rows_list.append(to_add)

        labels = pd.DataFrame(rows_list)
        if len(labels)==0:
            labels = pd.DataFrame(columns=['z', 'y', 'x', 'size', 'max_logit'])
        #print(labels)

        if self.state==1:
            raise 'todo'
        return labels

@dataclass
class SelectSingleMotors(fls.BaseClass):
    threshold: float = field(init=True, default = 0.)
    x_val = 'size'
    y_val = 'max_logit'
    vals: object = field(init=True, default_factory = lambda:np.linspace(-10,100,100))

    def select_motors(self,d):
        d.labels = copy.deepcopy(d.labels_unfiltered)
        d.labels = d.labels[d.labels[self.y_val]>self.threshold]
        if len(d.labels)>0:
            row = np.argmax(d.labels[self.y_val].to_numpy())
            d.labels = d.labels[row:row+1]        

    def calibrate(self, data, reference_data):
        cs_tp = []
        cs_fp = []
        log_tp = []
        log_fp = []
        tp = 0
        for t in data:
            has_tp = False
            for d in range(len(t.labels_unfiltered)):
                #print( t.labels_unfiltered['tf_pn']
                if t.labels_unfiltered['tf_pn'][d]==0:
                    cs_tp.append(t.labels_unfiltered[self.x_val][d])
                    log_tp.append(t.labels_unfiltered[self.y_val][d])
                    has_tp = True
                if t.labels_unfiltered['tf_pn'][d]==1:
                    cs_fp.append(t.labels_unfiltered[self.x_val][d])
                    log_fp.append(t.labels_unfiltered[self.y_val][d])
            if has_tp:
                tp += 1
        plt.figure(figsize=(18,18))
        print('Number of true positives before filter: ', tp)
        plt.scatter(cs_fp, log_fp, color='red')
        plt.scatter(cs_tp, log_tp, color='blue')        
        plt.grid(True)
        plt.pause(0.01)
        
        thresholds_try = self.vals
        scores = []
        for t in thresholds_try:
            data_try = copy.deepcopy(data)
            self.threshold = t
            for d in data_try:
                self.select_motors(d)
            scores.append(fls.score_competition_metric(data_try, reference_data))
            
        plt.figure()
        plt.plot(thresholds_try,scores)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.grid(True)
        self.threshold = thresholds_try[np.argmax(scores)]
        plt.pause(0.01)
        

        
        
@dataclass
class ThreeStepModel(fls.Model):
    # Runs modeling in 3 steps:
    # 1) generate heatmap
    # 2) create labels from heatmap, with probability
    # 3) select a subset as output (optionally one per tomogram)
    step1Heatmap: object = field(init=True, default_factory=flg_unet.UNetModel)
    step2Labels: object = field(init=True, default_factory=HeatMapToLocations)
    step3Output: object = field(init=True, default_factory=SelectSingleMotors)

    # Intermediate
    data_after_step2 = 0

    # Internal
    run_to: int = field(init=True, default=0) # 0: run all, 1: stop after creating labels

    TEMP_threshold = 20.

    def _train(self, train_data, validation_data):
        self.step1Heatmap.seed = self.seed
        self.step1Heatmap.preprocessor = self.preprocessor
        if self.step1Heatmap.model==0:
            self.step1Heatmap.train(train_data, validation_data)

        self_temp = copy.deepcopy(self)
        self_temp.run_to = 1
        self_temp.state = 1
        self.data_after_step2 = self_temp.infer(validation_data)
        fls.mark_tf_pn(self.data_after_step2, validation_data)

        self.step3Output.calibrate(self.data_after_step2, validation_data)

    
    def _infer_single(self,data):   
        if not self.data_after_step2 == 0:
            prev_names = [d.name for d in self.data_after_step2]
        if self.data_after_step2 == 0 or not data.name in prev_names:
            heatmap = self.step1Heatmap.infer(data)
            print('resize factor:', data.resize_factor)
            data.labels_unfiltered = self.step2Labels.make_labels(heatmap, data)
        else:
            for d in self.data_after_step2:
                if d.name == data.name:
                    data.labels_unfiltered = d.labels_unfiltered

        if self.run_to==0:
            self.step3Output.select_motors(data)

        # print(data.labels)
        # if not fls.is_submission:
        #     plt.figure()
        #     plt.imshow(np.max(heatmap, axis=0), cmap='bone')
        #     plt.colorbar()
        #     #plt.clim([0,0.1])
        #     plt.title(data.name)
        return data

@dataclass
class TwoStepModel(fls.Model):
    # Runs modeling in 2 steps:
    # 1) create labels from data
    # 2) select a subset as output (optionally one per tomogram)
    step1Labels: object = field(init=True, default_factory=flg_yolo2.YOLOModel)
    step2Output: object = field(init=True, default_factory=SelectSingleMotors)

    calibrate_step_2: bool = field(init=True, default=False)
    submission_threshold_offset: float = field(init=True, default=0.)
    # Intermediate
    data_after_step1 = 0

    # Internal
    run_to: int = field(init=True, default=0) # 0: run all, 1: stop after creating labels

    def __post_init__(self):
        super().__post_init__()
        self.step2Output.x_val = 'confidence'
        self.step2Output.y_val = 'confidence'
        self.step2Output.vals = np.linspace(0,1,1000)

    def _train(self, train_data, validation_data):
        self.step1Labels.seed = self.seed
        if self.step1Labels.trained_model==0:
            self.step1Labels.train(train_data, validation_data)

        if self.calibrate_step_2:
            self_temp = copy.deepcopy(self)
            self_temp.run_to = 1
            self_temp.state = 1
            self.data_after_step1 = self_temp.infer(validation_data)
            fls.mark_tf_pn(self.data_after_step1, validation_data)    
            self.step2Output.calibrate(self.data_after_step1, validation_data)

    
    def _infer_single(self,data):   
        if not self.data_after_step1 == 0:
            prev_names = [d.name for d in self.data_after_step1]
        if self.data_after_step1 == 0 or not data.name in prev_names:
            data.labels_unfiltered = self.step1Labels.infer(data)
        else:
            for d in self.data_after_step1:
                if d.name == data.name:
                    data.labels_unfiltered = d.labels_unfiltered

        #if fls.is_submission:
        data.labels_unfiltered['confidence'] = data.labels_unfiltered['confidence']+self.submission_threshold_offset
        if self.run_to==0:
            self.step2Output.select_motors(data)

        # print(data.labels)
        # if not fls.is_submission:
        #     plt.figure()
        #     plt.imshow(np.max(heatmap, axis=0), cmap='bone')
        #     plt.colorbar()
        #     #plt.clim([0,0.1])
        #     plt.title(data.name)
        return data



