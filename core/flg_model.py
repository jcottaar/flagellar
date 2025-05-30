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
    select_on_ratio = False
    vals: object = field(init=True, default_factory = lambda:np.linspace(-10,100,100))

    # Diagnostics
    precisions = 0
    recalls = 0
    scores = 0

    def select_motors(self,d):
        d.labels = copy.deepcopy(d.labels_unfiltered)
        if self.select_on_ratio:
            if len(d.labels)>1:
                all_vals = np.sort(d.labels[self.y_val])
                val = (all_vals[-1]-all_vals[-2])/all_vals[-1]
            else:
                val = 1
            if val<self.threshold:
                d.labels = d.labels[0:0]
        else:
            d.labels = d.labels[d.labels[self.y_val]>self.threshold]            
            val = np.max(d.labels[self.y_val])
        if len(d.labels)>0:
            row = np.argmax(d.labels[self.y_val].to_numpy())
            d.labels = d.labels[row:row+1].reset_index()
            d.labels.at[0, 'value'] = val

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
        print('Number of true positives before filter: ', tp, ' out of ', np.sum([len(d.labels) for d in reference_data]))
        plt.scatter(cs_fp, log_fp, color='red')
        plt.scatter(cs_tp, log_tp, color='blue')        
        plt.grid(True)
        plt.pause(0.01)
        
        thresholds_try = self.vals
        self.precisions = []
        self.recalls = []
        self.scores = []
        data_try_base = copy.deepcopy(data)
        for d in data_try_base:
            d.labels_unfiltered2 = []
        for t in thresholds_try:
            data_try = copy.deepcopy(data_try_base)
            self.threshold = t
            for d in data_try:
                self.select_motors(d)
            score = fls.score_competition_metric(data_try, reference_data)
            self.precisions.append(score[0])
            self.recalls.append(score[1])
            self.scores.append(score[2])
            
        plt.figure()
        plt.plot(thresholds_try,self.precisions)
        plt.plot(thresholds_try,self.recalls)
        plt.plot(thresholds_try,self.scores)
        plt.xlabel('Threshold')
        plt.ylabel('Value')
        plt.legend(['Precision', 'Recall', 'Score'])
        plt.grid(True)
        self.threshold = thresholds_try[np.argmax(self.scores)]
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
class SimpleHeatmapModel(fls.Model):
    step1Heatmap: object = field(init=True, default_factory=flg_unet.UNetModel)
    threshold: float = field(init=True, default=-np.inf)

    def _train(self, train_data, validation_data):
        self.step1Heatmap.seed = self.seed
        self.step1Heatmap.preprocessor = self.preprocessor
        if self.step1Heatmap.model==0:
            self.step1Heatmap.train(train_data, validation_data)

    def _infer_single(self,data):   
        heatmap = self.step1Heatmap.infer(data)
        if np.max(heatmap)>self.threshold:
            max_position = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            d = {'z':max_position[0], 'y':max_position[1], 'x':max_position[2]}
            data.labels = pd.DataFrame([d])
        else:
            data.labels = pd.DataFrame(columns=['z', 'y', 'x'])
        return data
            
@dataclass
class FindClusters(fls.BaseClass):
    distance_threshold = 10.
    min_confidence = np.nan
    n_models = np.nan
    max_n_detections = np.inf

    def cluster(self, data):
        """
        Perform 3D Non-Maximum Suppression on detections to merge nearby motors.
        """       
        detections = data.labels_unfiltered2.to_dict('records')
        if not detections:
            return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model'])
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        if len(detections)>self.max_n_detections:
            detections = detections[:self.max_n_detections]            
    
        zyx = []
        for d in detections:
            zyx.append([d['z'], d['y'], d['x']])
        zyx = np.array(zyx)
    
        import sklearn.cluster
        clustering = sklearn.cluster.DBSCAN(eps=self.distance_threshold, min_samples=1).fit(zyx)
    
        detections = pd.DataFrame(detections)
        final_detections = []
        for lab in np.unique(clustering.labels_):
            this_detections = detections[clustering.labels_==lab].reset_index()
            #print('this')
            #print(this_detections)
            #print('------')
            conf_per_model = []
            #print('------------------')
            for i_model in range(self.n_models):
                this_detections_this_model = this_detections[this_detections['i_model']==i_model]
                #print(this_detections_this_model)
                if len(this_detections_this_model)==0:
                    conf_per_model.append(self.min_confidence)
                else:
                    conf_per_model.append(np.max(this_detections_this_model['confidence']))
           # print('------------------')
            conf = np.mean(conf_per_model)
            final_detections.append({'z':this_detections['z'][0], 'y':this_detections['y'][0], 'x':this_detections['x'][0], 'confidence':conf})
            #print('FINAL')
            #print(final_detections)
    
        if len(final_detections)==0:
            print('FINAL')
            print(pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model']))
            print('')
            return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model'])
        else:
            print('FINAL')
            print(pd.DataFrame(final_detections))
            print('')
            return pd.DataFrame(final_detections)


@dataclass
class FindClustersMultiZ(fls.BaseClass):
    distance_threshold = 10.
    min_confidence = np.nan
    n_models = np.nan
    z_range = 1e-10
    #std_factor = 0.
    gamma = 1.

    def cluster(self, data):
        """
        Perform 3D Non-Maximum Suppression on detections to merge nearby motors.
        """       
        detections = data.labels_unfiltered2.to_dict('records')
        if not detections:
            return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model'])
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)        
    
        zyx = []
        for d in detections:
            zyx.append([d['z'], d['y'], d['x']])
        zyx = np.array(zyx)
    
        import sklearn.cluster
        clustering = sklearn.cluster.DBSCAN(eps=self.distance_threshold, min_samples=1).fit(zyx)
    
        detections = pd.DataFrame(detections)
        final_detections = []
        for lab in np.unique(clustering.labels_):
            this_detections = detections[clustering.labels_==lab].reset_index()
            #print('---')
            #print(this_detections)
            mean_z = np.round(np.sum(this_detections['z'] * this_detections['confidence']) / np.sum(this_detections['confidence'])).astype(int)
            mean_y = np.round(np.sum(this_detections['y'] * this_detections['confidence']) / np.sum(this_detections['confidence'])).astype(int)
            mean_x = np.round(np.sum(this_detections['x'] * this_detections['confidence']) / np.sum(this_detections['confidence'])).astype(int)
            #print(mean_z)

            all_scores_this_motor = []
            weights = []
            ran = np.round(self.z_range).astype(int)
            for z in np.arange(mean_z-ran, mean_z+ran+1):
                for i_model in range(self.n_models):
                    this_lab = this_detections[np.logical_and(this_detections['z']==z, this_detections['i_model']==i_model)]
                    weight = np.exp(-(mean_z-z)**2/(2*self.z_range**2))
                    weight = 1
                    if len(this_lab)==0:
                        all_scores_this_motor.append(0)
                    else:
                        all_scores_this_motor.append((np.max(this_lab['confidence'])**self.gamma) *weight)
                    weights.append(weight)
            final_score = np.sum(all_scores_this_motor)/np.sum(weights)#+self.std_factor*np.std(all_scores_this_motor)
            final_detections.append({'z':mean_z, 'y':mean_y, 'x':mean_x, 'confidence':final_score, 'all_scores':all_scores_this_motor})
    
        if len(final_detections)==0:
            print('FINAL')
            print(pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model', 'all_scores']))
            print('')
            return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model', 'all_scores'])
        else:
            print('FINAL')
            print(pd.DataFrame(final_detections))
            print('')
            return pd.DataFrame(final_detections)

@dataclass
class FindClustersMultiZSimple(fls.BaseClass):
    distance_threshold = 10.
    min_confidence = np.nan
    n_models = np.nan
    #z_range = 1e-10
    #std_factor = 0.
    #gamma = 1.

    def cluster(self, data):
        """
        Perform 3D Non-Maximum Suppression on detections to merge nearby motors.
        """       
        detections = data.labels_unfiltered2.to_dict('records')
        if not detections:
            return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model'])
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)        
    
        zyx = []
        for d in detections:
            zyx.append([d['z'], d['y'], d['x']])
        zyx = np.array(zyx)
    
        import sklearn.cluster
        clustering = sklearn.cluster.DBSCAN(eps=self.distance_threshold, min_samples=1).fit(zyx)
    
        detections = pd.DataFrame(detections)
        final_detections = []
        for lab in np.unique(clustering.labels_):
            this_detections = detections[clustering.labels_==lab].reset_index()
            #print('---')
            #print(this_detections)
            mean_z = np.round(np.sum(this_detections['z'] * this_detections['confidence']) / np.sum(this_detections['confidence'])).astype(int)
            mean_y = np.round(np.sum(this_detections['y'] * this_detections['confidence']) / np.sum(this_detections['confidence'])).astype(int)
            mean_x = np.round(np.sum(this_detections['x'] * this_detections['confidence']) / np.sum(this_detections['confidence'])).astype(int)
            #print(mean_z)

            all_scores_this_motor = []
            weights = []
            for z in np.unique(this_detections['z']):
                for i_model in range(self.n_models):
                    this_lab = this_detections[np.logical_and(this_detections['z']==z, this_detections['i_model']==i_model)]
                    if len(this_lab)==0:
                        all_scores_this_motor.append(0)
                    else:
                        all_scores_this_motor.append(np.max(this_lab['confidence']))
            final_score = np.sum(all_scores_this_motor)*data.voxel_spacing #+self.std_factor*np.std(all_scores_this_motor)
            final_detections.append({'z':mean_z, 'y':mean_y, 'x':mean_x, 'confidence':final_score, 'all_scores':all_scores_this_motor})
    
        if len(final_detections)==0:
            print('FINAL')
            print(pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model', 'all_scores']))
            print('')
            return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model', 'all_scores'])
        else:
            print('FINAL')
            print(pd.DataFrame(final_detections))
            print('')
            return pd.DataFrame(final_detections)
            
@dataclass
class ThreeStepModelLabelBased(fls.Model):
    # Runs modeling in 2 steps:
    # 1) create labels from data
    # 2) select a subset as output (optionally one per tomogram)
    step1Labels: object = field(init=True, default_factory=flg_yolo2.YOLOModel)
    step2Motors: object = field(init=True, default_factory=FindClusters)
    step3Output: object = field(init=True, default_factory=SelectSingleMotors)

    calibrate_step_3: bool = field(init=True, default=False)
    submission_threshold_offset: float = field(init=True, default=0.)
    # Intermediate
    data_after_step2 = 0

    # Internal
    run_to: int = field(init=True, default=0) # 0: run all, 1: stop after creating labels

    def __post_init__(self):
        super().__post_init__()
        self.step3Output.x_val = 'confidence'
        self.step3Output.y_val = 'confidence'
        self.step3Output.vals = np.linspace(0,1,100)
        self.step3Output.threshold = 0.

    def _train(self, train_data, validation_data):
        self.step1Labels.seed = self.seed
        if self.step1Labels.trained_model==0:
            self.step1Labels.train(train_data, validation_data)

        if self.calibrate_step_3:
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
            data.labels_unfiltered2 = self.step1Labels.infer(data)
        else:
            for d in self.data_after_step2:
                if d.name == data.name:
                    data.labels_unfiltered2 = d.labels_unfiltered2

        try: 
            self.step2Motors.min_confidence = self.step1Labels.confidence_threshold
            self.step2Motors.n_models = self.step1Labels.n_ensemble
        except:
            self.step2Motors.min_confidence = self.step1Labels.model_internal.confidence_threshold
            self.step2Motors.n_models = self.step1Labels.model_internal.n_ensemble
            
        data.labels_unfiltered = self.step2Motors.cluster(data) 

        #if fls.is_submission:
        data.labels_unfiltered['confidence'] = data.labels_unfiltered['confidence']+self.submission_threshold_offset
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
class TestTimeAugmentationStep1(fls.Model):
    model_internal:fls.Model = field(init=True, default_factory=ThreeStepModelLabelBased)
    flip_ud_vals:list = field(init=True, default_factory = lambda:[False,True])
    flip_lr_vals:list = field(init=True, default_factory = lambda:[False,False,True,True])
    voxel_scale_vals: list = field(init=True, default_factory = lambda:[1.])
    
    def train(self, train_data, validation_data):
        self.model_internal.train(train_data, validation_data)

    def infer(self,data):
        label_list = []
        assert len(self.model_internal.trained_model) == self.model_internal.n_ensemble
        for i_model in range(len(self.model_internal.trained_model)):
            flip_ud_val = self.flip_ud_vals[i_model%len(self.flip_ud_vals)]
            flip_lr_val = self.flip_lr_vals[i_model%len(self.flip_lr_vals)]
            voxel_scale_val = self.voxel_scale_vals[i_model%len(self.voxel_scale_vals)]
            print(flip_ud_val, flip_lr_val)
            
            model = copy.deepcopy(self.model_internal)
            model.preprocessor.apply_flipud = flip_ud_val
            model.preprocessor.apply_fliplr = flip_lr_val
            model.preprocessor.voxel_scale = voxel_scale_val
            model.trained_model = [model.trained_model[i_model]]
            label_list.append(model.infer(copy.deepcopy(data)))     
            if len(label_list[-1])>0:
                print(i_model)
                ind = np.argmax(label_list[-1]['confidence'])
                print(label_list[-1][ind:ind+1])
        labels = label_list[0]
        labels['i_model'] = 0
        for ii,lab in enumerate(label_list[1:]):
            lab['i_model'] = ii+1
            labels = pd.concat([labels, lab], ignore_index=True, sort=False)
        print(labels)
        if self.model_internal.final_relative_confidence_threshold:
            labels = labels[labels['confidence']>self.model_internal.relative_confidence_threshold*np.max(labels['confidence'])]
        return labels


@dataclass
class TestTimeAugmentation(fls.Model):
    model_internal:fls.Model = field(init=True, default_factory=ThreeStepModelLabelBased)
    voxel_spacing_scale_vals: list = field(init=True, default_factory=lambda:[0.5, np.sqrt(0.5), 1., np.sqrt(2)])
    rotation_vals: list = field(init=True, default_factory=lambda:[0,0,0,0])

    def _train(self, train_data, validation_data):
        self.model_internal.train(train_data, validation_data)

    def _infer(self,data):
        data_list = []
        self.model_internal.ratio_of_motors_allowed = 1.
        for v in self.voxel_spacing_scale_vals:
            data2 = copy.deepcopy(data)
            self.model_internal.step1Labels.preprocessor.voxel_scale = v
            data_list.append(self.model_internal.infer(data2))
        data_out = []
        for ii in range(len(data_list[0])):
            vals = []
            for jj in range(len(self.voxel_spacing_scale_vals)):
                print(data_list[jj][ii].labels)
                if len(data_list[jj][ii].labels)==0:
                    vals.append(0)
                else:
                    vals.append(data_list[jj][ii].labels['value'][0])
            ind = np.argmax(vals)
            print(ind)
            data_out.append(data_list[ind][ii])
            
        return data_out

@dataclass
class FinalModel(fls.Model):
    step1_list: list = field(init=True, default_factory=list)
    DIST: float = field(init=True, default=1000.)
    gauss_range: float = field(init=True, default=20.)
    models_first_go: int = field(init=True, default=2)
    extend_range: float = field(init=True, default=20.)
    conf_thresh: float = field(init=True, default=0.1)

    def _train(self, train_data, validation_data):
        raise Exception('Not implemented')

    def _infer_single(self,data):
        res = []

        # Model
        fls.do_gpu_clearing=True
        for i_model,model in enumerate(self.step1_list[:self.models_first_go]):
            flg_yolo2.slices_to_do_global=[]
            res.append(model.infer(copy.deepcopy(data)))
            res[-1]['i_model']=i_model
        labels = pd.concat(copy.deepcopy(res), ignore_index=True)
        labels = labels[labels['confidence']>self.conf_thresh]
        z_vals = np.unique(labels['z'])
        ran = np.round(self.extend_range/data.voxel_spacing).astype(int)
        z_vals = np.unique((z_vals[:,None]+np.arange(-ran,ran+1)[None,:]).flatten())
        fls.claim_gpu('')
        fls.do_gpu_clearing=False
        for i_model,model in enumerate(self.step1_list[self.models_first_go:]):
            flg_yolo2.slices_to_do_global=list(z_vals)
            res.append(model.infer(copy.deepcopy(data)))
            res[-1]['i_model']=i_model+self.models_first_go
        labels = pd.concat(res, ignore_index=True)
        fls.do_gpu_clearing=True
        fls.claim_gpu('cupy')
        fls.claim_gpu('pytorch')
        fls.claim_gpu('')

        # Extract per-slice per-model values
        results = dict()
        results['z'] = []
        results['y'] = []
        results['x'] = []
        results['value'] = []
        while True:        
            labels = labels.reset_index(drop=True)
            def get_zyx(ind):
                return labels['z'][ind], labels['y'][ind], labels['x'][ind]        
            if len(labels['confidence'])==0:
                break
            ind_max = np.argmax(labels['confidence'])
    
            if labels['confidence'][ind_max]<0.01:
                break
            z,y,x = get_zyx(ind_max)
            
            dist_scaled = self.DIST/data.voxel_spacing
            dist_scaled_int = np.floor(dist_scaled).astype(int)
            consider = []
            for ind in range(len(labels)):
                z2,y2,x2 = get_zyx(ind)
                consider.append((z2-z)**2+(y2-y)**2+(z2-z)**2 < dist_scaled**2)
            to_consider = labels[consider]
            labels = labels[np.logical_not(consider)]
        
            # 1) define your range of z values
            z_min = int(z - dist_scaled_int)
            z_max = int(z + dist_scaled_int)
            z_range = np.arange(z_min, z_max + 1)
            
            # 2) pivot to get max confidence per (z, i_model), filling missing with 0
            pivot = (
                to_consider
                  .pivot_table(
                     index='z',
                     columns='i_model',
                     values='confidence',
                     aggfunc='max',
                     fill_value=0
                  )
            )
            
            # 3) reindex to make sure every z in your window appears, and every model 0..n_model-1
            pivot = pivot.reindex(
                index=z_range,
                columns=np.arange(len(self.step1_list)),
                fill_value=0
            )        
            
            z_diff = z_range-z
            z_scaled = z_diff*data.voxel_spacing
            val = (np.mean(pivot,axis=1)).to_numpy()

            max_loc = np.argmax(val)
            ran = np.round(self.gauss_range/data.voxel_spacing).astype(int)
            ran = np.arange(max_loc-ran,max_loc+ran+1)
            gauss_weights = np.exp(-z_scaled**2/(2*25)**2)
            gauss_weights = gauss_weights/np.sum(gauss_weights)
            confidence = np.sum(val*gauss_weights)

            results['z'].append(z)
            results['y'].append(y)
            results['x'].append(x)
            results['value'].append(confidence)

        df = pd.DataFrame(results)

        if len(df)>0:
            ind_max = np.argmax(df['value'])
            df = df[ind_max:ind_max+1]

        data.labels = df

        return data
            
