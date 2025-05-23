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
import flg_model
import sklearn.neighbors
import sklearn.mixture
import sklearn.gaussian_process
import h5py

def baseline_runner(fast_mode = False, local_mode = False):
    res = ModelRunner()
    res.label = 'Baseline';
    res.base_model = flg_model.ThreeStepModelLabelBased()


    # print('8 ensemble, 141 epochs')
    # res.base_model.step1Labels.epochs_save = [20,40]
    # res.modifier_dict['n_ensemble'] = pm(1, lambda r:2, yolo)
    # res.modifier_dict['n_epochs'] = pm(50, lambda r:41, n_epochs)   
    # res.modifier_dict['use_best_epoch'] = pm(True, lambda r:False, use_best_epoch)   
    # res.modifier_dict['lr0'] = pm(0.001, lambda r:0.001, yolo)  
    # res.modifier_dict['cos_lr'] = pm(False, lambda r:False, cos_lr)  
    # res.modifier_dict['mosaic'] = pm(0., lambda r:0., yolo)  
    # res.modifier_dict['concentration'] = pm(1, lambda r:2, yolo)  
    # res.modifier_dict['extra_data'] = pm(False, lambda r:True, add_all_datasets)
    # res.modifier_dict['trust_neg'] = pm(0, lambda r:0, yolo)
    # res.modifier_dict['trust_extra'] = pm(4, lambda r:4, yolo)
    # model_list = ['yolov8m']
    # res.modifier_dict['model_name'] = pm('yolov9s', lambda r:model_list[r.integers(0,len(model_list))], yolo)
    # res.modifier_dict['use_pretrained_weights'] = pm(True, lambda r:r.uniform()>0.5, yolo)

       
    # Ensemble
    if local_mode:
        res.modifier_dict['n_ensemble'] = pm(1, lambda r:r.integers(1,2), yolo)
    else:
        res.modifier_dict['n_ensemble'] = pm(1, lambda r:1, yolo)
    res.modifier_dict['concentration'] = pm(1, lambda r:r.integers(1,2), yolo)  
    
    # Data
    res.base_model.train_data_selector.datasets = ['tom']
    res.modifier_dict['extra_data'] = pm(False, lambda r:True, add_all_datasets)
    res.modifier_dict['trust_neg'] = pm(0, lambda r:1, yolo)
    res.modifier_dict['trust_extra'] = pm(4, lambda r:1, yolo)
    res.modifier_dict['negative_label_threshold'] = pm(0.6, lambda r:0.6, yolo)

    # Preprocessing
    res.modifier_dict['target_voxel_spacing'] = pm(20., lambda r:25., prep)
    res.modifier_dict['blur_xy'] = pm(30, lambda r:30., prep)
    res.modifier_dict['blur_z'] = pm(0., lambda r:0., prep)
    res.modifier_dict['scale_moving_std'] = pm(True, lambda r:True, prep)
    res.modifier_dict['scale_moving_average_size'] = pm(3000, lambda r:3000, prep)
    res.modifier_dict['scale_moving_std_size_fac'] = pm(1., lambda r:1.3, scale_moving_std_size_fac)
    res.modifier_dict['blur_xy_moving_std'] = pm(60., lambda r:60., prep)
    res.modifier_dict['clip_value'] = pm(3., lambda r:3., prep)
    res.modifier_dict['scale_percentile_value'] = pm(3., lambda r:3., prep)
    res.modifier_dict['img_size'] = pm(640, lambda r:640-32, yolo)
    res.modifier_dict['box_size'] = pm(18, lambda r:18, yolo)

    # Learning
    res.modifier_dict['n_epochs'] = pm(50, lambda r:(r.integers(20,51)).item(), n_epochs)   
    res.modifier_dict['use_best_epoch'] = pm(True, lambda r:False, use_best_epoch)   
    res.modifier_dict['lr0'] = pm(0.001, lambda r:10**-3.5, yolo)  
    res.modifier_dict['cos_lr'] = pm(False, lambda r:True, cos_lr)  
    res.modifier_dict['lrf'] = pm(0.01, lambda r:0.1), yolo)  
    res.modifier_dict['dropout'] = pm(0., lambda r:0., yolo)  
    res.modifier_dict['weight_decay'] = pm(0.0005, lambda r:0.0003, yolo)  
    res.modifier_dict['momentum'] = pm(0.937, lambda r:0.937, yolo)
    res.modifier_dict['warmup_epochs'] = pm(3., lambda r:3., yolo)

    # Cost function
    res.modifier_dict['box'] = pm(7.5, lambda r:4., yolo)

    # Model
    model_list = ['yolov8s', 'yolov8m', 'yolov8l']
    res.modifier_dict['model_name'] = pm('yolov9s', lambda r:model_list[r.integers(0,len(model_list))], yolo)
    res.modifier_dict['use_pretrained_weights'] = pm(True, lambda r:False, pretrained_weights)

    # Augmentation
    res.modifier_dict['mosaic_mode'] = pm(0, lambda r:1., mosaic_mode) 
    res.modifier_dict['translate'] = pm(0.1, lambda r:0., yolo) 
    res.modifier_dict['scale'] = pm(0.5, lambda r:0.5, yolo)
    res.modifier_dict['mixup'] = pm(0.2, lambda r:0.2, yolo)
    res.modifier_dict['erasing'] = pm(0.4, lambda r:0.2, yolo)
    res.modifier_dict['hsv_h'] = pm(0.015, lambda r:0., yolo)
    res.modifier_dict['hsv_s'] = pm(0.7, lambda r:0., yolo)
    res.modifier_dict['hsv_v'] = pm(0.4, lambda r:0.2, yolo)
    res.modifier_dict['fliplr'] = pm(0.5, lambda r:0.5, yolo)
    res.modifier_dict['flipud'] = pm(0.5, lambda r:0.5, yolo)
    res.modifier_dict['degrees'] = pm(0., lambda r:30., yolo)   

    # Post processing
    res.modifier_dict['absolute_threshold'] = pm(False, lambda r:False, absolute_threshold)
    res.modifier_dict['distance_threshold'] = pm(10., lambda r:10., clusters) 
    def z_range_func(r):
        if r.uniform()<0.4:
            return -1
        else:
            return r.integers(3,7)
    res.modifier_dict['z_range'] = pm(0, lambda r:4, z_range) 
    res.modifier_dict['adjust_voxel_scale'] = pm(1., lambda r:1., adjust_prep_multiply)
    res.modifier_dict['adjust_voxel_scale'].modify_after_train = True
    res.modifier_dict['adjust_clip_value'] = pm(1., lambda r:1., adjust_prep_multiply)
    res.modifier_dict['adjust_clip_value'].modify_after_train = True
    res.modifier_dict['adjust_blur_xy'] = pm(1., lambda r:1., adjust_prep_multiply)
    res.modifier_dict['adjust_blur_xy'].modify_after_train = True
    res.modifier_dict['adjust_blur_z'] = pm(1., lambda r:1., adjust_prep_add)
    res.modifier_dict['adjust_blur_z'].modify_after_train = True

    offset_vals = [0.,1.,15.,30.,45.,60.,100.,500.]
    res.modifier_dict['rgb_offset'] = pm(0., lambda r:offset_vals[r.integers(0,len(offset_vals))], yolo)
    res.modifier_dict['pad_with_noise'] = pm(False, lambda r:r.uniform()>0.5, yolo)
    

    

    
    
    

    
    #res.modifier_dict['blur_z'] = pm(0., lambda r:r.uniform(0.,15.), prep)
    

    #res.modifier_dict['erasing'] = pm(0.4, lambda r:0.4*(r.uniform()>0.5), yolo)



    
    res.do_inference = True    
    if fast_mode:
        res.label = 'Baseline fast mode'
        res.train_part = slice(0,40)
        res.test_part = slice(None)
        res.N_test_positive = 20
        res.N_test_negative = 1
        res.base_model.step1Labels.n_epochs = 10
        res.base_model.step1Labels.n_ensemble = 1
        del res.modifier_dict['n_epochs']
        del res.modifier_dict['n_ensemble']
    return res

@dataclass
class ModelRunner(fls.BaseClass):
    # Inputs
    label: str = field(init=False, default = '')
    seed=0
    env=''
    base_model=0
    modifier_dict: dict = field(init=True, default_factory=dict)
    use_missing_value = False
    N_test_positive = 200
    N_test_negative = 30
    include_test_data_in_train = False
    train_part: slice = field(init=True, default_factory = lambda:slice(None,None,None))
    test_part: slice = field(init=True, default_factory = lambda:slice(None,None,None))
    train_in_subprocess = True
    do_inference = True

    # Outputs    
    git_commit_id: str = field(init=False, default = '')
    modifier_values=0
    untrained_model=0
    trained_model=0
    train_data=0
    test_data=0
    inferred_test_data=0     
    cv_score = (np.nan,np.nan,np.nan)
    exception = 0
            
    def run(self):
        try:
            
            # Split train and test data
            all_data = fls.load_all_train_data() + fls.load_all_extra_data()
            np.random.default_rng(seed=0).shuffle(all_data)
            n_motors = np.array([len(d.labels) for d in all_data])
            inds_zero = np.argwhere(n_motors==0)[:self.N_test_negative,0]
            inds_one = np.argwhere(n_motors==1)[:self.N_test_positive,0]
            inds_test = np.concatenate((inds_zero,inds_one))
            inds_train = np.setdiff1d(np.arange(len(n_motors)), inds_test)
    
            train_data = []
            for i in inds_train:
                train_data.append(all_data[i])
            test_data = []
            for i in inds_test:
                test_data.append(all_data[i])        
            print(len(train_data), len(test_data))
            self.train_data = train_data[self.train_part]
            self.test_data = test_data[self.test_part]
            print(len(self.train_data), len(self.test_data))
    
            if self.include_test_data_in_train:
                self.train_data = self.train_data + self.test_data
            
            # Set up modified model
            rng = np.random.default_rng(seed=self.seed)
            while True:
                model = copy.deepcopy(self.base_model)
                self.modifier_values = dict()
                self.modifier_values['seed'] = self.seed
                model.seed = self.seed            
                for key, value in self.modifier_dict.items():  
                    if not self.use_missing_value:
                        self.modifier_values[key] = value.random_function(rng)
                    else:
                        self.modifier_values[key] = value.missing_value
                    if not value.modify_after_train:
                        value.modifier_function(model, key, self.modifier_values[key])
                model.step1Labels.epochs_save = list(np.arange(30,model.step1Labels.n_epochs,30))
                self.untrained_model = copy.deepcopy(model)
                print(self.modifier_values)
                if len(model.train_data_selector.datasets)>0:
                    break            
            self.untrained_model.step1Labels.epochs_save = list(np.arange(30,self.untrained_model.step1Labels.n_epochs,30))
            #return
    
            # Train model
            if self.train_in_subprocess:
                model = model.train_subprocess(self.train_data, self.test_data)
            else:
                model.train(self.train_data, self.test_data)
            for key, value in self.modifier_dict.items():  
                if value.modify_after_train:
                    value.modifier_function(model, key, self.modifier_values[key])
            self.trained_model = copy.deepcopy(model)

            # Infer
            if fls.env=='vast':
                model.run_in_parallel = False            
            model.ratio_of_motors_allowed = np.sum([len(d.labels)>0 for d in self.test_data])/len(self.test_data)
            print('ratio: ', model.ratio_of_motors_allowed)
            if self.do_inference:
                self.inferred_test_data = model.infer(self.test_data)       
                self.cv_score = fls.score_competition_metric(self.inferred_test_data, self.test_data)
            else:
                self.cv_score = [0.1,0.1,0.1]
            
        except Exception as e:
            print('ERROR!')
            import traceback
            self.exception = traceback.format_exc()
            print(self.exception)
            


def pm(missing_value, random_function, modifier_function):
    res = PropertyModifier()
    res.missing_value = missing_value
    res.random_function = random_function
    res.modifier_function = modifier_function
    return res
    
        
@dataclass
class PropertyModifier(fls.BaseClass):
    default_value = 0
    missing_value = 0 # value to assume for this if it's missing in older output
    random_function = 0 # gets RNG as input, should return a new value
    modifier_function = 0 # gets model, name (in dict), and value as input, should adapt model (does not have to return)    
    modify_after_train = False

def prep(model, name, value):
    setattr(model.step1Labels.preprocessor, name, value)

def data_sel(model, name, value):
    setattr(model.train_data_selector, name, value)

def yolo(model, name, value):
    setattr(model.step1Labels, name, value)

def clusters(model, name, value):
    setattr(model.step2Motors, name, value)

def add_dataset(model, name, value):
    if value:
        model.train_data_selector.datasets.append(name)

def add_all_datasets(model, name, value):
    if value:
        model.train_data_selector.datasets.append('ycw')
        model.train_data_selector.datasets.append('mba')
        model.train_data_selector.datasets.append('aba')

def use_best_epoch(model, name, value):
    model.step1Labels.use_best_epoch = value
    if not model.step1Labels.use_best_epoch:
        model.step1Labels.patience = 0

def set_scale_approach(model, name, value):
    model.step1Labels.preprocessor.scale_std = False
    model.step1Labels.preprocessor.scale_moving_average = False
    model.step1Labels.preprocessor.scale_also_moving_std = False
    if value>=1:
        model.step1Labels.preprocessor.scale_std = True
    if value>=2:
        model.step1Labels.preprocessor.scale_moving_average = True
    if value>=3:
        model.step1Labels.preprocessor.scale_also_moving_std = True

def n_epochs(model,name,value):
    model.step1Labels.n_epochs = value
    model.step1Labels.close_mosaic = value//2

def cos_lr(model,name,value):
    model.step1Labels.cos_lr = value
    if value:
        model.step1Labels.lrf = 0.01
        model.step1Labels.n_epochs = np.round(model.step1Labels.n_epochs*1.2).astype(int).item()

def pretrained_weights(model,name,value):
    model.step1Labels.use_pretrained_weights = value
    if not value:
        model.step1Labels.n_epochs = np.round(model.step1Labels.n_epochs*1.5).astype(int).item()

def scale_moving_std_size_fac(model,name,value):
    model.step1Labels.preprocessor.scale_moving_std_size = np.round(model.step1Labels.preprocessor.scale_moving_average_size * value).astype(int)

def mosaic_mode(model,name,value):
    if value<0.33333:
        model.step1Labels.mosaic = 0.0
        model.step1Labels.close_mosaic = 100000
    elif value<0.666667:
        model.step1Labels.mosaic = 1.0
        model.step1Labels.close_mosaic = model.step1Labels.n_epochs//2
    else:
        model.step1Labels.mosaic = 1.0
        model.step1Labels.close_mosaic = 100000

def absolute_threshold(model,name,value):
    if value:
        model.step1Labels.relative_confidence_threshold = 0.
        model.step1Labels.confidence_threshold = 0.01

def z_range(model,name,value):
    if value>=0:
        model.step2Motors = flg_model.FindClustersMultiZ()
        model.step2Motors.z_range = value
        print('range: ', model.step2Motors.z_range)

def adjust_prep_multiply(model,name,value):
    setattr(model.step1Labels.preprocessor, name[7:], value*getattr(model.step1Labels.preprocessor, name[7:]))
def adjust_prep_add(model,name,value):
    setattr(model.step1Labels.preprocessor, name[7:], value+getattr(model.step1Labels.preprocessor, name[7:]))