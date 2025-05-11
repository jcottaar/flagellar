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

    
    res.modifier_dict['n_ensemble'] = pm(1, lambda r:4, yolo)
    res.modifier_dict['n_epochs'] = pm(50, lambda r:(r.integers(30,71)).item(), n_epochs)   
    res.modifier_dict['use_best_epoch'] = pm(True, lambda r:False, use_best_epoch)   
    res.modifier_dict['lr0'] = pm(0.001, lambda r:10**(r.uniform(-3.5,-3.)), yolo)  
    res.modifier_dict['cos_lr'] = pm(False, lambda r:r.uniform()>0.5, cos_lr)  
    #res.modifier_dict['dropout'] = pm(0., lambda r:r.uniform(0.,0.1), yolo)  
    res.modifier_dict['mosaic'] = pm(0., lambda r:1.0*(r.uniform()>0.5), yolo)  
    res.modifier_dict['concentration'] = pm(1, lambda r:r.integers(1,3), yolo)  

    #res.modifier_dict['box'] = pm(7.5, lambda r:r.uniform(1.,7.5), yolo)

    res.base_model.train_data_selector.datasets = ['tom']
    res.modifier_dict['extra_data'] = pm(False, lambda r:True, add_all_datasets)
    res.modifier_dict['trust_neg'] = pm(0, lambda r:r.integers(-1,2), yolo)
    res.modifier_dict['trust_extra'] = pm(4, lambda r:r.integers(0,5), yolo)

    model_list = ['yolov8s', 'yolov8m']
    res.modifier_dict['model_name'] = pm('yolov9s', lambda r:model_list[r.integers(0,len(model_list))], yolo)
    res.modifier_dict['use_pretrained_weights'] = pm(True, lambda r:r.uniform()>0.5, pretrained_weights)

    #res.modifier_dict['blur_xy'] = pm(30, lambda r:r.uniform(15.,45.), prep)
    #res.modifier_dict['blur_z'] = pm(0., lambda r:r.uniform(0.,15.), prep)
    #res.modifier_dict['scale_moving_std'] = pm(True, lambda r:r.uniform()>0.5, prep)

    #res.modifier_dict['erasing'] = pm(0.4, lambda r:0.4*(r.uniform()>0.5), yolo)



    
    res.do_inference = local_mode
    if local_mode:
        res.modifier_dict['n_ensemble'] = pm(1, lambda r:2, yolo)
        res.modifier_dict['extra_data'] = pm(False, lambda r:False, add_all_datasets)
    if fast_mode:
        res.label = 'Baseline fast mode'
        res.train_part = slice(0,400)
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
                    value.modifier_function(model, key, self.modifier_values[key])
                self.untrained_model = copy.deepcopy(model)
                print(self.modifier_values)
                if len(model.train_data_selector.datasets)>0:
                    break
            self.step1Labels.epochs_save = list(np.arange(30,self.step1Labels.n_epochs,30))
            #return
    
            # Train model
            print('XXX', np.sum([len(d.labels)>0 for d in self.test_data])/len(self.test_data))
            if self.train_in_subprocess:
                model = model.train_subprocess(self.train_data, self.test_data)
            else:
                model.train(self.train_data, self.test_data)
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
        