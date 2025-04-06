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
import flg_yolo
import sklearn.neighbors
import sklearn.mixture
import sklearn.gaussian_process
import h5py

def baseline_runner(fast_mode = False):
    res = ModelRunner()
    res.label = 'Baseline';
    res.base_model = flg_yolo.YOLOModel()
    res.modifier_dict['scale_percentile_value'] = pm(2., lambda r:r.uniform(1.,5.), prep)
    res.modifier_dict['img_size'] = pm(640, lambda r:(320+64*r.integers(0,6)).item(), setattr)
    res.modifier_dict['n_epochs'] = pm(30, lambda r:(r.integers(20,100)).item(), setattr)    
    model_list = ['yolov8m', 'yolov8l', 'yolo11m']
    res.modifier_dict['model_name'] = pm('yolov8m', lambda r:model_list[r.integers(0,len(model_list))], setattr)
    res.modifier_dict['use_pretrained_weights'] = pm(True, lambda r:r.uniform()>0.2, setattr)
    res.modifier_dict['box_size'] = pm(24, lambda r:r.integers(16,32).item(), setattr)
    res.modifier_dict['trust'] = pm(4, lambda r:r.integers(0,6).item(), setattr)
    res.modifier_dict['fix_norm_bug'] = pm(False, lambda r:r.uniform()>0.5, setattr)
    res.modifier_dict['weight_decay'] = pm(0.0005, lambda r:r.uniform(0,0.001), setattr)
    res.modifier_dict['hsv_h'] = pm(0.015, lambda r:0.015*(r.uniform()>0.8), setattr)
    res.modifier_dict['hsv_s'] = pm(0.7, lambda r:0.7*(r.uniform()>0.8), setattr)
    res.modifier_dict['hsv_v'] = pm(0.4, lambda r:0.4*(r.uniform()>0.8), setattr)
    res.modifier_dict['translate'] = pm(0.1, lambda r:0.1*(r.uniform()>0.2), setattr)
    res.modifier_dict['scale'] = pm(0.5, lambda r:r.uniform(0,0.7)*(r.uniform()>0.2), setattr)
    res.modifier_dict['fliplr'] = pm(0.5, lambda r:0.5*(r.uniform()>0.2), setattr)
    res.modifier_dict['flipud'] = pm(0.5, lambda r:0.5*(r.uniform()>0.2), setattr)
    res.modifier_dict['degrees'] = pm(0., lambda r:180*(r.uniform()>0.8), setattr)
    res.modifier_dict['shear'] = pm(0., lambda r:0.2*(r.uniform()>0.8), setattr)
    res.modifier_dict['mosaic'] = pm(1.0, lambda r:1.0*(r.uniform()>0.5), setattr)
    res.modifier_dict['mixup'] = pm(0.2, lambda r:0.2*(r.uniform()>0.5), setattr)
    res.modifier_dict['erasing'] = pm(0.4, lambda r:0.4*(r.uniform()>0.5), setattr)
    res.modifier_dict['use_albumentations'] = pm(False, lambda r:(r.uniform()>0.5), setattr)
    res.modifier_dict['confidence_threshold'] = pm(0.45, lambda r:r.uniform(0.35,0.55), setattr)
    res.modifier_dict['include_multi_motor'] = pm(True, lambda r:r.uniform()>0.5, data_sel)

    res.base_model.train_data_selector.datasets = []
    res.modifier_dict['tom'] = pm(True, lambda r:r.uniform()>0.5, add_dataset)
    res.modifier_dict['mba'] = pm(True, lambda r:r.uniform()>0.5, add_dataset)
    res.modifier_dict['aba'] = pm(True, lambda r:r.uniform()>0.5, add_dataset)
    res.modifier_dict['ycw'] = pm(True, lambda r:r.uniform()>0.5, add_dataset)
    if fast_mode:
        res.label = 'Baseline fast mode'
        res.train_part = slice(0,10)
        res.test_part = slice(0,10)
        res.base_model.n_epochs = 2
        del res.modifier_dict['n_epochs']
    return res

@dataclass
class ModelRunner(fls.BaseClass):
    # Inputs
    label: str = field(init=False, default = '')
    seed=0
    base_model=0
    modifier_dict: dict = field(init=True, default_factory=dict)
    N_test_positive = 300
    N_test_negative = 50
    train_part: slice = field(init=True, default_factory = lambda:slice(None,None,None))
    test_part: slice = field(init=True, default_factory = lambda:slice(None,None,None))
    train_in_subprocess = True

    # Outputs    
    git_commit_id: str = field(init=False, default = '')
    modifier_values=0
    untrained_model=0
    trained_model=0
    train_data=0
    test_data=0
    inferred_test_data=0            
            
    def run(self):
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
        
        # Set up modified model
        rng = np.random.default_rng(seed=self.seed)
        while True:
            model = copy.deepcopy(self.base_model)
            self.modifier_values = dict()
            self.modifier_values['seed'] = self.seed
            model.seed = self.seed            
            for key, value in self.modifier_dict.items():  
                self.modifier_values[key] = value.random_function(rng)
                value.modifier_function(model, key, self.modifier_values[key])
            self.untrained_model = copy.deepcopy(model)
            print(self.modifier_values)
            if len(model.train_data_selector.datasets)>0:
                break
        #return

        # Train model
        if self.train_in_subprocess:
            model = model.train_subprocess(self.train_data, self.test_data)
        else:
            model.train(self.train_data, self.test_data)
        self.trained_model = copy.deepcopy(model)

        # Infer
        if fls.env=='vast':
            model.run_in_parallel = False
        self.inferred_test_data = model.infer(self.test_data)             


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
    setattr(model.preprocessor, name, value)

def data_sel(model, name, value):
    setattr(model.train_data_selector, name, value)

def add_dataset(model, name, value):
    if value:
        model.train_data_selector.datasets.append(name)