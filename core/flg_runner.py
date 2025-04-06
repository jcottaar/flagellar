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
    res.label = 'Baseline'
    res.base_model = flg_yolo.YOLOModel()
    if fast_mode:
        res.label = 'Baseline fast mode'
        res.train_part = slice(0,3)
        res.test_part = slice(0,3)
        res.model.n_epochs = 2
    return res

@dataclass
class ModelRunner(fls.BaseClass):
    # Inputs
    label = str = field(init=False, default = '')
    seed=0
    base_model=0
    modifier_dict: dict = field(init=True, default_factory=dict)
    N_test_positive = 300
    N_test_negative = 100
    train_part: slice = field(init=True, default_factory = lambda:slice(None,None,None))
    test_part: slice = field(init=True, default_factory = lambda:slice(None,None,None))

    # Outputs    
    git_commit_id: str = field(init=False, default = '')
    modifier_values=0
    untrained_model=0
    trained_model=0
    train_data=0
    test_data=0
    inferred_test_data=0            
            
    def run(self):
        # Preliminaries
        import git 
        repo = git.Repo(search_parent_directories=True)
        self.git_commit_id = repo.head.object.hexsha

        # Split train and test data
        all_data = fls.load_all_train_data() + fls.load_all_extra_data()
        n_motors = np.array([len(d.labels) for d in all_data])
        inds_zero = np.argwhere(n_motors==0)[:self.N_test_negative,0]
        inds_one = np.argwhere(n_motors==1)[:self.N_test_positive,0]
        inds_test = np.concatenate((inds_zero,inds_one))
        inds_train = np.setdiff1d(np.arange(len(n_motors)), inds_test)
        print(inds_test.shape, inds_train.shape)
        
        train_data = []
        for i in inds_train:
            train_data.append(all_data[i])
        test_data = []
        for i in inds_test:
            test_data.append(all_data[i])
        np.random.default_rng(seed=0).shuffle(test_data)
        test_data = test_data
        len(train_data), len(test_data)
        self.train_data = train_data[self.train_part]
        self.test_data = test_data[self.test_part]

        # Set up modified model
        model = copy.deepcopy(self.base_model)
        self.modifier_values = dict()
        self.modifier_values['seed'] = self.seed
        model.seed = self.seed
        rng = np.random.default_rng(seed=self.seed)
        for key, value in self.modifier_dict.items():  
            self.modifier_values[key] = value.random_function(rng)
            value.modifier_function(model, key, self.modifier_values[key])
        self.untrained_model = copy.deepcopy(model)

        # Train model
        model.train(train_data, test_data)
        self.trained_model = model

        # Infer
        self.inferred_test_data = model.infer(test_data)             


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