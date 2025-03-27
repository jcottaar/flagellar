'''
This module provides general support functionality for my "BYU - Locating Bacterial Flagellar Motors 2025" Kaggle competition work, as well as data loading functions.
'''

import pandas as pd
import numpy as np
import scipy as sp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import IPython
from dataclasses import dataclass, field, fields
import enum
import typing
import pathlib
import multiprocess
multiprocess.set_start_method('spawn', force=True)
from decorator import decorator
from line_profiler import LineProfiler
import os
import gc
import torch
import concurrent
import glob
import cv2
import h5py
import time
import sklearn


'''
Determine environment and globals
'''

is_submission = False
if os.path.isdir('d:/flagellar/'):
    env = 'local'
elif os.path.isdir('/kaggle/working/'):
    env = 'kaggle'
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        is_submission = True
else:
    env = 'vast';
    
debugging_mode = 2
verbosity = 1

match env:
    case 'local':
        data_dir = 'd:/flagellar/data/'
        temp_dir = 'd:/flagellar/temp/'     
        h5py_cache_dir = 'd:/flagellar/cache/'
        model_dir = 'd:/flagellar/models/'
        output_dir = temp_dir
        loader_threads = 32
    case 'kaggle':
        data_dir = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/'
        temp_dir = '/kaggle/temp/'
        h5py_cache_dir = '/kaggle/temp/cache/'
        model_dir = '/kaggle/input/my-flg-models/'
        output_dir = '/kaggle/working/'
        loader_threads = 8
    case 'vast':
        data_dir = '/flagellar/data/'
        temp_dir = '/flagellar/temp/'
        h5py_cache_dir = '/flagellar/cache/'
        model_dir = '/flagellar/models/'
        output_dir = temp_dir
        loader_threads = 32
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(h5py_cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
profiling = False
if is_submission:
    profiling = False

# How many workers is optimal for parallel pool?
def recommend_n_workers():
    return torch.cuda.device_count()

n_cuda_devices = recommend_n_workers()
if not multiprocess.current_process().name == "MainProcess":
    pid = int(multiprocess.current_process().name[-1])-1    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.mod(pid, n_cuda_devices))
    print('CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]);

'''
Helper classes and functions
'''

# Helper class - doesn't allow new properties after construction, and enforces property types. Partially written by ChatGPT.
@dataclass
class BaseClass:
    _is_frozen: bool = field(default=False, init=False, repr=False)

    def check_constraints(self, debugging_mode_offset = 0):
        global debugging_mode
        debugging_mode = debugging_mode+debugging_mode_offset
        try:
            if debugging_mode > 0:
                self._check_types()
                self._check_constraints()
            return
        finally:
            debugging_mode = debugging_mode - debugging_mode_offset

    def _check_constraints(self):
        pass

    def _check_types(self):
        type_hints = typing.get_type_hints(self.__class__)
        for field_info in fields(self):
            field_name = field_info.name
            expected_type = type_hints.get(field_name)
            actual_value = getattr(self, field_name)
            
            if expected_type and not isinstance(actual_value, expected_type):
                raise TypeError(
                    f"Field '{field_name}' expected type {expected_type}, "
                    f"but got value {actual_value} of type {type(actual_value).__name__}.")

    def __post_init__(self):
        # Mark the object as frozen after initialization
        object.__setattr__(self, '_is_frozen', True)

    def __setattr__(self, key, value):
        # If the object is frozen, prevent setting new attributes
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"Cannot add new attribute '{key}' to frozen instance")
        super().__setattr__(key, value)

# Small wrapper for dill loading
def dill_load(filename):
    filehandler = open(filename, 'rb');
    data = dill.load(filehandler)
    filehandler.close()
    return data

# Small wrapper for dill saving
def dill_save(filename, data):
    filehandler = open(filename, 'wb');
    data = dill.dump(data, filehandler)
    filehandler.close()
    return data

def prep_pytorch(seed, deterministic, deterministic_needs_cpu):
    if seed is None:
        seed = np.random.default_rng(seed=None).integers(0,1e6).item()
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    if deterministic and deterministic_needs_cpu:
        device = cpu
    else:
        device = torch.device("cuda")
        claim_gpu('pytorch')
    return cpu, device

gpu_claimant = ''
def claim_gpu(new_claimant):
    global gpu_claimant
    old_claimant = gpu_claimant
    gpu_claimant = new_claimant
    if new_claimant == old_claimant or old_claimant == '':
        return
    gc.collect()
    if old_claimant == 'cupy':
        print('Clearing cupy')
        import cupy # can't do earlier or it will select wrong device
        cache = cupy.fft.config.get_plan_cache()
        cache.clear()
        cupy.get_default_memory_pool().free_all_blocks()
    elif old_claimant == 'pytorch':
        print('Clearing pytorch')
        import torch
        torch.cuda.empty_cache()
    else:
        raise Exception('Unrecognized GPU claimant')

@decorator
def profile_each_line(func, *args, **kwargs):
    if not profiling:
        return func(*args, **kwargs)
    profiler = LineProfiler()
    profiled_func = profiler(func)
    try:
        s=profiled_func(*args, **kwargs)
        profiler.print_stats()
        return s
    except:
        profiler.print_stats()
        raise

def profile_print(string):
    if profiling: print(string)

'''
Data definition and loading
'''
loading_executor = None
all_train_labels = pd.read_csv(data_dir + 'train_labels.csv').rename(columns={"Motor axis 0": "z", "Motor axis 1": "y", "Motor axis 2": "x"})
@dataclass
class Data(BaseClass):
    # Holds one cryoET measurement, including ground truth or predicted labels
    is_train: bool = field(init=True, default=False)
    name: str = field(init=True, default='')
    labels: pd.DataFrame = field(init=True, default_factory=pd.DataFrame)
    labels_unfiltered: pd.DataFrame = field(init=True, default_factory=pd.DataFrame)
    loaded_state: str = field(init=False, default='unloaded') # unloaded, h5py, memory
    data: object = field(init=False, default=None) # None, 3D np array, or filename to h5py
    data_shape: tuple = field(init=False, default = (0,0,0))
    voxel_spacing: float = field(init=True, default=np.nan) # in Angstrom
    mean_per_slice: np.ndarray = field(init=False, default_factory = lambda:np.ndarray(0))
    std_per_slice: np.ndarray = field(init=False, default_factory = lambda:np.ndarray(0))

    def _check_constraints(self):
        if not self.loaded_state == 'unloaded':
            assert(len(self.data_shape)==3)
            if self.loaded_state == 'h5py':
                assert type(self.data)==str
            else:
                assert self.loaded_state == 'memory'
                assert type(self.data)==np.ndarray
                assert self.data.shape == self.data_shape
            assert(self.mean_per_slice.shape==(self.data_shape[0],))
            assert(self.std_per_slice.shape==(self.data_shape[0],))

    def load_to_h5py(self):
        assert self.is_train

        if self.loaded_state == 'h5py': return

        filename = h5py_cache_dir + self.name + '.h5'
        if env == 'kaggle':
            filename = '/kaggle/input/byu-flagellar-motors-as-h5py/' + self.name + '.h5';
            if not os.path.isfile(filename):
                filename = '/kaggle/input/byu-flagellar-motors-as-h5py-part-2/' + self.name + '.h5';
                if not os.path.isfile(filename):
                    filename = '/kaggle/input/byu-flagellar-motors-as-h5py-part-3/' + self.name + '.h5';
                    assert os.path.isfile(filename)  
        
        # Create h5py if needed
        if not os.path.isfile(filename):
            self.load_to_memory()                  
            with h5py.File(filename, 'w') as f:
                dset=f.create_dataset('data', shape = self.data.shape, dtype='uint8')
                dset[...] = self.data
                dset=f.create_dataset('mean_per_slice', shape = self.mean_per_slice.shape, dtype='float64')
                dset[...] = self.mean_per_slice
                dset=f.create_dataset('std_per_slice', shape = self.std_per_slice.shape, dtype='float64')
                dset[...] = self.std_per_slice

        # Import h5py
        with h5py.File(filename, 'r') as f:        
            self.data = filename
            self.mean_per_slice = f['mean_per_slice'][...]
            self.std_per_slice = f['std_per_slice'][...]
            self.data_shape = f['data'].shape

        self.loaded_state = 'h5py'
        self.check_constraints()

    @profile_each_line
    def load_to_memory(self):  
        if self.loaded_state == 'memory': return
        
        if self.is_train and os.path.isfile(h5py_cache_dir + self.name + '.h5'):
            # Load from cache
            with h5py.File(h5py_cache_dir + self.name + '.h5', 'r') as f:
                self.data = f['data'][...]
                self.mean_per_slice = f['mean_per_slice'][...]
                self.std_per_slice = f['std_per_slice'][...]
                self.data_shape = self.data.shape
        else:
            # Read directly
            global loading_executor
            if loading_executor is None:
                loading_executor = concurrent.futures.ThreadPoolExecutor(max_workers=loader_threads)
            if self.is_train:
                files = glob.glob(data_dir + 'train/' + self.name + '/*.jpg')            
            else:
                files = glob.glob(data_dir + 'test/' + self.name + '/*.jpg')
            files.sort()
            def load_image(f):
                return cv2.imread(f, cv2.IMREAD_GRAYSCALE)            
            imgs = list(loading_executor.map(load_image, files))            
            self.data = np.stack(imgs)
            self.data_shape = self.data.shape

            if env=='vast':
                self.mean_per_slice = np.mean(self.data,axis=(1,2))
                self.std_per_slice = np.std(self.data,axis=(1,2))
            else:
                claim_gpu('cupy')
                import cupy as cp            
                data_cp = cp.array(self.data)
                self.mean_per_slice = cp.asnumpy(cp.mean(data_cp,axis=(1,2)))
                self.std_per_slice = cp.asnumpy(cp.std(data_cp,axis=(1,2)))

        assert type(self.data)==np.ndarray
        self.loaded_state = 'memory'
        self.check_constraints()

    def unload(self):
        self.data = None
        self.mean_per_slice = np.ndarray(0)
        self.std_per_slice = np.ndarray(0)
        self.loaded_state = 'unloaded'
        self.check_constraints()            

def load_one_measurement(name, is_train, include_train_labels):
    result = Data()
    result.name = name
    result.is_train = is_train
    if include_train_labels:
        assert is_train
        result.labels = all_train_labels[all_train_labels['tomo_id']==name].reset_index()[['z', 'y', 'x']]
        if result.labels['z'][0]==-1:
            assert result.labels['y'][0]==-1
            assert result.labels['x'][0]==-1
            assert len(result.labels)==1
            result.labels = result.labels[0:0]
        result.voxel_spacing = all_train_labels[all_train_labels['tomo_id']==name].reset_index()[0:1][['Voxel spacing']].to_numpy()[0,0]
    result.check_constraints()    
    return result

def load_all_train_data():    
    #if env=='vast':
    #    directories = glob.glob(h5py_cache_dir + '*.h5')
    #else:
    directories = glob.glob(data_dir + 'train/tomo*')
    directories.sort()
    result = []
    for d in directories:
        name = d[max(d.rfind('\\'), d.rfind('/'))+1:]
        # if env=='vast':
        #     name = name[:-3]
        if not name in['tomo_2b3cdf', 'tomo_62eea8', 'tomo_c84b8e', 'tomo_e6f7f7']: # mislabeled
            result.append(load_one_measurement(name, True, True))
    return result

def load_all_test_data():
    directories = glob.glob(data_dir + 'test/tomo*')
    directories.sort()
    result = []
    for d in directories:
        name = d[max(d.rfind('\\'), d.rfind('/'))+1:]
        result.append(load_one_measurement(name, False, False))
    return result
'''
General model definition
'''
# Function is used below, I ran into issues with multiprocessing if it was not a top-level function
model_parallel = None
def infer_internal_single_parallel(data):    
    try:
        global model_parallel
        if model_parallel is None:
            model_parallel= dill_load(temp_dir+'parallel.pickle')
        data.load_to_memory()
        return_data = model_parallel._infer_single(data)
        return_data.unload()
        return return_data
    except Exception as err:
        import traceback
        print(traceback.format_exc())     
        raise

@dataclass
class Model(BaseClass):
    # Loads one or more cryoET measuerements
    state: int = field(init=False, default=0) # 0: untrained, 1: trained
    quiet: bool = field(init=False, default=True)
    run_in_parallel: bool = field(init=False, default=False)    
    seed: object = field(init=True, default=None)

    def _check_constraints(self):
        assert(self.state>=0 and self.state<=1)

    def train(self, train_data, validation_data):
        if self.state>1:
            return
        train_data = copy.deepcopy(train_data)
        validation_data = copy.deepcopy(validation_data)
        for d in train_data:
            d.unload()
        for d in validation_data:
            d.unload()
        self._train(train_data, validation_data)
        self.state = 1
        self.check_constraints()        

    def _train_real(self, real_data, return_inferred_labels, test_data):
        pass

    def infer(self, test_data):
        assert self.state == 1
        test_data = copy.deepcopy(test_data)
        for t in test_data:
            t.labels  = pd.DataFrame()
        test_data = self._infer(test_data)
        for t in test_data:
            t.check_constraints()
        return test_data

    def _infer(self, test_data):
        # Subclass must implement this OR _infer_single
        if self.run_in_parallel:
            claim_gpu('')
            with multiprocess.Pool(recommend_n_workers()) as p:
                dill_save(temp_dir+'parallel.pickle', self)
                result = p.starmap(infer_internal_single_parallel, zip(test_data))            
        else:
            result = []
            for xx in test_data:     
                t = time.time()
                x = copy.deepcopy(xx)                
                was_loaded = (x.loaded_state=='memory')                
                if not was_loaded: x.load_to_memory()
                profile_print(x.name + ' loading: ' + str(time.time()-t))
                x = self._infer_single(x)
                if not was_loaded: x.unload()
                result.append(x)
                profile_print(x.name + ' total infer time: ' + str(time.time()-t))
        result = self._post_process(result)
        return result

    def _post_process(self, result):
        return result

def mark_tf_pn(data, reference_data, mark_false_negative=False):
    assert not mark_false_negative # todo
    for d,r in zip(data,reference_data):
        assert d.name==r.name
        #d.labels_unfiltered['tf_pn'] = np.nan
        for row_d in range(len(d.labels_unfiltered)):
            is_true_positive = False
            for row_r in range(len(r.labels)):
                coordinate_cols = ['z', 'y', 'x']
                loc_d = d.labels_unfiltered[coordinate_cols][row_d:row_d+1].values
                loc_r = r.labels[coordinate_cols][row_r:row_r+1].values
                distance = np.linalg.norm(loc_d - loc_r)*r.voxel_spacing
                if distance<1000:
                    is_true_positive = True
                    break
            if is_true_positive:
                d.labels_unfiltered.at[row_d,'tf_pn'] = 0
            else:
                d.labels_unfiltered.at[row_d,'tf_pn'] = 1


def create_submission_dataframe(submission_data, reference_data = load_all_test_data(), include_voxel_spacing = False):

     #submission = pd.read_csv(data_dir + '/sample_submission.csv')
    #print(submission)
    #submission = submission[0:0]
    #submission = submission.set_index("id")
    
    rows = []  # Collect rows as a list of lists or tuples
    #ind = 0
    
    for dat in submission_data:
        if len(dat.labels)==0:
            pass
            if include_voxel_spacing:
                rows.append([dat.name, -1,-1,-1,10,0])
            else:
                rows.append([dat.name, -1,-1,-1])
        else:
            assert(len(dat.labels)==1)
            lab = copy.deepcopy(dat.labels).reset_index()
            if include_voxel_spacing:
                rows.append([dat.name, lab['z'][0], lab['y'][0], lab['x'][0], dat.voxel_spacing, 1])
            else:
                rows.append([dat.name, lab['z'][0], lab['y'][0], lab['x'][0]])

    all_names = [d.name for d in reference_data]
    seen_names = [r[0] for r in rows]
    assert np.all([(name in all_names) for name in seen_names])
    for name in all_names:
        if not name in seen_names:
            if include_voxel_spacing:
                rows.append([name, -1,-1,-1,10,0])
            else:
                rows.append([name, -1,-1,-1])
    
    # Create a new DataFrame from collected rows
    if include_voxel_spacing:
        rows_df = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2", "Voxel spacing", "Has motor"])
    else:
        rows_df = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
    #rows_df = rows_df.set_index("id")

    return rows_df

def write_submission_file(submission_data):   

    rows_df = create_submission_dataframe(submission_data)
    print(rows_df)
    rows_df.to_csv(output_dir + 'submission.csv', index=False)


def score_competition_metric(data, reference_data):
    def distance_metric(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        thresh_ratio: float,
        min_radius: float,
    ):
        coordinate_cols = ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']
        label_tensor = solution[coordinate_cols].values.reshape(len(solution), -1, len(coordinate_cols))
        predicted_tensor = submission[coordinate_cols].values.reshape(len(submission), -1, len(coordinate_cols))
        # Find the minimum euclidean distances between the true and predicted points
        solution['distance'] = np.linalg.norm(label_tensor - predicted_tensor, axis=2).min(axis=1)
        # Convert thresholds from angstroms to voxels
        solution['thresholds'] = solution['Voxel spacing'].apply(lambda x: (min_radius * thresh_ratio) / x)
        solution['predictions'] = submission['Has motor'].values
        solution.loc[(solution['distance'] > solution['thresholds']) & (solution['Has motor'] == 1) & (submission['Has motor'] == 1), 'predictions'] = 0
        return solution['predictions'].values
        
    def score(solution: pd.DataFrame, submission: pd.DataFrame, min_radius: float, beta: float) -> float:
        """
        Parameters:
        solution (pd.DataFrame): DataFrame containing ground truth motor positions.
        submission (pd.DataFrame): DataFrame containing predicted motor positions.
    
        Returns:
        float: FBeta score.
    
        Example
        --------
        >>> solution = pd.DataFrame({
        ...     'tomo_id': [0, 1, 2, 3],
        ...     'Motor axis 0': [-1, 250, 100, 200],
        ...     'Motor axis 1': [-1, 250, 100, 200],
        ...     'Motor axis 2': [-1, 250, 100, 200],
        ...     'Voxel spacing': [10, 10, 10, 10],
        ...     'Has motor': [0, 1, 1, 1]
        ... })
        >>> submission = pd.DataFrame({
        ...     'tomo_id': [0, 1, 2, 3],
        ...     'Motor axis 0': [100, 251, 600, -1],
        ...     'Motor axis 1': [100, 251, 600, -1],
        ...     'Motor axis 2': [100, 251, 600, -1]
        ... })
        >>> score(solution, submission, 1000, 2)
        0.3571428571428571
        """
    
        solution = solution.sort_values('tomo_id').reset_index(drop=True)
        submission = submission.sort_values('tomo_id').reset_index(drop=True)
    
        filename_equiv_array = solution['tomo_id'].eq(submission['tomo_id'], fill_value=0).values
    
        if np.sum(filename_equiv_array) != len(solution['tomo_id']):
            raise ValueError('Submitted tomo_id values do not match the sample_submission file')
    
        submission['Has motor'] = 1
        # If any columns are missing an axis, it's marked with no motor
        select = (submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] == -1).any(axis='columns')
        submission.loc[select, 'Has motor'] = 0
    
        cols = ['Has motor', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']
        assert all(col in submission.columns for col in cols)
    
        # Calculate a label of 0 or 1 using the 'has motor', and 'motor axis' values
        predictions = distance_metric(
            solution,
            submission,
            thresh_ratio=1.0,
            min_radius=min_radius,
        )
    
        return sklearn.metrics.fbeta_score(solution['Has motor'].values, predictions, beta=beta)

    row_df_sub = create_submission_dataframe(data, reference_data = reference_data)
    row_df_ref = create_submission_dataframe(reference_data, reference_data = reference_data, include_voxel_spacing=True)
    return score(row_df_ref, row_df_sub, 1000, 2)



# '''
# Scoring
# '''

# def mark_tf_pn(candidate_label, ref_label, mark_false_negatives = True, radius_adjust=0.5, allow_double_counting = False):
#     # 0: TP, 1: FP, 2:FN
#     output_label = copy.deepcopy(candidate_label)   
#     assert candidate_label.particle_id==ref_label.particle_id
#     reference_radius = particle_radius[candidate_label.particle_id]*radius_adjust
#     candidate_tree = sp.spatial.KDTree(candidate_label.zyx)
#     ref_tree = sp.spatial.KDTree(ref_label.zyx)

#     # Find true and false positives
#     match = candidate_tree.query_ball_tree(ref_tree, reference_radius)
#     #if not np.all([(len(x)<=1) for x in match]):
#     #    print('double match')
#     tf_pn = [0 if x else 1 for x in match]

#     # Find false negatives and double counting
#     match = ref_tree.query_ball_tree(candidate_tree, reference_radius)
#     for i in range(len(match)):
#         m = match[i]
#         if len(m)==1:
#             pass # true positives -> already marked
#         elif len(m)==0:
#             # false negative
#             if mark_false_negatives:
#                 output_label.zyx = np.concatenate((output_label.zyx, ref_label.zyx[i:i+1,:]))
#                 tf_pn.append(2)
#         else:
#             if not allow_double_counting:
#                 # double counting, make all extra points false positive iso true positive
#                 for j in range(len(m)-1):
#                     tf_pn[m[j]] = 1

#     tf_pn = np.array(tf_pn)
#     output_label.notes['tf_pn'] = tf_pn
#     output_label.check_constraints()
#     assert(output_label.zyx.shape == (tf_pn.shape[0],3))   
    
#     return output_label

# def animate_tf_pn(data1,data2,scale,marked_labels):
    
#     r=30
#     fig, ax = plt.subplots(1,2,figsize=(15,7.5))
#     plt.sca(ax[0])
#     im = dict()
#     im1 = plt.imshow(data1[0,:,:],aspect='auto',interpolation='none', cmap='gray');
#     plt.clim([np.percentile(data1.flatten(), 0.3), np.percentile(data1.flatten(), 99.7)])
#     #plt.colorbar()
#     im3 = plt.scatter(200,200, s=250, facecolors='none', edgecolors='r')
#     #plt.colorbar()
#     plt.sca(ax[1])
#     im2 = plt.imshow(data2[0,:,:],aspect='auto',interpolation='none', cmap='gray');
#     plt.clim([np.percentile(data2.flatten(), 0.3), np.percentile(data2.flatten(), 99.7)])
#     im4 = plt.scatter(200,200, s=250, facecolors='none', edgecolors='r')
#     im = [im1,im2]
#     #plt.colorbar()
#     plt.close()

#     cvals = [ [0,0,1], [1,1,0], [1,0,0] ]
    
#     # animation function. This is called sequentially
#     def animate(i):
#         im1.set_data(data1[i,:,:])
#         im2.set_data(data2[i,:,:])
#         x = []; y = []; z=i*scale[0]; rad = []; colors = [];
#         for lab in marked_labels:
#             for row in range(lab.zyx.shape[0]):
#                 if np.abs(z-lab.zyx[row,0])<particle_radius[lab.particle_id]:
#                     x.append(lab.zyx[row,1]/scale[1])
#                     y.append(lab.zyx[row,2]/scale[2])
#                     rad.append( 2.5*np.sqrt(particle_radius[lab.particle_id]**2 - (z-lab.zyx[row,0])**2) )
#                     colors.append(cvals[lab.notes['tf_pn'][row]])
#         im3.set_offsets(np.stack((np.array(y),np.array(x))).T)
#         im3.set_sizes(rad)
#         im3.set_edgecolor(colors)
#         im4.set_offsets(np.stack((np.array(y),np.array(x))).T)
#         im4.set_sizes(rad)
#         im4.set_edgecolor(colors)
#         return [im1,im2]
    
#     # call the animator. blit=True means only re-draw the parts that have changed.
#     anim = animation.FuncAnimation(fig, animate, 
#                                    frames=data1.shape[0], interval=200, blit=True)
    
#     display(IPython.display.HTML(anim.to_html5_video()))


# '''
# General model definition
# '''
# # Function is used above, I ran into issues with multiprocessing if it was not a top-level function
# model_parallel = None
# def infer_internal_single_parallel(data):    
#     try:
#         global model_parallel
#         if model_parallel is None:
#             model_parallel= dill_load(output_loc()+'parallel.pickle')
#         data.unpack()
#         for p in model_parallel.preprocessors:
#             p.process(data)
#         return_data = copy.deepcopy(model_parallel._infer_single(data))
#         data.pack()
#         return_data.pack()
#         return return_data
#     except Exception as err:
#         import traceback
#         print(traceback.format_exc())     
#         raise

# @dataclass
# class Model(BaseClass):
#     # Loads one or more cryoET measuerements
#     state: int = field(init=False, default=0) # 0: untrained, 1: synthetic trained, 2: fully trained
#     quiet: bool = field(init=False, default=True)
#     run_in_parallel: bool = field(init=False, default=False)
#     particles_to_do: list = field(init=False, default_factory = lambda:[0,1,2,3,4,5])
#     cache_name: str = field(init=False, default='model')
#     preprocessors: list = field(init=False, default_factory = list)
#     seed: int = field(init=False, default=42)
#     fill_infer_cache: bool = field(init=False, default=False)
#     infer_cache: dict = field(init=False, default_factory=dict)

#     @property
#     def hyperparameters(self):
#         return [self._get_hyperparameters_per_particle(ind) for ind in range(len(particle_names))]

#     @hyperparameters.setter
#     def hyperparameters(self, value):
#         for ind in range(len(particle_names)):
#             self._set_hyperparameters_per_particle(ind,value[ind])
#         self.check_constraints()
#         assert value == self.hyperparameters

#     def get_hyperparameters_metainfo(self):
#         # Dict per particle per particle:
#         # -min: Minimum value for this hyperparameter
#         # -max: Maximum value for this hyperparameter
#         # -change: Suggested change value
#         return [self._get_hyperparameters_metainfo_per_particle(ind) for ind in range(len(particle_names))]

#     def _check_constraints(self):
#         assert(self.state>=0 and self.state<=2)

#     def train_synthetic(self, synthetic_data):
#         if self.state > 0:
#             return
#         for d in synthetic_data:
#             d.unpack()
#         self._train_synthetic(synthetic_data)
#         for d in synthetic_data:
#             d.pack()
#         self.state = 1
#         self.check_constraints()
        
#     def _train_synthetic(self, synthetic_data):
#         pass

#     def train_real(self, real_data, return_inferred_labels=False, test_data = None):
#         if self.state>1:
#             return
#         real_data = copy.deepcopy(real_data)
#         for d in real_data:
#             d.unpack()
#             for p in self.preprocessors:
#                 p.process(d)
#         labels_output = self._train_real(real_data, return_inferred_labels, test_data)
#         for d in real_data:
#             d.pack()
#         self.state = 2
#         self.check_constraints()
#         if return_inferred_labels:
#             return labels_output

#     def _train_real(self, real_data, return_inferred_labels, test_data):
#         pass

#     def infer(self, test_data):
#         assert self.state == 2
#         test_data = copy.deepcopy(test_data)
#         for t in test_data:
#             t.labels  = []
#         test_data = self._infer(test_data)
#         for t in test_data:
#             t.check_constraints()
#         return test_data

#     def _infer(self, test_data):
#         # Subclass must implement this OR _infer_single
#         if self.run_in_parallel:
#             claim_gpu('')
#             p = multiprocess.Pool(recommend_n_workers())
#             #test_data = p.starmap(infer_internal_single_parallel, zip(test_data, itertools.repeat(self)))
#             dill_save(output_loc()+'parallel.pickle', self)
#             test_data = p.starmap(infer_internal_single_parallel, zip(test_data))
#             p.close()
#             return test_data
#         else:
#             result = []
#             for xx in test_data:
#                 if xx.name in self.infer_cache.keys():
#                     xx.labels = copy.deepcopy(self.infer_cache[xx.name])
#                     result.append(xx)
#                 else:
#                     x = copy.deepcopy(xx)
#                     x.unpack()
#                     for p in self.preprocessors:
#                         p.process(x)
#                     xx = copy.deepcopy(self._infer_single(x))
#                     xx.pack()
#                     result.append(xx)
#                     x.pack()
#                     if self.fill_infer_cache:  
#                         self.infer_cache[xx.name] = copy.deepcopy(xx.labels)
#             return result        
        
# def evaluate_model_in_sample(model, synthetic_data, train_data):
#     model = copy.deepcopy(model)
#     model.train_synthetic(synthetic_data)

#     train_data = copy.deepcopy(train_data)
#     for dat in train_data:
#         for p in model.preprocessors:
#             p.process(dat)
#     model.preprocessors = []
    
#     labels_output = model.train_real(copy.deepcopy(train_data), return_inferred_labels=True)
#     predicted_data = copy.deepcopy(train_data)
#     for lab,d in zip(labels_output, predicted_data):
#         d.labels = lab
#     #predicted_data = model.infer(train_data)
#     combined_score,score = evaluate_predictions(predicted_data, train_data)
#     return combined_score, score, predicted_data, model

# def evaluate_model_cv(model, synthetic_data, train_data, pass_test_data=False, n_to_do=None, check_low_threshold=False,augment_test_data=False):
#     print('Sure you don''t want alt?')
#     model = copy.deepcopy(model)
#     model.train_synthetic(synthetic_data)

#     train_data = copy.deepcopy(train_data)
#     # for dat in train_data:
#     #     for p in model.preprocessors:
#     #         p.process(dat)
#     # model.preprocessors = []

#     if n_to_do is None:
#         n_to_do = len(train_data)
        
#     predicted_data = []
#     all_test_data = []
#     predicted_data_low_threshold = []
#     model_list= []
#     for i in range(n_to_do):
#         this_train_data = copy.deepcopy(train_data);
#         del this_train_data[i]        
#         this_model = copy.deepcopy(model)
#         this_model.seed = this_model.seed + i
#         this_model.cache_name = 'cv'+str(i)
#         if pass_test_data:
#             this_model.train_real(this_train_data, return_inferred_labels=False, test_data=this_test_data)
#         else:
#             this_model.train_real(this_train_data, return_inferred_labels=False)
#         if check_low_threshold:
#             this_model2 = copy.deepcopy(this_model)
#             this_model2.thresholds = [0.01]*6
#             this_predicted_data = this_model2.infer(copy.deepcopy(train_data[i:i+1]))
#             predicted_data_low_threshold = predicted_data_low_threshold  + this_predicted_data

#         this_test_data = train_data[i:i+1]
#         if augment_test_data:
#             for i_data in range(len(this_test_data)):
#                 for k in [1,2,3]:
#                     dat = copy.deepcopy(this_test_data[i_data])
#                     for kk in range(k):
#                         dat.rot90_xy()
#                     this_test_data.append(dat)
#         this_predicted_data = this_model.infer(this_test_data)
#         predicted_data = predicted_data + this_predicted_data
#         all_test_data = all_test_data + this_test_data
#         model_list.append(this_model)
#     if check_low_threshold:
#         print('threshold=0.01, not augmented')
#         print(evaluate_predictions(predicted_data_low_threshold, train_data[:n_to_do]))
#     combined_score,scores = evaluate_predictions(predicted_data, all_test_data)
#     return combined_score, scores, predicted_data, model_list

# def evaluate_model_cv_alt(model, synthetic_data, train_data, pass_test_data=False, n_to_do=3,augment_test_data=False,given_model_list=None,skip_evaluate=False,compute_in_sample=False):
#     if given_model_list is None:
#         model = copy.deepcopy(model)
#         model.train_synthetic(synthetic_data)

#     train_data = copy.deepcopy(train_data)
#     # for dat in train_data:
#     #     for p in model.preprocessors:
#     #         p.process(dat)
#     # model.preprocessors = []
    
#     assert n_to_do <= 3
        
#     predicted_data = []
#     all_test_data = []
#     predicted_data_low_threshold = []
#     model_list= []
#     predicted_data_in_sample = []
#     for i in range(n_to_do):
#         this_train_data = copy.deepcopy(train_data);
#         del this_train_data[2*i+1]       
#         del this_train_data[2*i]  
#         this_test_data = train_data[2*i:2*i+2]
#         if given_model_list is None:
#             this_model = copy.deepcopy(model)                        
#             this_model.seed = this_model.seed + i
#             this_model.cache_name = 'cv'+str(i)
#         else:
#             this_model = copy.deepcopy(given_model_list[i])
#         if pass_test_data:
#             this_model.train_real(this_train_data, return_inferred_labels=False, test_data=this_test_data)
#         else:
#             this_model.train_real(this_train_data, return_inferred_labels=False)

        
#         if augment_test_data:
#             for i_data in range(len(this_test_data)):
#                 for k in [1,2,3]:
#                     dat = copy.deepcopy(this_test_data[i_data])
#                     for kk in range(k):
#                         dat.rot90_xy()
#                     this_test_data.append(dat)
#         this_predicted_data = this_model.infer(this_test_data)
#         if compute_in_sample:
#             predicted_data_in_sample.append(this_model.infer(this_train_data))
#         predicted_data = predicted_data + this_predicted_data
#         all_test_data = all_test_data + this_test_data
#         #print([t.name for t in this_train_data])
#         #print([t.name for t in this_test_data])
#         model_list.append(this_model)
#     if not skip_evaluate:
#         combined_score,scores = evaluate_predictions(predicted_data, all_test_data)
#     else:
#         combined_score = []
#         scores = []
#     if compute_in_sample:
#         return combined_score, scores, predicted_data, model_list, predicted_data_in_sample
#     else:
#         return combined_score, scores, predicted_data, model_list

# def evaluate_predictions(predicted_data, reference, radius_adjust = 0.5, mark_false_negatives=True):
#     assert reference is None or (len(predicted_data) == len(reference))
#     scores = pd.DataFrame(0., index=range(6), columns=["name", "tp", "fp", "fn", "precision", "recall", "score"])
#     scores["name"] = particle_names;
#     for i in range(len(predicted_data)):
#         assert predicted_data[i].name == reference[i].name
#         for j in range(len(predicted_data[i].labels)):  
#             particle_id = predicted_data[i].labels[j].particle_id
#             if not reference is None:
#                 predicted_data[i].labels[j] = mark_tf_pn(predicted_data[i].labels[j], reference[i].labels[particle_id],radius_adjust=radius_adjust,mark_false_negatives=mark_false_negatives);
#             scores.loc[particle_id, "tp"] += np.sum(predicted_data[i].labels[j].notes['tf_pn']==0)
#             scores.loc[particle_id, "fp"] += np.sum(predicted_data[i].labels[j].notes['tf_pn']==1)
#             scores.loc[particle_id, "fn"] += np.sum(predicted_data[i].labels[j].notes['tf_pn']==2)    
#     for j in range(scores.shape[0]):
#         if scores.loc[j, "tp"]==0:
#             scores.loc[j, "precision"] = 0
#             scores.loc[j, "recall"] = 0
#             scores.loc[j, "score"] = 0
#         else:
#             scores.loc[j, "precision"] = scores.loc[j, "tp"] / (scores.loc[j, "tp"] + scores.loc[j, "fp"])
#             scores.loc[j, "recall"] = scores.loc[j, "tp"] / (scores.loc[j, "tp"] + scores.loc[j, "fn"])
#             beta = 4
#             scores.loc[j, "score"] = (1+beta**2)*(scores.loc[j, "precision"] * scores.loc[j, "recall"])/(beta**2*scores.loc[j, "precision"] + scores.loc[j, "recall"])
#     #display(scores)
#     weights = np.array([1,1e-20,2,1,2,1]);
#     combined_score = np.sum(scores["score"].to_numpy() * weights) / np.sum(weights)
#     #print('Combined score:', combined_score)
#     return combined_score,scores

# '''
# Baseline models
# '''

# def baseline_model(ind):
#     import cz_unet2_ensemble as czu
#     import cz_model as czm
#     import cz_processors as czp
#     model = czu.UNetModel()
#     model.preprocessors = []
#     model.quiet = False
#     model.print_epochs = False
#     model.store_heat_maps = False
#     model.plot_every = 500
#     model.test_loss_every = 20
#     model.save_every = 1e6
    
#     mask_model = czm.MaskModel2()
#     mask_model.quiet = True
#     mask_model.register_p_matrices=False
    
#     union_model = czm.UnionModel()
#     union_model.quiet = True
#     union_model.models_internal = [model, mask_model]
#     union_model.run_in_parallel = False
#     union_model.seed = int(np.random.default_rng(seed=None).integers(1e6))


#     method = 'recursive'
#     for ii in range(6):
#         model.heat_map_to_locations[ii].threshold = 0.95
#         model.heat_map_to_locations[ii].gaussian_smoothing_heat_map = 0
#         model.heat_map_to_locations[ii].cluster_filter_limits = (0, np.inf)
#         model.heat_map_to_locations[ii].additional_threshold_filter = False
#         model.heat_map_to_locations[ii].threshold_search_ratio = 1.
#         model.heat_map_to_locations[ii].radius_dilate = 3
#         model.heat_map_to_locations[ii].radius_kmeans = 22
#         model.heat_map_to_locations[ii].method = 'kmeans'

#     union_model.models_internal[0].heat_map_to_locations[0].radius_kmeans = 10
#     model.heat_map_to_locations[0].radius_dilate = 1
#     model.heat_map_to_locations[0].threshold = 0.1
#     model.heat_map_to_locations[0].locator_kmeans = 2

#     union_model.models_internal[0].heat_map_to_locations[2].radius_kmeans = np.inf
#     model.heat_map_to_locations[2].threshold = 0.3

#     union_model.models_internal[0].heat_map_to_locations[3].radius_kmeans = 20
#     model.heat_map_to_locations[3].threshold = 0.9

#     union_model.models_internal[0].heat_map_to_locations[4].radius_kmeans = 15
#     model.heat_map_to_locations[4].threshold = 0.9

#     union_model.models_internal[0].heat_map_to_locations[5].radius_kmeans = np.inf
#     model.heat_map_to_locations[5].threshold = 0.9

#     union_model.relative_radius = [3.5, np.inf, np.inf, 2, 3.5, np.inf]

#     union_model.stats_model_threshold = [-1, -1, 0.025, 1e-10, 1e-10, -1]

#     if isinstance(ind, int):
#         union_model.particles_to_do = [ind]
#     else:
#         union_model.particles_to_do = ind
#         if len(ind)>0:
#             print('extending')
#             diff = 300
#             union_model.models_internal[0].n_epochs = union_model.models_internal[0].n_epochs+3*diff+600
#             union_model.models_internal[0].epoch_end_scheduler = union_model.models_internal[0].epoch_end_scheduler+3*diff+600
#             union_model.models_internal[0].epoch_start_scheduler = union_model.models_internal[0].epoch_start_scheduler+diff+600
#             union_model.models_internal[0].learning_rate_final = 1e-6
#             model.heat_map_to_locations[2].threshold = 0.7


#     # if ind==0:
#     #     union_model.relative_radius = 3.5
#     #     #union_model.models_internal[0].heat_map_to_locations[ind].radius_kmeans = 10
#     #     #model.heat_map_to_locations[ind].radius_dilate = 1
#     #     #model.heat_map_to_locations[ind].threshold = 0.5
#     # elif ind==2:
#     #     union_model.relative_radius = np.inf
#     #     #union_model.models_internal[0].heat_map_to_locations[ind].radius_kmeans = np.inf
#     #     #model.heat_map_to_locations[ind].threshold = 0.8
#     # elif ind==3:
#     #     union_model.relative_radius = 2
#     #     #union_model.models_internal[0].heat_map_to_locations[ind].radius_kmeans = 20
#     #     #model.heat_map_to_locations[ind].threshold = 0.8
#     # elif ind==4:
#     #     union_model.relative_radius = 3.5 
#     #     #union_model.models_internal[0].heat_map_to_locations[ind].radius_kmeans = 15 
#     #     #model.heat_map_to_locations[ind].threshold = 0.9
#     # elif ind==5:
#     #     union_model.relative_radius = np.inf
#     #     #print(union_model.models_internal[0].heat_map_to_locations[ind].radius_kmeans)
#     #     #union_model.models_internal[0].heat_map_to_locations[ind].radius_kmeans = np.inf
#     #     #print(model.heat_map_to_locations[ind].threshold)
#     #     #model.heat_map_to_locations[ind].threshold = 0.8

#     return union_model