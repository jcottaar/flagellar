import plotly.express as px
from PIL import Image, ImageDraw
import random
import seaborn as sns
from matplotlib.patches import Rectangle
import yaml
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import math
import flg_support as fls
import shutil
from dataclasses import dataclass, field, fields
import copy
import subprocess
import importlib
import flg_preprocess
import matplotlib


ran_with_albs = False

@dataclass
class YOLOModel(fls.BaseClass):
    preprocessor: object = field(init=True, default_factory=flg_preprocess.Preprocessor2)
    seed = None
    
    #Input
    n_ensemble = 1
    img_size = 640
    prevent_ultralytics_resize = True
    n_epochs = 50
    model_name = 'yolov9s'
    use_pretrained_weights = True    
    fix_norm_bug = True
    box_size = 18
    trust = 4
    trust_neg = 0
    remove_suspect_areas = True
    negative_slice_ratio = 0.
    negative_label_threshold = 0.6

    alternative_slice_selection = True
    trust_expanded = 6
    forbidden_range = 20 # if there is a motor inside forbidden_range but outside trust_expanded in z, discard this slice

    patience=10
    use_best_epoch = True # else use last
    lr0=0.001
    lrf=0.01
    cos_lr = True
    weight_decay = 0.0005
    dropout= 0.0
    momentum=0.937
    multi_scale_training = False
    

    box=7.5
    
    
    
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    translate = 0.1
    scale = 0.5
    fliplr = 0.5
    flipud = 0.5
    degrees = 0.0    
    shear = 0.0
    perspective = 0.0
    mosaic = 1.0
    close_mosaic = 10
    mixup = 0.2
    auto_augment = None
    erasing = 0.4
    copy_paste = 0.0
    crop_fraction = 1.0
    use_albumentations = False

    # infer
    confidence_threshold = 0.
    relative_confidence_threshold = 0.2
    final_relative_confidence_threshold = True
    concentration = 1

    # trained
    trained_model = 0
    train_results = 0
    
    def __post_init__(self):
        super().__post_init__()
        #self.preprocessor.scale_std = False
        #self.preprocessor.scale_percentile = True
        #self.preprocessor.return_uint8 = True
            

    
   
    def train(self,train_data,validation_data):


        global ran_with_albs
        if self.use_albumentations:
            ran_with_albs = True
            if not fls.env=='kaggle':
                print(subprocess.run(["pip", "install", "albumentations"]))
        else:
            assert not ran_with_albs
            print(subprocess.run(["pip", "uninstall", "-y",  "albumentations"]))
        import ultralytics
        importlib.reload(ultralytics)

        # Preprocess data
        
        # Output directories for YOLO dataset (adjust as needed)
        yolo_dataset_dir = fls.temp_dir + '/yolo_dataset/'
        yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
        yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
        yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
        yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")
        
        # Create necessary directories
        for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
            try: shutil.rmtree(dir_path)
            except: pass
            os.makedirs(dir_path, exist_ok=True)
        
        # Define the preprocessing function to extract slices, normalize, and generate YOLO annotations.
        def prepare_yolo_dataset(trust):
            """
            Extract slices containing motors and save images with corresponding YOLO annotations.
            
            Steps:
            - Load the motor labels.
            - Perform a train/validation split by tomogram.
            - For each motor, extract slices in a range (± trust parameter).
            - Normalize each slice and save it.
            - Generate YOLO format bounding box annotations with a fixed box size.
            - Create a YAML configuration file for YOLO training.
            
            Returns:
                dict: A summary containing dataset statistics and file paths.
            """

            #train_data_filtered = []
            #for d in train_data:
            #    if len(d.labels)>0:
            #        train_data_filtered.append(d)
            #validation_data_filtered = []
            #for d in validation_data:
            #    if len(d.labels)>0:
            #        validation_data_filtered.append(d)
            train_data_filtered = copy.deepcopy(train_data)
            validation_data_filtered = copy.deepcopy(validation_data)

            # Set train and test
            np.random.shuffle(train_data_filtered)

            
            # Helper function to process a list of tomograms
            def process_tomogram_set(data_list, images_dir, labels_dir, set_name):
                
                def write_image(dest_filename, normalized_img, x_center, y_center, x_width, y_width, poi_x, poi_y):

                    def find_coords(cur_size, target_size, poi):
                        poi = np.round(poi).astype(int)
                        c_start = poi - target_size//2
                        c_end = poi + target_size//2
                        if c_start<0:
                            c_end = c_end-c_start
                            c_start = 0
                        if c_end>cur_size:
                            c_start = c_start-(c_end-cur_size)
                            c_end = cur_size
                        assert c_end - c_start == target_size
                        scaling = target_size/cur_size
                        #print(c_start,c_end,scaling)
                        return c_start, c_end, scaling
                        

                    # Pad if necessary
                    if self.prevent_ultralytics_resize:
                        after_x = max(0, self.img_size-normalized_img.shape[1])
                        after_y = max(0, self.img_size-normalized_img.shape[0])
                        normalized_img = np.pad(normalized_img, ((0,after_y),(0,after_x)), mode='constant', constant_values=127)

                        x_start, x_end, x_scaling = find_coords(normalized_img.shape[1], self.img_size, poi_x)
                        y_start, y_end, y_scaling = find_coords(normalized_img.shape[0], self.img_size, poi_y)
                        normalized_img = normalized_img[y_start:y_end, x_start:x_end]
                        assert normalized_img.shape == (self.img_size, self.img_size)
                    else:
                        x_start = 0; y_start = 0;
                    
                    dest_path = os.path.join(images_dir, dest_filename)
                    Image.fromarray(normalized_img).save(dest_path)        
                    
                    x_center = np.array(x_center)
                    y_center = np.array(y_center)
                    x_width = np.array(x_width)
                    y_width = np.array(y_width)
    
                    label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                    # plt.pause(0.001)
                    # plt.figure()
                    # plt.imshow(normalized_img, cmap='bone')
                    # plt.title(data.name)
                    with open(label_path, 'w') as f:
                        for ii in range(x_center.shape[0]):
                            img_width, img_height = (normalized_img.shape[1], normalized_img.shape[0])
                            x_center_norm = (x_center[ii]-x_start)/ img_width
                            y_center_norm = (y_center[ii]-y_start) / img_height
                            box_width_norm = x_width[ii] / img_width
                            box_height_norm = y_width[ii] / img_height
                            f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                            x1 = (x_center_norm-box_width_norm/2)*normalized_img.shape[1]
                            x2 = (x_center_norm+box_width_norm/2)*normalized_img.shape[1]
                            y1 = (y_center_norm-box_height_norm/2)*normalized_img.shape[0]
                            y2 = (y_center_norm+box_height_norm/2)*normalized_img.shape[0]
                            #plt.gca().add_patch(matplotlib.patches.Rectangle((x1,y1), x2-x1,y2-y1, alpha=0.5, facecolor='blue'))

                
                if self.alternative_slice_selection:
                    neg_slice_counter = 0.
                    neg_ind = 0
                    neg_slice_selector = np.random.default_rng(seed=self.seed)
                    assert self.trust_expanded >= self.trust
                    for data in tqdm(data_list):
                        slices_to_do = []
                        for i_slice in range(data.data_shape[0]):
                            in_any_range = False
                            in_forbidden_range = False
                            for i_row in range(len(data.labels)):
                                dist = np.abs(data.labels['z'][i_row]-i_slice)
                                if np.abs(dist)<=self.trust:
                                    in_any_range = True
                                if np.abs(dist)>=self.trust_expanded and np.abs(dist)<=self.forbidden_range:
                                    in_forbidden_range = True
                            for i_row in range(len(data.negative_labels)):
                                if data.negative_labels['confidence'][i_row]>self.negative_label_threshold:
                                    dist = np.abs(data.negative_labels['z'][i_row]-i_slice)
                                    if np.abs(dist)<=self.trust_neg:
                                        in_any_range = True
                                if self.remove_suspect_areas and data.negative_labels['suspect'][i_row]==1.:
                                    dist = np.abs(data.negative_labels['z'][i_row]-i_slice)
                                    if np.abs(dist)<=self.forbidden_range:
                                        in_forbidden_range = True
                            if in_any_range and not in_forbidden_range:
                                slices_to_do.append(i_slice)
                        if len(slices_to_do)==0:
                            continue
                        dd = copy.deepcopy(data)
                        self.preprocessor.load_and_preprocess(dd, desired_original_slices = slices_to_do)
                        for i_z,z in enumerate(dd.slices_present):
                            normalized_img = dd.data[i_z,:,:]
                            dest_filename = f"{data.name}_z{z:04d}.jpg"                                            
    
                            
                            
                            x_center = []
                            y_center = []
                            x_width = []
                            y_width = []

                            x_poi = np.nan
                            y_poi = np.nan
                            for i_row in range(len(dd.labels)):
                                dist = np.abs(dd.labels['z'][i_row]-z)
                                if np.abs(dist)<=self.trust_expanded:
                                    x_center.append(dd.labels['x'][i_row])
                                    y_center.append(dd.labels['y'][i_row])                                    
                                    x_width.append(self.box_size)
                                    y_width.append(self.box_size)                                    
                                    if np.isnan(x_poi):
                                        x_poi = x_center[-1]
                                        y_poi = y_center[-1]
                            for i_row in range(len(dd.negative_labels)):
                                if dd.negative_labels['confidence'][i_row]>self.negative_label_threshold:
                                    dist = np.abs(dd.negative_labels['z'][i_row]-z)
                                    if np.abs(dist)<=self.trust_neg:                                 
                                        if np.isnan(x_poi):
                                            x_poi = dd.negative_labels['x'][i_row]
                                            y_poi = dd.negative_labels['y'][i_row]
                            assert not np.isnan(x_poi)
                            write_image(dest_filename, normalized_img, x_center, y_center, x_width, y_width, x_poi, y_poi)
                            neg_slice_counter += self.negative_slice_ratio
                        while neg_slice_counter>=1:
                            while True:
                                data_ind = neg_slice_selector.integers(0,len(data_list))
                                if len(data_list[data_ind].labels)==0:
                                    break
                            i_z = neg_slice_selector.integers(0,data_list[data_ind].data_shape[0])
                            ddd = copy.deepcopy(data_list[data_ind])
                            self.preprocessor.load_and_preprocess(ddd, desired_original_slices = [i_z])
                            normalized_img = ddd.data[0,:,:]
                            dest_filename = f"neg_{neg_ind}.jpg"
                            # dest_path = os.path.join(images_dir, dest_filename)
                            # Image.fromarray(normalized_img).save(dest_path)          
                            # label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                            # with open(label_path, 'w') as f:
                            #     pass
                            write_image(dest_filename, normalized_img, [],[],[],[], normalized_img.shape[0]//2, normalized_img.shape[1]//2)
                            neg_ind += 1
                            neg_slice_counter-=1
                    return 0,0
                else:
                    motor_counts = []
                    for d in data_list:
                         #Get motor annotations for the current tomogram
                        tomo_motors = d.labels
                        for _, motor in tomo_motors.iterrows():                        
                            motor_counts.append(
                                (d, 
                                 int(motor['z']), 
                                 int(motor['y']), 
                                 int(motor['x']),
                                 d.data_shape[0])
                            )
                    
                    print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
                    processed_slices = 0
                    
                    # Loop over each motor annotation
                    for d, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
                        dd=copy.deepcopy(d)
                        z_min = max(0, z_center - trust)
                        z_max_bound = min(z_max - 1, z_center + trust)
                        self.preprocessor.load_and_preprocess(dd, desired_original_slices = np.arange(z_min,z_max_bound+1).tolist())
                        for z in range(z_min, z_max_bound + 1):
                            normalized_img = dd.data[z-z_min,:,:]                                   
                            dest_filename = f"{d.name}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                            dest_path = os.path.join(images_dir, dest_filename)
                            Image.fromarray(normalized_img).save(dest_path)
                            
                            # Prepare YOLO bounding box annotation (normalized values)
                            img_width, img_height = (normalized_img.shape[1], normalized_img.shape[0])
                            x_center_norm = x_center / img_width
                            y_center_norm = y_center / img_height
                            box_width_norm = self.box_size / img_width
                            box_height_norm = self.box_size / img_height

                            label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                            with open(label_path, 'w') as f:
                                f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                            
                            processed_slices += 1                    
                    
                    return processed_slices, len(motor_counts)
            
            # Process training tomograms
            train_slices, train_motors = process_tomogram_set(train_data_filtered, yolo_images_train, yolo_labels_train, "training")
            # Process validation tomograms
            val_slices, val_motors = process_tomogram_set(validation_data_filtered, yolo_images_val, yolo_labels_val, "validation")
            
            # Generate YAML configuration for YOLO training
            yaml_content = {
                'path': yolo_dataset_dir,
                'train': 'images/train',
                'val': 'images/val',
                'names': {0: 'motor'}
            }
            with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
                     
            return {
                "dataset_dir": yolo_dataset_dir,
                "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
                "train_tomograms": len(train_data_filtered),
                "val_tomograms": len(validation_data_filtered),
                "train_motors": train_motors,
                "val_motors": val_motors,
                "train_slices": train_slices,
                "val_slices": val_slices
            }

        # Set random seeds for reproducibility
        fls.prep_pytorch(self.seed, True, False)
        
        # Run the preprocessing
        summary = prepare_yolo_dataset(self.trust)
        print(f"\nPreprocessing Complete:")
        print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
        print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
        print(f"- Dataset directory: {summary['dataset_dir']}")
        print(f"- YAML configuration: {summary['yaml_path']}")
        print("\nReady for YOLO training!")
        
        yolo_weights_dir = fls.temp_dir + '/yolo_weights/'
        def fix_yaml_paths(yaml_path):
            """
            Fix the paths in the YAML file to match the actual Kaggle directories.
            
            Args:
                yaml_path (str): Path to the original dataset YAML file.
                
            Returns:
                str: Path to the fixed YAML file.
            """

            print(f"Fixing YAML paths in {yaml_path}")
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if 'path' in yaml_data:
                yaml_data['path'] = yolo_dataset_dir
            
            fixed_yaml_path = fls.temp_dir + "fixed_dataset.yaml"
            with open(fixed_yaml_path, 'w') as f:
                yaml.dump(yaml_data, f)
            
            print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")
            return fixed_yaml_path

        def plot_dfl_loss_curve(run_dir):
            """
            Plot the DFL loss curves for training and validation, marking the best model.
            
            Args:
                run_dir (str): Directory where the training results are stored.
            """
            results_csv = os.path.join(run_dir, 'results.csv')
            if not os.path.exists(results_csv):
                print(f"Results file not found at {results_csv}")
                return
            
            results_df = pd.read_csv(results_csv)
            train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
            val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
            
            if not train_dfl_col or not val_dfl_col:
                print("DFL loss columns not found in results CSV")
                print(f"Available columns: {results_df.columns.tolist()}")
                return
            
            train_dfl_col = train_dfl_col[0]
            val_dfl_col = val_dfl_col[0]
            
            best_epoch = results_df[val_dfl_col].idxmin()
            best_val_loss = results_df.loc[best_epoch, val_dfl_col]
            
            plt.figure(figsize=(10, 6))
            plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
            plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
            plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                        label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
            plt.xlabel('Epoch')
            plt.ylabel('DFL Loss')
            plt.title('Training and Validation DFL Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
            plt.savefig(plot_path)
            #plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png'))
            
            print(f"Loss curve saved to {plot_path}")
            plt.close()
            
            return best_epoch, best_val_loss

        def train_yolo_model(yaml_path, batch_size=12, img_size=640):
            """
            Train a YOLO model on the prepared dataset with optimized accuracy settings.
        
            Args:
                yaml_path (str): Path to the dataset YAML file.
                pretrained_weights_path (str): Path to pre-downloaded weights file.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                img_size (int): Image size for training.
        
            Returns:
                model (YOLO): Trained YOLO model.
                results: Training results.
            """
            model_list =[]
            train_results = []
            for i_ensemble in range(self.n_ensemble):
                fls.remove_and_make_dir(yolo_weights_dir)
                if self.model_name.startswith('yolo'):
                    model_setup_func = ultralytics.YOLO
                else:
                    model_setup_func = ultralytics.RTDETR
                if self.use_pretrained_weights:         
                    if not fls.env=='kaggle':
                        model = model_setup_func(self.model_name + '.pt')
                    else:
                        model = model_setup_func('/kaggle/usr/lib/ultralytics_for_offline_install_mine/' + self.model_name + '.pt')
                else:
                    model = model_setup_func(self.model_name + '.yaml')
    
                from ultralytics import settings
    
                # Update a setting
                settings.update({"mlflow": False})
                #settings.update({"seed": self.seed, "deterministic": True})

                if self.multi_scale_training:
                    batch_size = 6
    
                results = model.train(
                    data=yaml_path,
                    epochs=self.n_epochs,
                    batch=batch_size,
                    imgsz=self.img_size,
                    project=yolo_weights_dir,
                    name='motor_detector',
                    exist_ok=True,
                    patience=self.patience,  # Stop training if no improvement after 10 epochs
                    save_period=5,  # Save model every 5 epochs
                    val=True,
                    verbose=True,
                    optimizer="AdamW",  # AdamW optimizer for stability
                    lr0=self.lr0,  # Initial learning rate
                    lrf=self.lrf,  # Final learning rate factor
                    cos_lr=self.cos_lr,  # Use cosine learning rate decay
                    weight_decay=self.weight_decay,  # Prevent overfitting
                    dropout= self.dropout,
                    momentum=self.momentum,  # Momentum for better gradient updates
                    multi_scale = self.multi_scale_training,
                    close_mosaic=self.close_mosaic,  # Disable mosaic augmentation after 10 epochs
                    box = self.box,
                    workers=4,  # Speed up data loading
                    augment=True,  # Enable additional augmentations
                    amp=True,  # Mixed precision training for faster performance
                    seed=self.seed+100000*i_ensemble,
                    hsv_h=self.hsv_h, hsv_s=self.hsv_s, hsv_v=self.hsv_v, degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear, perspective=self.perspective, flipud=self.flipud, fliplr=self.fliplr, bgr=0.0, mosaic=self.mosaic, mixup=self.mixup, copy_paste=self.copy_paste, auto_augment=self.auto_augment, erasing=self.erasing, crop_fraction=self.crop_fraction,
                )

                run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
                
                # If function is defined, plot loss curves for better insights
                train_results.append(pd.read_csv(os.path.join(run_dir, 'results.csv')))
                #best_epoch_info = plot_dfl_loss_curve(run_dir)
                #if best_epoch_info:
                #    best_epoch, best_val_loss = best_epoch_info
                #    print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")

                if self.use_best_epoch:
                    model_list.append(ultralytics.YOLO(fls.temp_dir + 'yolo_weights/motor_detector/weights/best.pt'))
                else:
                    model_list.append(ultralytics.YOLO(fls.temp_dir + 'yolo_weights/motor_detector/weights/last.pt'))
                

            self.trained_model = model_list

        
        def prepare_dataset():
             
            yaml_data = {
                'path': yolo_dataset_dir,
                'train': 'images/train',
                'val': 'images/val',
                'names': {0: 'motor'}
            }
            new_yaml_path = fls.temp_dir + 'training.yaml'
            with open(new_yaml_path, 'w') as f:
                yaml.dump(yaml_data, f)
            print(f"Created new YAML at {new_yaml_path}")
            return new_yaml_path
    
        print("Starting YOLO training process...")
        yaml_path = prepare_dataset()
        print(f"Using YAML file: {yaml_path}")
        with open(yaml_path, 'r') as f:
            print(f"YAML contents:\n{f.read()}")
        
        print("\nStarting YOLO training...")
        train_yolo_model(yaml_path)
        

        
        print("\nTraining complete!")

        

    def infer(self,data):   
            
        def preload_image_batch(file_paths):
            """Preload a batch of images to CPU memory."""
            images = []
            for path in file_paths:
                img = cv2.imread(path)
                if img is None:
                    img = np.array(Image.open(path))
                images.append(img)
            return images
        
       
        #@fls.profile_each_line
        def process_tomogram(tomo_id, model, index=0, total=1):
            """
            Process a single tomogram and return the most confident motor detection.
            """
            print(f"Processing tomogram {tomo_id} ({index}/{total})")
            #tomo_dir = img_dir
            #slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

            all_detections = []
            for i_model,this_model in enumerate(self.trained_model):

                selected_indices = np.linspace(0, data.data.shape[0]-1, int(data.data.shape[0] // self.concentration))
                selected_indices = np.round(selected_indices).astype(int)
                #slice_files = [slice_files[i] for i in selected_indices]
                
                print(f"Processing {len(selected_indices)} out of {data.data.shape[0]} slices (CONCENTRATION={self.concentration})")
                
                streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
                
                for batch_start in range(0, len(selected_indices), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(selected_indices))
                    batch_indices = selected_indices[batch_start:batch_end]
                    
                    sub_batches = np.array_split(batch_indices, len(streams))
                    for i, sub_batch in enumerate(sub_batches):
                        if len(sub_batch) == 0:
                            continue
                        stream = streams[i % len(streams)]
                        with torch.cuda.stream(stream):
                            #sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                            sub_batch_slice_nums = sub_batch   
                            #print(sub_batch_slice_nums)
                            with torch.amp.autocast('cuda'), torch.no_grad():
                                data_in = []
                                for i_slice in sub_batch_slice_nums:
                                    data_in.append(data.data[i_slice,:,:,None])
                                    data_in[-1] = data_in[-1][:,:,[0,0,0]]
                                sub_results = this_model(data_in, verbose=False, conf=self.confidence_threshold, half=True, imgsz=image_size)
                            for j, result in enumerate(sub_results):
                                result = result.cpu()
                                all_conf = result.boxes.conf.numpy()#np.array([b.conf for b in result.boxes])
                                if len(all_conf)==0: continue
                                todo = np.logical_and(all_conf>self.confidence_threshold, all_conf>self.relative_confidence_threshold*np.max(all_conf))
                                if len(result.boxes) > 0:
                                    for box_idx in range(len(result.boxes.conf)):
                                        #if confidence >= self.confidence_threshold:
                                        if todo[box_idx]:
                                            x1, y1, x2, y2 = result.boxes.xyxy[box_idx].cpu().numpy()
                                            x_center = (x1 + x2) / 2 / data.resize_factor
                                            y_center = (y1 + y2) / 2 / data.resize_factor
                                            if self.preprocessor.apply_flipud:
                                                y_center = data.data_shape[1]-y_center
                                            if self.preprocessor.apply_transpose:
                                                tmp = x_center
                                                x_center = y_center
                                                y_center = tmp
                                            
                                            all_detections.append({
                                                'z': round(data.slices_present[sub_batch_slice_nums[j]]),
                                                'y': round(y_center),
                                                'x': round(x_center),
                                                'confidence': float(all_conf[box_idx]),
                                                'i_model': i_model
                                            })
                    torch.cuda.synchronize()

            #if len(all_detections)==0:
            #    all_detections = pd.DataFrame(all_detections, column = ['z', 'y', 'x', 'confidence', 'i_model'])
            all_detections = pd.DataFrame(all_detections, columns = ['z', 'y', 'x', 'confidence', 'i_model'])
            if self.final_relative_confidence_threshold:
                all_detections = all_detections[all_detections['confidence']>self.relative_confidence_threshold*np.max(all_detections['confidence'])]
            #final_detections = perform_3d_nms(all_detections)
            #final_detections.sort(key=lambda x: x['confidence'], reverse=True)

            #print(all_detections)
            return all_detections
            
            # if not final_detections:
            #     return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}

            # print(final_detections)
            # best_detection = final_detections[0]
            # return {
            #     'tomo_id': tomo_id,
            #     'Motor axis 0': round(best_detection['z']),
            #     'Motor axis 1': round(best_detection['y']),
            #     'Motor axis 2': round(best_detection['x'])
            # }
        

        preprocessor = copy.deepcopy(self.preprocessor)
        if not self.fix_norm_bug:
            preprocessor.scale_std = False
            preprocessor.scale_percentile = False
        preprocessor.load_and_preprocess(data) 

        if self.prevent_ultralytics_resize:
            #print('before ', data.data.shape)
            image_size = ( (np.ceil(data.data.shape[1]/32)*32).astype(int).item(), (np.ceil(data.data.shape[2]/32)*32).astype(int).item() )
            after_x = max(0, image_size[1]-data.data.shape[2])
            after_y = max(0, image_size[0]-data.data.shape[1])
            data.data = np.pad(data.data, ((0,0),(0,after_y),(0,after_x)), mode='constant', constant_values=127)
            assert data.data.shape == (data.data.shape[0], image_size[0], image_size[1])
            #print('after ', data.data.shape)
        else:
            image_size = self.img_size
        #print('image_size: ', image_size)
        
        cpu, device = fls.prep_pytorch(self.seed, True, False)
        BATCH_SIZE = 32

        for this_model in self.trained_model:
            this_model.to(device)
            this_model.fuse()

        results = []
        motors_found = 0

        return process_tomogram(data.name, self.trained_model, 1, 1)
        # if len(result)==0:
        #     return pd.DataFrame(columns=['z', 'y', 'x', 'confidence', 'i_model'])
        # else:
        #     return pd.DataFrame(result)
        # print(data.labels)
        # print(result)
        # raise 'stop'
        # return pd.DataFrame(result)
        # has_motor = not pd.isna(result['Motor axis 0'])
        # if has_motor:
        #     motors_found += 1
        #     print(f"Motor found in {data.name} at position: z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
        #     if not result['Motor axis 0']==-1:
        #         d = {'z': [result['Motor axis 0']], 'y': [result['Motor axis 1']], 'x': [result['Motor axis 2']]}
        #         data.labels = pd.DataFrame(d)                    
        #     else:
        #         data.labels = data.labels[0:0]
        # else:
        #     print(f"No motor detected in {tomo_id}")
        #     data.labels = data.labels[0:0]

        #return data
                