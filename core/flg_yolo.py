import plotly.express as px
from PIL import Image, ImageDraw
import random
import seaborn as sns
from matplotlib.patches import Rectangle
from ultralytics import YOLO
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


@dataclass
class YOLOModel(fls.Model):
    #Input
    n_epochs = 30
    use_pretrained_weights = True
    fix_norm_bug = False
    
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    translate = 0.1
    scale = 0.5
    fliplr = 0.5
    mosaic = 1.0
    mixup = 0.2
    auto_augment = 'randaugment'
    erasing = 0.4
    
    trained_model = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.preprocessor.scale_std = False
        self.preprocessor.scale_percentile = True
        self.preprocessor.return_uint8 = True
            

    
   
    def _train(self,train_data,validation_data):

        # Preprocess data
        # Define YOLO dataset structure and parameters
        data_path = fls.data_dir
        train_dir = os.path.join(data_path, "train")
        
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
        
        # Define constants for processing
        TRUST = 4       # Number of slices above and below center slice (total slices = 2*TRUST + 1)
        BOX_SIZE = 24   # Bounding box size (in pixels)
        
        # Define the preprocessing function to extract slices, normalize, and generate YOLO annotations.
        def prepare_yolo_dataset(trust=TRUST):
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

            train_data_filtered = []
            for d in train_data:
                if len(d.labels)>0:
                    train_data_filtered.append(d)
            validation_data_filtered = []
            for d in validation_data:
                if len(d.labels)>0:
                    validation_data_filtered.append(d)

            # Set train and test
            np.random.shuffle(train_data_filtered)
            
            # Helper function to process a list of tomograms
            def process_tomogram_set(data_list, images_dir, labels_dir, set_name):
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
                    z_min = max(0, z_center - trust)
                    z_max_bound = min(z_max - 1, z_center + trust)
                    self.preprocessor.load_and_preprocess(d, desired_original_slices = slice(z_min,z_max_bound+1))
                    for z in range(z_min, z_max_bound + 1):
                        normalized_img = d.data[z-z_min,:,:]                                   
                        dest_filename = f"{d.name}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                        dest_path = os.path.join(images_dir, dest_filename)
                        Image.fromarray(normalized_img).save(dest_path)
                        
                        # Prepare YOLO bounding box annotation (normalized values)
                        img_width, img_height = (normalized_img.shape[1], normalized_img.shape[0])
                        x_center_norm = x_center / img_width
                        y_center_norm = y_center / img_height
                        box_width_norm = BOX_SIZE / img_width
                        box_height_norm = BOX_SIZE / img_height
                        label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                        with open(label_path, 'w') as f:
                            f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                        
                        processed_slices += 1
                    d.unload()    
                
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
        print(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        
        # Run the preprocessing
        summary = prepare_yolo_dataset(TRUST)
        print(f"\nPreprocessing Complete:")
        print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
        print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
        print(f"- Dataset directory: {summary['dataset_dir']}")
        print(f"- YAML configuration: {summary['yaml_path']}")
        print("\nReady for YOLO training!")
        
        # # Define paths for the Kaggle environment
        yolo_weights_dir = fls.temp_dir + '/yolo_weights/'
        yolo_pretrained_weights = fls.model_dir +"/yolov8m.pt"  # Pre-downloaded weights
        
        # Create a directory to store the weights
        #os.makedirs(yolo_weights_dir, exist_ok=True)
        
        # This will download the YOLOv8 nano weights
        #model = YOLO("yolov8m.pt")
        
        # Save the downloaded weights to a local file
        #model.save(yolo_weights_dir + "yolov8m.pt")

        #model = YOLO(fls.model_dir + 'yolov8m.pt')

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
            plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png'))
            
            print(f"Loss curve saved to {plot_path}")
            plt.close()
            
            return best_epoch, best_val_loss

        def train_yolo_model(yaml_path, pretrained_weights_path, epochs=50, batch_size=16, img_size=640):
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
            if self.use_pretrained_weights:
                print(f"Loading pre-trained weights from: {pretrained_weights_path}")
                model = YOLO(pretrained_weights_path)
            else:
                model = YOLO('yolov8m.yaml')

            from ultralytics import settings

            # Update a setting
            settings.update({"mlflow": False})

            results = model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=yolo_weights_dir,
                name='motor_detector',
                exist_ok=True,
                patience=10,  # Stop training if no improvement after 10 epochs
                save_period=5,  # Save model every 5 epochs
                val=True,
                verbose=True,
                optimizer="AdamW",  # AdamW optimizer for stability
                lr0=0.001,  # Initial learning rate
                lrf=0.01,  # Final learning rate factor
                cos_lr=True,  # Use cosine learning rate decay
                weight_decay=0.0005,  # Prevent overfitting
                momentum=0.937,  # Momentum for better gradient updates
                close_mosaic=10,  # Disable mosaic augmentation after 10 epochs
                workers=4,  # Speed up data loading
                augment=True,  # Enable additional augmentations
                amp=True,  # Mixed precision training for faster performance
                seed=self.seed,
                hsv_h=self.hsv_h, hsv_s=self.hsv_s, hsv_v=self.hsv_v, degrees=0.0, translate=self.translate, scale=self.scale, shear=0.0, perspective=0.0, flipud=0.0, fliplr=self.fliplr, bgr=0.0, mosaic=self.mosaic, mixup=self.mixup, copy_paste=0.0, auto_augment=self.auto_augment, erasing=self.erasing, crop_fraction=1.0,
            )
        
        
            run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
            
            # If function is defined, plot loss curves for better insights
            if 'plot_dfl_loss_curve' in globals():
                best_epoch_info = plot_dfl_loss_curve(run_dir)
                if best_epoch_info:
                    best_epoch, best_val_loss = best_epoch_info
                    print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
        
            return model, results
        
        def prepare_dataset():
            """
            Check if the dataset exists and create/fix a proper YAML file for training.
            
            Returns:
                str: Path to the YAML file to use for training.
            """
            train_images_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
            val_images_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
            train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
            val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'val')
            
            print(f"Directory status:")
            print(f"- Train images exists: {os.path.exists(train_images_dir)}")
            print(f"- Val images exists: {os.path.exists(val_images_dir)}")
            print(f"- Train labels exists: {os.path.exists(train_labels_dir)}")
            print(f"- Val labels exists: {os.path.exists(val_labels_dir)}")
            
            original_yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')
            if os.path.exists(original_yaml_path):
                print(f"Found original dataset.yaml at {original_yaml_path}")
                return fix_yaml_paths(original_yaml_path)
            else:
                print("Original dataset.yaml not found, creating a new one")
                yaml_data = {
                    'path': yolo_dataset_dir,
                    'train': 'images/train',
                    'val': 'images/train' if not os.path.exists(val_images_dir) else 'images/val',
                    'names': {0: 'motor'}
                }
                new_yaml_path = "/kaggle/working/dataset.yaml"
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
        model, results = train_yolo_model(
            yaml_path,
            pretrained_weights_path=yolo_pretrained_weights,
            epochs=self.n_epochs  # For demonstration, using 30 epochs
        )
        
        print("\nTraining complete!")

        self.trained_model = YOLO(fls.temp_dir + 'yolo_weights/motor_detector/weights/best.pt')

    def _infer_single(self,data):   
            
        def preload_image_batch(file_paths):
            """Preload a batch of images to CPU memory."""
            images = []
            for path in file_paths:
                img = cv2.imread(path)
                if img is None:
                    img = np.array(Image.open(path))
                images.append(img)
            return images
        
        def perform_3d_nms(detections, iou_threshold):
            """
            Perform 3D Non-Maximum Suppression on detections to merge nearby motors.
            """
            if not detections:
                return []
            
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            final_detections = []
            def distance_3d(d1, d2):
                return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)
            
            box_size = 24
            distance_threshold = box_size * iou_threshold
            
            while detections:
                best_detection = detections.pop(0)
                final_detections.append(best_detection)
                detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
            
            return final_detections
        
        def process_tomogram(tomo_id, model, index=0, total=1):
            """
            Process a single tomogram and return the most confident motor detection.
            """
            print(f"Processing tomogram {tomo_id} ({index}/{total})")
            tomo_dir = img_dir
            slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

            selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
            selected_indices = np.round(selected_indices).astype(int)
            slice_files = [slice_files[i] for i in selected_indices]
            
            print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices (CONCENTRATION={CONCENTRATION})")
            all_detections = []
            
            if device.startswith('cuda'):
                streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
            else:
                streams = [None]
            
            next_batch_thread = None
            next_batch_images = None
            
            for batch_start in range(0, len(slice_files), BATCH_SIZE):
                if next_batch_thread is not None:
                    next_batch_thread.join()
                    next_batch_images = None
                    
                batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
                batch_files = slice_files[batch_start:batch_end]
                
                next_batch_start = batch_end
                next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
                next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
                if next_batch_files:
                    next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
                    next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
                    next_batch_thread.start()
                else:
                    next_batch_thread = None
                
                sub_batches = np.array_split(batch_files, len(streams))
                for i, sub_batch in enumerate(sub_batches):
                    if len(sub_batch) == 0:
                        continue
                    stream = streams[i % len(streams)]
                    with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                        sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                        sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                        with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                            sub_results = model(sub_batch_paths, verbose=False)
                        for j, result in enumerate(sub_results):
                            if len(result.boxes) > 0:
                                for box_idx, confidence in enumerate(result.boxes.conf):
                                    if confidence >= CONFIDENCE_THRESHOLD:
                                        x1, y1, x2, y2 = result.boxes.xyxy[box_idx].cpu().numpy()
                                        x_center = (x1 + x2) / 2
                                        y_center = (y1 + y2) / 2
                                        all_detections.append({
                                            'z': round(sub_batch_slice_nums[j]),
                                            'y': round(y_center),
                                            'x': round(x_center),
                                            'confidence': float(confidence)
                                        })
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
            
            if next_batch_thread is not None:
                next_batch_thread.join()
            
            final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
            final_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            if not final_detections:
                return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}
            
            best_detection = final_detections[0]
            return {
                'tomo_id': tomo_id,
                'Motor axis 0': round(best_detection['z']),
                'Motor axis 1': round(best_detection['y']),
                'Motor axis 2': round(best_detection['x'])
            }
        
        def debug_image_loading(tomo_id):
            """
            Debug function to test image loading methods.
            """
            tomo_dir = os.path.join(test_dir, tomo_id)
            slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
            if not slice_files:
                print(f"No image files found in {tomo_dir}")
                return
                
            print(f"Found {len(slice_files)} image files in {tomo_dir}")
            sample_file = slice_files[len(slice_files)//2]
            img_path = os.path.join(tomo_dir, sample_file)
            
            try:
                img_pil = Image.open(img_path)
                print(f"PIL Image shape: {np.array(img_pil).shape}, dtype: {np.array(img_pil).dtype}")
                img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                print(f"OpenCV RGB Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
                print("Image loading successful!")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                
            try:
                test_model = YOLO(model_path)
                test_results = test_model([img_path], verbose=False)
                print("YOLO model successfully processed the test image")
            except Exception as e:
                print(f"Error with YOLO processing: {e}")

        data.load_to_memory()
        if self.fix_norm_bug:
            for ii in range(data.data.shape[0]):
                data.data[ii,:,:] = normalize_slice(data.data[ii,:,:])
        img_dir = fls.temp_dir + '/yolo_img_temp' + fls.process_name + '/'
        try: shutil.rmtree(img_dir)
        except: pass
        os.makedirs(img_dir)
        for ii in range(data.data.shape[0]):
            cv2.imwrite(img_dir + f"slice_{ii:04d}.jpg", data.data[ii,:,:])
            
            
        CONFIDENCE_THRESHOLD = 0.45
        MAX_DETECTIONS_PER_TOMO = 3
        NMS_IOU_THRESHOLD = 0.2
        CONCENTRATION = 1  # Process a fraction of slices for fast submission
        
        # GPU profiling context manager for timing
        class GPUProfiler:
            def __init__(self, name):
                self.name = name
                self.start_time = None
                
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.start_time = time.time()
                return self
                
            def __exit__(self, *args):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - self.start_time
                #print(f"[PROFILE] {self.name}: {elapsed:.3f}s")
        
        # Set device and dynamic batch size
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 8
        if device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            #print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
            free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
            BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
            #print(f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
        else:
            print("GPU not available, using CPU")
            BATCH_SIZE = 4
        
        self.trained_model.to(device)
        if device.startswith('cuda'):
            self.trained_model.fuse()
            self.trained_model.model.half()

        results = []
        motors_found = 0

        if data.is_train:
            test_tomos = [fls.data_dir + '/train/' + data.name]
        else:
            test_tomos = [fls.data_dir + '/test/' + data.name]

        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_tomo = {}
            for i, tomo_id in enumerate(test_tomos, 1):
                future = executor.submit(process_tomogram, tomo_id, self.trained_model, i, 1)
                future_to_tomo[future] = tomo_id
            
            for future in future_to_tomo:
                tomo_id = future_to_tomo[future]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = future.result()
                print(result)
                results.append(result)
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                    if not result['Motor axis 0']==-1:
                        d = {'z': [result['Motor axis 0']], 'y': [result['Motor axis 1']], 'x': [result['Motor axis 2']]}
                        data.labels = pd.DataFrame(d)                    
                    else:
                        data.labels = data.labels[0:0]
                else:
                    print(f"No motor detected in {tomo_id}")
                    data.labels = data.labels[0:0]
                #print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")

        return data
                