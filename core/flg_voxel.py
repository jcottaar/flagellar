import flg_support as fls
import importlib
import numpy as np
import flg_diagnostics
import flg_numerics
import matplotlib.pyplot as plt
import glob
import copy
import flg_preprocess
import os
import shutil
import pandas as pd
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

def estimate_voxel_spacing(data):
    cpu, device = fls.prep_pytorch(0, True, False)

    model = fls.dill_load(fls.model_dir + 'voxel_spacing_model_n25.pickle')
    model.to(device)
    model.eval()

    for d in data:
        # Collect files
        fls.remove_and_make_dir(fls.temp_dir + '/voxel/')
        files = glob.glob(d.data_dir() + '/*.jpg')
        files.sort()
        spacing = len(files)//20+1
        for f in files[::spacing][2:-2]:
            shutil.copyfile(f,fls.temp_dir + '/voxel/'+os.path.basename(f))
    

        # Prep dataset
        tfm = transforms.Compose([
            #transforms.Resize(IMG_SIZE),         # smaller side → IMG_SIZE, uniform scale
            #transforms.CenterCrop(IMG_SIZE),    # cut to IMG_SIZE×IMG_SIZE
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])    
        test_ds     = VoxelDataset(fls.temp_dir + '/voxel/',  transform=tfm, train=False)
        test_loader = DataLoader(test_ds,  batch_size=256,
                                  shuffle=False, num_workers=0, pin_memory=True)

        with torch.no_grad():
            for imgs, _, _ in test_loader:
                imgs     = imgs.to(device)                
                preds    = model(imgs)                

        d.voxel_spacing = np.median(preds.detach().cpu().numpy()).item() / scale
    
    

IMG_SIZE   = 224*2
r = np.random.default_rng(seed=0)
scale = 0
class VoxelDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, transform=None, IMG_SIZE=IMG_SIZE, train=True):
        self.img_dir   = img_dir
        self.transform = transform
        self.IMG_SIZE  = IMG_SIZE
        if csv_file:
            df = pd.read_csv(csv_file)
            self.samples = [(row['filename'], float(row['voxel_spacing'])) 
                            for _, row in df.iterrows()]
        else:
            self.samples = [(f, None) for f in sorted(os.listdir(img_dir))
                            if f.lower().endswith('.jpg')]

        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, orig_spacing = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img  = Image.open(path).convert('L')
        orig_w, orig_h = img.size

        # 1) Pad to square on the shorter side
        max_dim = max(orig_w, orig_h)
        pad_w   = max_dim - orig_w
        pad_h   = max_dim - orig_h
        # (left, top, right, bottom)
        padding = (pad_w//2, pad_h//2,
                   pad_w - pad_w//2, pad_h - pad_h//2)
        img = TF.pad(img, padding, fill=0)

        assert max_dim == img.size[0]
        assert max_dim == img.size[0]
        if r.uniform()>0.5 and self.train:
            # pad extra
            rr = r.uniform()
            img = TF.pad(img, (0,0,np.round(rr*max_dim//2).astype(int),np.round(rr*max_dim//2).astype(int)), fill=0)
            max_dim = img.size[0]
        assert max_dim == img.size[0]
        assert img.size[0] == img.size[1]


        # 2) Compute the scale factor: new_pixels = old_pixels * (IMG_SIZE/max_dim)
        #    so each new_pixel represents orig_spacing * (max_dim/IMG_SIZE)
        global scale
        scale       = max_dim / self.IMG_SIZE / 20
        if orig_spacing is not None:            
            new_spacing = orig_spacing * scale
            #new_spacing = 1
        else:
            new_spacing = None

        # 3) Resize the padded square down to IMG_SIZE×IMG_SIZE
        img = TF.resize(img, [self.IMG_SIZE, self.IMG_SIZE])
        # 4) Run your usual ToTensor+Normalize
        if self.transform:
            img = self.transform(img)

        if r.uniform()>0.5 and self.train:
            img = torch.flip(img,dims=[1])

        if r.uniform()>0.5 and self.train:
            img = torch.flip(img,dims=[2])

        if r.uniform()>0.5 and self.train:
            img = torch.rot90(img, dims=[1,2])


        if new_spacing is None:
            return img, fname, torch.tensor([0], dtype=torch.float32)
        else:
            return img, fname, torch.tensor([new_spacing], dtype=torch.float32)