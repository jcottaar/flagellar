import flg_support as fls
import flg_unet
import flg_numerics
import flg_model
import flg_yolo
import importlib
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

importlib.reload(flg_unet)
importlib.reload(flg_yolo)

def test_unet(update_reference=False):
    train_data = fls.load_all_train_data()
    model = flg_unet.UNetModel()
    model.deterministic_train = True
    model.n_images_per_update = 6
    model.dataset.size = (64,64,64)
    model.dataset.offset_range_for_pos = (6,6,6)
    model.n_images_test = 1
    model.n_epochs = 2
    model.seed = 42
    model.train(train_data[1:3], train_data[4:5])
    train_data[4].load_to_memory()
    d = fls.load_all_test_data()[1]
    d.load_to_memory()
    heatmap = model.infer(d)
    plt.figure()
    plt.imshow(np.max(heatmap, axis=0), cmap='bone')
    plt.colorbar()
    plt.pause(0.01)

    ref_name = fls.code_dir + 'ref_unet.pickle';
    if update_reference:
        fls.dill_save(ref_name, np.std(heatmap))
    else:
        assert np.std(heatmap) == fls.dill_load(ref_name)
    

def test_yolo(update_reference=False):
    train_data = fls.load_all_train_data()
    model = flg_yolo.YOLOModel()
    model.seed = 0
    model.n_epochs = 5
    model.train(train_data[1:150], train_data[16:30])
    res = [r.labels for r in model.infer(fls.load_all_test_data()[1:3])]

    print(res)

    ref_name = fls.code_dir + 'ref_yolo.pickle';
    if update_reference:
        fls.dill_save(ref_name, res)
    else:
        assert str(res) == str(fls.dill_load(ref_name))
    
    

def run_all_tests(update_reference=False):
    test_unet(update_reference=update_reference)
    test_yolo(update_reference=update_reference)
    print('All tests passed')
    