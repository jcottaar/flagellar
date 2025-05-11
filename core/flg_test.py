import flg_support as fls
import flg_unet
import flg_numerics
import flg_model
#import flg_yolo
import flg_yolo2
import flg_preprocess
import importlib
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

importlib.reload(flg_unet)
#importlib.reload(flg_yolo)
importlib.reload(flg_yolo2)
importlib.reload(flg_preprocess)

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

    if fls.env=='local':
        ref_name = fls.code_dir + 'ref_unet.pickle';
        if update_reference:
            fls.dill_save(ref_name, np.std(heatmap))
        else:
            assert np.std(heatmap) == fls.dill_load(ref_name)

def test_unet_alt(update_reference=False):
    train_data = fls.load_all_train_data()
    model = flg_unet.UNetModel()    
    model.deterministic_train = True
    model.n_images_per_update = 6
    model.dataset.size = (64,64,64)
    model.dataset.offset_range_for_pos = (6,6,6)
    model.n_images_test = 1
    model.n_epochs = 2    
    #model.dataset.normalize = 2
    model2 = model
    model = flg_model.ThreeStepModel()
    model.step1Heatmap = model2
    #model.target_size = 320
    model.preprocessor = flg_preprocess.Preprocessor()
    model.preprocessor.scale_percentile = True
    model.preprocessor.resize = True
    model.preprocessor.resize_target= 320
    model.preprocessor.scale_std = False
    model.seed = 41
    model = model.train_subprocess(train_data[1:3], train_data[4:5])
    d = fls.load_all_test_data()[1]
    heatmap = model.step1Heatmap.infer(d)
    plt.figure()
    plt.imshow(np.max(heatmap, axis=0), cmap='bone')
    plt.colorbar()
    plt.pause(0.01)

    if fls.env=='local':
        ref_name = fls.code_dir + 'ref_unet_alt.pickle';
        if update_reference:
            fls.dill_save(ref_name, np.std(heatmap))
        else:
            assert np.std(heatmap) == fls.dill_load(ref_name)
    
def test_yolo_infer(update_reference=False):
    train_data = fls.load_all_train_data()
    model = fls.dill_load(fls.temp_dir + 'yolo_test.pickle')
    model.step1Labels = flg_model.TestTimeAugmentationStep1(model_internal = model.step1Labels)
    model.step3Output.threshold = 0.
    rr = model.infer(train_data[19:21])
    res = [r.labels for r in rr]

    print(rr[0].labels_unfiltered2)
    print(rr[1].labels_unfiltered2)
    print(res)

    if fls.env=='local':
        ref_name = fls.code_dir + 'ref_yolo.pickle';
        if update_reference:
            fls.dill_save(ref_name, res)
        else:
            assert str(res) == str(fls.dill_load(ref_name))

def test_yolo(update_reference=False):
    train_data = fls.load_all_train_data()[1:100]+fls.load_all_extra_data()[::50]
    model = flg_model.ThreeStepModelLabelBased()
    model.step1Labels.preprocessor = flg_preprocess.Preprocessor2()
    model.step1Labels.fix_norm_bug = True
    model.run_in_parallel = False
    model.calibrate_step_3 = False
    model.seed = 0
    model.step1Labels.n_ensemble = 2
    model.step1Labels.n_epochs = 3
    model.step1Labels.img_size = 320
    model.step1Labels.alternative_slice_selection = True
    model.step1Labels.trust = 0
    model.step1Labels.trust_neg = 2
    model.step1Labels.trust_extra = 1
    model.step1Labels.negative_slice_ratio = 0.1
    model.step1Labels.remove_suspect_areas = True
    model.train_data_selector.datasets = ['tom','mba']
    model.train(train_data, fls.load_all_train_data()[216:230])
    fls.dill_save(fls.temp_dir + 'yolo_test.pickle', model)
    test_yolo_infer(update_reference = update_reference)
    
    

def run_all_tests(update_reference=False):
    test_yolo(update_reference=update_reference)
    test_unet(update_reference=update_reference)
    test_unet_alt(update_reference=update_reference)    
    print('All tests passed')
    