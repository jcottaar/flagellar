import flg_support as fls
from matplotlib import animation, rc; rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1000
import flg_numerics
import numpy as np
import copy
import flg_preprocess


def animate_3d_matrix(animation_arr, fps=20, figsize=(6,6), axis_off=True):

    animation_arr= copy.deepcopy(animation_arr[...])
    
    # Initialise plot
    fig = plt.figure(figsize=figsize)  # if size is too big then gif gets truncated

    im = plt.imshow(animation_arr[0], cmap='bone')    
    plt.clim([0, 1])
    if axis_off:
        plt.axis('off')
    #plt.title(f"{tomo_id}", fontweight="bold")

    min_val = np.percentile(animation_arr, 2)
    max_val = np.percentile(animation_arr,98)
    animation_arr = (animation_arr-min_val)/(max_val-min_val)
    # Load next frame
    def animate_func(i):
        im.set_data(animation_arr[i])
        #plt.clim([0, 1])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//fps, blit=True)
        
    return anim

def animate_3d_matrix_no_rescale(animation_arr, fps=20, figsize=(6,6), axis_off=True):

    animation_arr= copy.deepcopy(animation_arr[...])
    
    # Initialise plot
    fig = plt.figure(figsize=figsize)  # if size is too big then gif gets truncated

    im = plt.imshow(animation_arr[0], cmap='bone')    
    plt.clim([0, 255])
    if axis_off:
        plt.axis('off')
    #plt.title(f"{tomo_id}", fontweight="bold")

    #min_val = np.percentile(animation_arr, 2)
    #max_val = np.percentile(animation_arr,98)
    #animation_arr = (animation_arr-min_val)/(max_val-min_val)
    # Load next frame
    def animate_func(i):
        im.set_data(animation_arr[i])
        #plt.clim([0, 1])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//fps, blit=True)
        
    return anim

def animate_labels_full_slice(data_list, z_size):

    slices=[]
    for d in data_list:
        if len(d.labels)>0:
            z = np.round(d.labels['z'][0].astype(int))
            z_min = max(0,z-z_size)
            z_max = min(d.data_shape[0], z+z_size+1)
            desired_slices = list(np.arange(z_min,z_max))
            dat = copy.deepcopy(d)            
            prep = flg_preprocess.Preprocessor()
            prep.scale_percentile = True
            prep.scale_std = True
            prep.scale_std_clip_value = 2.
            prep.resize = True
            prep.resize_target = 320
            prep.return_uint8 = True
            prep.load_and_preprocess(dat, desired_original_slices = desired_slices)
            mat = np.pad(dat.data, ((0,0), (0,max(0,320-dat.data.shape[1])), (0,max(0,320-dat.data.shape[2]))))

            dat = copy.deepcopy(d)          
            prep2 = prep
            #prep2.scale_std = False
            prep2.scale_moving_average = True
            prep2.scale_also_moving_std = False
            #prep2.scale_std_clip_value=2.
            prep2.load_and_preprocess(dat, desired_original_slices = desired_slices)
            mat2 = np.pad(dat.data, ((0,0), (0,max(0,320-dat.data.shape[1])), (0,max(0,320-dat.data.shape[2]))))

            dat = copy.deepcopy(d)          
            prep2 = prep
            #prep2.scale_std = False
            prep2.scale_moving_average = True
            prep2.scale_also_moving_std = True
            #prep2.scale_std_clip_value=2.
            prep2.load_and_preprocess(dat, desired_original_slices = desired_slices)
            mat3 = np.pad(dat.data, ((0,0), (0,max(0,320-dat.data.shape[1])), (0,max(0,320-dat.data.shape[2]))))

            mat4 = np.concatenate((mat, mat2, mat3),axis=2)
            slices.append(np.mean(mat4,axis=0))
    mat_combined = np.stack(slices)
    
    return animate_3d_matrix_no_rescale(mat_combined,figsize=(12,4), fps=5)
    
def animate_labels(data_list, sizes, tile_num=5, normalize_slices=False, animate=True):
    mat = flg_numerics.collect_patches(data_list, np.array(sizes),normalize_slices=normalize_slices)[0]
    mat = np.nansum(mat,axis=1)[:,np.newaxis,:,:]    
    #print(mat.shape)
    #mat = np.reshape(mat, (-1, mat.shape[2], mat.shape[3]))
    #return animate_3d_matrix(mat,figsize=(4,4))

    mat_tiled_list = []
    cur_tile_x = 0
    cur_tile_y = 0
    for ii in range(mat.shape[0]):
        #print(ii, len(mat_tiled_list))
        m=mat[ii,:,:,:].astype(np.float32)
        m = (m-np.nanmean(m))/np.nanstd(m)
        if cur_tile_x==0 and cur_tile_y==0:
            cur_mat = np.zeros(shape=(m.shape[0], tile_num*m.shape[1], tile_num*m.shape[2]))
        cur_mat[:, cur_tile_x*m.shape[1]:(cur_tile_x+1)*m.shape[1], cur_tile_y*m.shape[1]:(cur_tile_y+1)*m.shape[1]] = m
        cur_tile_x+=1
        if cur_tile_x==tile_num:
            cur_tile_x=0
            cur_tile_y+=1
            if cur_tile_y==tile_num:
                mat_tiled_list.append(cur_mat)
                cur_tile_y=0
    mat_tiled_list.append(cur_mat)
    #print(len(mat_tiled_list))
    #print(mat_tiled_list[0].shape)
    mat_combined = np.concatenate(mat_tiled_list)
    #print(mat_combined.shape)

    if animate:
        return animate_3d_matrix(mat_combined,figsize=(8,8), fps=5)
    else:
        plt.figure(figsize=(6,6))
        plt.imshow(mat_combined[0,:,:], cmap='bone')
        min_val = np.percentile(mat_combined, 2)
        max_val = np.percentile(mat_combined,98)
        plt.clim([min_val, max_val])


def show_tf_pn(inferred_data, reference_data):
    inferred_data_x = copy.deepcopy(inferred_data)
    reference_data_x = copy.deepcopy(reference_data)

    inds = np.random.default_rng(seed=0).permutation(len(inferred_data_x))

    inferred_data = []; reference_data = [];
    for i in inds:
        inferred_data.append(inferred_data_x[i]); reference_data.append(reference_data_x[i]);

    del inferred_data_x, reference_data_x

    # True positives
    for d in inferred_data:
        d.labels_unfiltered2 = copy.deepcopy(d.labels_unfiltered)
        d.labels_unfiltered = copy.deepcopy(d.labels)
        assert(len(d.labels)<=1)
        assert(len(d.labels_unfiltered)<=1)
    fls.mark_tf_pn(inferred_data, reference_data)
    for d in inferred_data:
        assert(len(d.labels)<=1)
        assert(len(d.labels_unfiltered)<=1)
    to_plot = []
    all_pos=0  
    done = []
    for i,(d,r) in enumerate(zip(inferred_data,reference_data)):
        if len(r.labels)==1:
            all_pos+=1
        if len(d.labels_unfiltered)>0:
            assert len(d.labels_unfiltered)==1
            if d.labels_unfiltered.reset_index().at[0,'tf_pn'] == 0:
                to_plot.append(r)
                done.append(i)
    print(f'True positives: {len(to_plot)} out of {all_pos}')
    if len(to_plot)>64:
        to_plot = to_plot[:64]
    animate_labels(to_plot, (5,150,150), tile_num=8, normalize_slices=True, animate=False)
    plt.title('True positives')

    # False negatives - seen but not selected
    for d in inferred_data:
        d.labels_unfiltered = copy.deepcopy(d.labels_unfiltered2)
    fls.mark_tf_pn(inferred_data, reference_data)
    to_plot = []
    for i,(d,r) in enumerate(zip(inferred_data,reference_data)):
        if len(d.labels_unfiltered)>0 and np.any(d.labels_unfiltered['tf_pn']==0) and (not i in done):
            assert(len(r.labels)==1)
            to_plot.append(r)
            done.append(i)
    print(f'False negatives - seen but not selected: {len(to_plot)} out of {all_pos}')
    if len(to_plot)>64:
        to_plot = to_plot[:64]
    animate_labels(to_plot, (5,150,150), tile_num=8, normalize_slices=True, animate=False)
    plt.title('False negatives - seen but not selected')


    # False negatives - not seen
    to_plot = []
    for i,(d,r) in enumerate(zip(inferred_data,reference_data)):
        if len(r.labels)==1 and (not i in done):
            assert len(d.labels_unfiltered)==0 or np.all(d.labels_unfiltered['tf_pn']==1)
            to_plot.append(r)
            done.append(i)
    print(f'False negatives - not seen: {len(to_plot)} out of {all_pos}')
    if len(to_plot)>64:
        to_plot = to_plot[:64]
    animate_labels(to_plot, (5,150,150), tile_num=8, normalize_slices=True, animate=False)
    plt.title('False negatives - not seen')

    # False positives
    # for d in inferred_data:
    #     d.labels_unfiltered2 = copy.deepcopy(d.labels_unfiltered)
    #     d.labels_unfiltered = copy.deepcopy(d.labels)
    #     assert(len(d.labels)<=1)
    #     assert(len(d.labels_unfiltered)<=1)
    # fls.mark_tf_pn(inferred_data, reference_data)
    to_plot = []
    all_neg = 0
    for i,(d,r) in enumerate(zip(inferred_data,reference_data)):  
        if len(r.labels)==0:
            all_neg+=1
        if len(d.labels)>0 and (not i in done):
            assert len(r.labels)==0
            to_plot.append(d)
            done.append(i)
    print(f'False positives: {len(to_plot)} out of {all_neg}')
    if len(to_plot)>64:
        to_plot = to_plot[:64]
    animate_labels(to_plot, (5,150,150), tile_num=8, normalize_slices=True, animate=False)
    plt.title('False positives')

    
    
    
            
                                       