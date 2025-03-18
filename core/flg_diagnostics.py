import flg_support as fls
from matplotlib import animation, rc; rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1000
import flg_numerics
import numpy as np


def animate_3d_matrix(animation_arr, fps=20, figsize=(6,6)):
    
    # Initialise plot
    fig = plt.figure(figsize=figsize)  # if size is too big then gif gets truncated

    im = plt.imshow(animation_arr[0], cmap='bone')
    plt.axis('off')
    #plt.title(f"{tomo_id}", fontweight="bold")
    
    # Load next frame
    def animate_func(i):
        im.set_data(animation_arr[i])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//fps, blit=True)
        
    return anim

def animate_labels(data_list, sizes, tile_num=5):
    mat = flg_numerics.collect_patches(data_list, np.array(sizes))[0]
    #print(mat.shape)
    #mat = np.reshape(mat, (-1, mat.shape[2], mat.shape[3]))
    #return animate_3d_matrix(mat,figsize=(4,4))

    mat_tiled_list = []
    cur_tile_x = 0
    cur_tile_y = 0
    for ii in range(mat.shape[0]):
        m=mat[ii,:,:,:]
        if cur_tile_x==0 and cur_tile_y==0:
            cur_mat = np.zeros(shape=(2*sizes[0]+1, tile_num*m.shape[1], tile_num*m.shape[2]))
        cur_mat[:, cur_tile_x*m.shape[1]:(cur_tile_x+1)*m.shape[1], cur_tile_y*m.shape[1]:(cur_tile_y+1)*m.shape[1]] = m
        cur_tile_x+=1
        if cur_tile_x==tile_num:
            cur_tile_x=0
            cur_tile_y+=1
            if cur_tile_y==tile_num:
                mat_tiled_list.append(cur_mat)
                cur_tile_y=0
    mat_tiled_list.append(cur_mat)
    mat_combined = np.concatenate(mat_tiled_list)
    print(mat_combined.shape)

    return animate_3d_matrix(mat_combined,figsize=(8,8), fps=5)

    
            
                                       