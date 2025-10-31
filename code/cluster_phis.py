import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import torchvision
import random
from torch.optim import Adam
import os
import time
import torch
import sys
from sklearn.cluster import KMeans

sys.path.insert(0, '../code')
from model_loader_func import * 
from dataloader_func import *
from quality_metrics_func import *
from linear_approx import *
from algorithm_inv_prob import * 

#########################################################################################################



def main():
    data_name = 'texture'
    ######################################################################################################### 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

     
    ### texture 
    if data_name == 'texture':
        paths = { 
                    'mixture-color-no-skip-deep': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_237580_color_no_skip_deep_enc_80x80/' , 
                    # 'mixture-color-128': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_106911_color_128x128/' , 
            
                }
        phis = torch.load('../results/phis_noise_ave_80x80_mixture-color-no-skip-deep_texture.pt', weights_only= True)
        

    ### imagenet 
    elif data_name == 'imagenet':
        paths = { 
                     'mixture-color-no-skip-deep': 'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/',
                    # 'mixture-color-large-no-skip':'UNet_flex/imagenet/0to300_RF_232x232_set_size_1232457_color_no_skips_64x64/', 
                    # 'mixture-color-no-skip':'UNet_flex/imagenet/0to255_RF_128x128_set_size_1232457_color_no_skip_deep_dec_128x128/',
                    # 'mixture-color-no-skip_smaller':'UNet_flex/imagenet/0to255_RF_128x128_set_size_1232457_color_no_skip_128x128/'
                }
        phis = torch.load('../results/phis_noise_ave_64x64_mixture-color-no-skip-deep_imagenet.pt', weights_only= True)

    #### img_align_celeba
    elif data_name == 'img_align_celeba': 
        paths = { 'mixture-color-deep-no-skip' : 'UNet_flex/img_align_celeba/0to255_RF_84x84_set_size_202399_color_no_skip_deep_dec_80x80/'}
        a_bars = torch.load('../results/a_bar_noise_ave_mixture-color-deep-no-skip_img_align_celeba80x80.pt', weights_only = True)
        phis = {}
        for sig in a_bars.keys():
            mid = 3
            phis[sig] = [a_bars[sig][i][:,  64+128+256: 64+128+256+512 ] for i in range(len(a_bars[sig]))]


    #### LSUN Bedroom
    elif data_name == 'bedroom': 
        paths = { 'mixture-color-deep-no-skip' : 'UNet_flex/bedroom/0to255_RF_84x84_set_size_299718_color_no_skip_deep_dec_80x80/'}
        a_bars = torch.load('../results/a_bar_noise_ave_mixture-color-deep-no-skip_bedroom80x80.pt', weights_only = True)
        phis = {}
        for sig in a_bars.keys():
            mid = 3
            phis[sig] = [a_bars[sig][i][:,  64+128+256: 64+128+256+512 ] for i in range(len(a_bars[sig]))]
    

    ######################################################################################################### 

    root_path = '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/denoisers/'

    denoisers = {}
    
    groups = paths.keys()
    for group in groups: 
        path = root_path + paths[group]
    
        print('loading group ' , group )
        denoisers[group] = load_learned_model(path, print_args=True)
        
    
    ######################################################################################################### 
    sigmas = phis.keys()
    
    # use the centroids of labeled data to initialize 
    init_centoids = {}
    # for sig in sigmas:
        # init_centoids[sig] = torch.vstack([phis[sig][c].mean(dim = 0) for c in range(len(phis[sig])) ]).cpu()                

    
    kmeans_results = {}
    for group in groups:
        for sig in sigmas:
            kmeans_results[sig] ={}
            for n_clusters in [ 100,1000,10000 ]:                 
                # if n_clusters == len(phis[sig]): 
                if n_clusters == 10000000000:                     
                    n_init = 1
                    init = init_centoids[sig]
                else: 
                    n_init = 10
                    init = 'k-means++'
                kmeans = KMeans(n_clusters=n_clusters, random_state = 42, n_init=n_init, init = init )
                kmeans.fit(torch.concat(phis[sig]).cpu() )

                kmeans_results[sig][n_clusters] = kmeans
        # Save with pickle - different file per group 
        name = '/mnt/home/zkadkhodaie/projects/22_representation_in_UNet_denoiser/results/kmeans_noise_ave_'+group+'_'+data_name+ '_cluster_sizes.pkl'
        with open(name, 'wb') as f:
            pickle.dump(kmeans_results, f)

if __name__ == "__main__" :
    main()