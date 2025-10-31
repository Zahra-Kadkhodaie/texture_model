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

sys.path.insert(0, '../code')
from model_loader_func import * 
from dataloader_func import *
from quality_metrics_func import *
from linear_approx import *
from algorithm_inv_prob import * 

#########################################################################################################
def get_phi_dataset(model, dataset, sig, n_noise=1, batch_size=512): 
    ## get all the phi's
    my_phis = []
    for c in range(len(dataset)):
        my_phis_c = []
        for k in range(0,len(dataset[c]),batch_size):    
            batch_in = dataset[c][k:k+batch_size]
            sigs = sig*torch.ones(batch_in.shape[0],1,1,1) 
            phi = phi_c_noise_averaged(model, batch_in, sigs, n_noise )
            phi = torch.hstack(phi)
            my_phis_c.append( phi.detach().cpu() )         
            
        my_phis.append(torch.vstack(my_phis_c))
        
    return my_phis

    
#########################################################################################################

def main():
    data_name = 'imagenet'
    ######################################################################################################### 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

     
    ### texture 
    if data_name == 'texture':
        paths = { 
                    'mixture-color-no-skip-deep': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_237580_color_no_skip_deep_enc_80x80/' , 
                    # 'mixture-color-128': 'UNet_flex/texture_EPS_color/0to255_RF_84x84_set_size_106911_color_128x128/' , 
            
                }
        
        data = torch.load(  '/mnt/home/zkadkhodaie/ceph/datasets/texture_EPS/patched_1024x1024_to_80x80_all_sets_color.pt',weights_only=True)
        dim = data.shape[2]
        n_patches = int(1024/dim)**2 # 144
        
        num_classes = int(data.shape[0]/n_patches) # for 80x80: 882 + 786 #for 128x128:786
        
        ##### list of tensors. Each tensor contains images from same class 
        train_sets = []
        test_sets = []
    
        if dim==80: 
            n_test = 4
        if dim ==128: 
            n_test = 1
        
        for d in range(num_classes): 
            train_sets.append(data[n_patches * d: (n_patches * (d+1)) -n_test ])
            test_sets.append(data[(n_patches * (d+1)) -n_test : (n_patches * (d+1)) ] ) 


    
    ### imagenet 
    elif data_name == 'imagenet':
        paths = { 
             'mixture-color-deep-no-skip':'UNet_flex/imagenet/0to255_RF_84x84_set_size_1232457_color_no_skip_deep_dec_64x64/',

        
        train_sets = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/imagenet/test_64x64_color_list.pt', weights_only=True) 
        train_sets_cat = torch.cat(train_sets)
        train_sets = [train_sets_cat[i:i+100] for i in range(0,train_sets_cat.shape[0],100)]
        
    #### multi_class
    elif data_name == 'multi_class': 
        paths = { 
             'mixture-color-deep-no-skip' :  'UNet_flex/multi_class_dataset/0to255_RF_84x84_set_size_270000_color_no_skip_deep_dec_80x80/'               }
    
        temp = []
        categories = ['img_align_celeba', 'bedroom','nabirds','afhq', 'stanford_car', 'flowers102']
        for category in categories:
            data = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/' + category +'/train_color_80x80.pt', weights_only=True)    
            print(category, data.shape)
            if category == 'img_align_celeba' or category == 'bedroom': 
                N = 100_000
            elif  category == 'nabirds':
                N = 30000
            elif category == 'afhq' or category == 'stanford_car' : 
                N = 16000
            elif category == 'flowers102' : 
                N = 8000
                
            temp.append(data[0:N] )
        temp = torch.cat(temp)
        Ntotal = temp.shape[0]
        train_sets = [temp[i:i+100] for i in range(0,Ntotal,100)]
    
    #### face_bedroom
    elif data_name == 'face_bedroom': 
        paths = { 
             'mixture-color-deep-no-skip' :  'UNet_flex/face_bedroom/0to255_RF_84x84_set_size_200000_color_no_skip_deep_dec_80x80/'               }
    
        temp = []
        categories = ['img_align_celeba', 'bedroom']
        for category in categories:
            data = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/' + category +'/train_color_80x80_no_repeats.pt', weights_only=True)    
            print(category, data.shape)
            N = 100_000
            temp.append(data[0:N] )
        temp = torch.cat(temp)
        Ntotal = temp.shape[0]
        train_sets = [temp[i:i+100] for i in range(0,Ntotal,100)]

    #### img_align_celeba
    elif data_name == 'img_align_celeba': 
        paths = { 'mixture-color-deep-no-skip' : 'UNet_flex/img_align_celeba/0to255_RF_84x84_set_size_202399_color_no_skip_deep_dec_80x80/'}
        
        data = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba/train_color_80x80.pt', weights_only=True)    
        temp = data[0:-100] 
        Ntotal = temp.shape[0]
        train_sets = [temp[i:i+100] for i in range(0,Ntotal,100)]

    #### LSUN Bedroom
    elif data_name == 'bedroom': 
        paths = { 'mixture-color-deep-no-skip' : 'UNet_flex/bedroom/0to255_RF_84x84_set_size_299718_color_no_skip_deep_dec_l1_penalty_80x80/'}
        
        data = torch.load( '/mnt/home/zkadkhodaie/ceph/datasets/bedroom/train_color_80x80.pt', weights_only=True)    
        temp = data[0:-100] 
        Ntotal = temp.shape[0]
        train_sets = [temp[i:i+100] for i in range(0,Ntotal,100)]
        
    ##### 
    n_channels = train_sets[0].shape[1]
    print(len(train_sets))
    im_dim = train_sets[0].shape[2]
    
    

    

    ######################################################################################################### 

    root_path = '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/denoisers/'

    denoisers = {}
    
    groups = paths.keys()
    for group in groups: 
        path = root_path + paths[group]
    
        print('loading group ' , group )
        denoisers[group] = load_learned_model(path, print_args=True)
        
    
    ######################################################################################################### 
    # sigmas = [0,10,50, 150, 255, 510, 765]
    sigmas = [0,10, 30,50,70,80, 90, 100,110, 120, 150, 170, 190, 220, 255, 300, 400, 255*2, 650, 255 * 3]
    phis={}
    for group in groups:
        # phis = torch.load( '../results/a_bar_noise_ave_samples_from_'+group+'_'+data_name+str(im_dim)+'x'+str(im_dim) + '.pt'  ,weights_only=True)    
        for sig in sigmas:
            if sig == 0: 
                n_noise = 1
            else: 
                n_noise = 20
            phis[sig] = get_phi_dataset(denoisers[group], train_sets, sig/255 ,n_noise, batch_size=512)
            
            torch.save(phis, '/mnt/home/zkadkhodaie/projects/22_representation_in_UNet_denoiser/results/a_bar_noise_ave_broad_noise_range_'+group+'_'+data_name+str(im_dim)+'x'+str(im_dim) + '.pt' )

if __name__ == "__main__" :
    main()