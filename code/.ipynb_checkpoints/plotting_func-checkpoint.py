import numpy as np
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from quality_metrics_func import batch_psnr_numpy,normalized_distance_np
from dataloader_func import rescale_image_range,rescale_image

######################################################################################
############################################### plots train time #####################
######################################################################################

def plot_loss(loss_list, loss_list_test, file_name):

    fig , axs = plt.subplots(1,2 , figsize= (12,5), sharey = True)
    axs[0].plot( range(len(loss_list)), loss_list, 'b-o', label = 'train')
    axs[0].set_ylabel('MSE')
    axs[0].set_title('min train loss ' + str(round(min(loss_list),3)) + ' from epoch ' + str(loss_list.index(min(loss_list))) + '\n final loss '+str(round(loss_list[-1],3)))
    axs[0].legend()
    axs[1].plot( range(len(loss_list_test)), loss_list_test, 'r-o', label = 'test loss')
    axs[1].set_title('min test loss ' + str(round(min(loss_list_test),3)) + ' from epoch ' + str(loss_list_test.index(min(loss_list_test))) + '\n final test loss '+str(round(loss_list_test[-1],3)))
    axs[1].legend()   
    plt.savefig(file_name)

def plot_psnr(psnr_list,psnr_list_test, file_name):

    fig , axs = plt.subplots(1,2 , figsize= (12,5), sharey = True)
    axs[0].plot( range(len(psnr_list)), psnr_list, 'orange','-o', label = 'train')
    axs[0].set_ylabel('PSNR')
    axs[0].set_title('max train psnr ' + str(round(max(psnr_list),3)) + ' from epoch ' + str(psnr_list.index(max(psnr_list))) + '\n final psnr '+str(round(psnr_list[-1],3)))
    axs[0].legend()
    axs[1].plot( range(len(psnr_list_test)), psnr_list_test, 'green','-o', label = 'test')
    axs[1].set_title('max test psnr ' + str(round(max(psnr_list_test),3)) + ' from epoch ' + str(psnr_list_test.index(max(psnr_list_test))) + '\n final psnr '+str(round(psnr_list_test[-1],3)))
    axs[1].legend()    
    plt.savefig(file_name)


def plot_denoised_image( clean, noisy, denoised ,dir_name):
    clean = clean.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    noisy = noisy.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    denoised = denoised.detach().cpu().permute(0,2,3,1).squeeze().numpy()

    f, axes = plt.subplots(3, 3 )
    ax = axes.ravel()

    for i ,j in zip(range(0, 9, 3), range(3) ):
        fig = ax[i].imshow(clean[j], 'gray')
        ax[i].set_axis_off()
        ax[i].set_title('clean')
        plt.colorbar(fig, ax=ax[i], fraction=.05)

        fig = ax[i+1].imshow(noisy[j], 'gray')
        ax[i+1].set_title( 'PSNR '+ str(round(peak_signal_noise_ratio(clean[j],noisy[j], data_range=1),3) ) + '\n SSIM ' + str(round(structural_similarity(clean[j], noisy[j],multichannel=True),3)), fontsize = 5)
        ax[i+1].set_axis_off()
        plt.colorbar(fig, ax=ax[i+1], fraction=.05)

        fig = ax[i+2].imshow(denoised[j], 'gray')
        ax[i+2].set_title(  'PSNR '+ str(round(peak_signal_noise_ratio(clean[j],denoised[j], data_range=1),3)) + '\n SSIM ' + str(round(structural_similarity(clean[j], denoised[j] ,multichannel=True),3)),fontsize = 5)
        ax[i+2].set_axis_off()
        plt.colorbar(fig, ax=ax[i+2], fraction=.05)

    file_name = dir_name + '/denoised_test_image.png'
    plt.savefig(file_name)
    plt.close('all')

def plot_denoised_range(clean, noisy, denoised, noise_range, file_name,data_max, writer=None, h=None):
    clean = torch.stack([clean]*noise_range.shape[0])
    clean = clean.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    noisy = noisy.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    denoised = denoised.detach().cpu().permute(0,2,3,1).squeeze().numpy()

    psnr_noisy = batch_psnr_numpy(clean, noisy ,data_max )
    psnr_denoised = batch_psnr_numpy(clean, denoised ,data_max) 
    f, ax = plt.subplots(noise_range.shape[0], 3 , figsize = (3*3, noise_range.shape[0]*3))
    f.tight_layout()
    for i  in  range(noise_range.shape[0]):
        fig = ax[i,0].imshow(np.clip(clean[i] , 0,1), 'gray')
        ax[i,0].set_axis_off()
        ax[i,0].set_title('clean')
        plt.colorbar(fig, ax=ax[i,0], fraction=.05)

        fig = ax[i,1].imshow(np.clip(noisy[i],0,1), 'gray')
        ax[i,1].set_title( 'PSNR '+ str(round(psnr_noisy[i],3))   , fontsize = 15)
        ax[i,1].set_axis_off()
        plt.colorbar(fig, ax=ax[i,1], fraction=.05)

        fig = ax[i,2].imshow(np.clip(denoised[i],0,1), 'gray')
        ax[i,2].set_title( 'PSNR '+ str(round(psnr_denoised[i],3))   , fontsize = 15)
        ax[i,2].set_axis_off()
        plt.colorbar(fig, ax=ax[i,2], fraction=.05)
   
    plt.savefig(file_name)
    
    if writer is not None:
        writer.add_figure(file_name.split('/')[-1], f, global_step=h) 

    plt.close('all')


    
######################################################################################


def plot_denoising(clean, noisy, denoised, sup_label, device, vmin=0, vmax=1, im_size=3): 
    f, axs = plt.subplots(1,3, figsize = (3 * im_size, im_size+1) )
    f.suptitle(sup_label, fontsize = 20)
    if device.type == 'cuda':
        clean = clean.cpu().squeeze().numpy()
        noisy = noisy.cpu().squeeze().numpy()
        denoised = denoised.cpu().squeeze().numpy()
        
    axs[0].imshow(clean, 'gray', vmin=vmin, vmax = vmax) 
    axs[0].set_title('clean')
    axs[1].imshow(noisy, 'gray',vmin=vmin, vmax = vmax)
    axs[1].set_title(  'PSNR '+ str(round(peak_signal_noise_ratio(clean,noisy, data_range=1),3)) + '\n SSIM ' + str(round(structural_similarity(clean, noisy ,channel_axis=True),3)),fontsize = 12)    
    axs[2].imshow(denoised, 'gray',vmin=vmin, vmax = vmax)
    axs[2].set_title(  'PSNR '+ str(round(peak_signal_noise_ratio(clean,denoised, data_range=1),3)) + '\n SSIM ' + str(round(structural_similarity(clean, denoised ,channel_axis=True),3)),fontsize = 12)    
    
    for i in range(3): 
        axs[i].axis('off')



def plot_many_denoised(x, y, x_hat,  device, suptitle, label, train, vmin=0, vmax=1, im_size=3, n_columns=7):
    '''plot many denoised images from the same clean image (either different denoisers or different noise levels )
    @x: a single clean image
    @y: dictionary of noisy images
    @x_hat: dictionary of denoised images
    @label: to be used for subplot title
    @train: if True, clean comes from train set. If False, it comes from test set
    '''
    ############ plot clean image


    x = x.cpu().permute(0,2,3,1).squeeze()

    f, axs = plt.subplots(1 ,1, figsize = ( im_size+.5, im_size+.5) )
    f.suptitle(suptitle)
    plt.tight_layout()


    axs.imshow(x, 'gray', vmin=vmin, vmax = vmax)
    if train:
        axs.set_title('clean ' +  'Train image' ,fontsize = im_size*5)
    else:
        axs.set_title('clean ' +  'Test image' ,fontsize = im_size*5)
    axs.axis('off')

    ############ plot noisy images
    n_rows = int(len(y)/n_columns)
    if len(y)%n_columns != 0:
        n_rows = n_rows+1

    im_labels = [key for key,im in y.items() ]

    for i in range(len(y)):
        y[im_labels[i]] = y[im_labels[i]].permute(0,2,3,1).cpu().squeeze()

    f, axs = plt.subplots(n_rows ,n_columns, figsize = ( im_size*n_columns, n_rows*im_size ) )
    axs = axs.ravel()
    plt.tight_layout()

    for i in range(len(y)):
        axs[i].imshow(y[im_labels[i]], 'gray',vmin=vmin, vmax = vmax)
        axs[i].set_title( 'PSNR '+ str(round(peak_signal_noise_ratio(x.numpy(),y[im_labels[i]].numpy(), data_range=1),3))
                         + '\n distance from clean \n or (noise sigma ): ' + str( round(normalized_distance_np(x.numpy(), y[im_labels[i]].numpy()) ,3)),
                         fontsize =im_size*5)

    for i in range(len(axs)):
        axs[i].axis('off')

    ############ plot denoised images
    im_labels = [key for key,im in x_hat.items() ]

    n_rows = int(len(x_hat)/n_columns)
    if len(x_hat)%n_columns != 0:
        n_rows = n_rows+1

    for i in range(len(x_hat)):
        x_hat[im_labels[i]] = x_hat[im_labels[i]].permute(0,2,3,1).cpu().squeeze()

    f, axs = plt.subplots(n_rows ,n_columns, figsize = ( im_size*n_columns, n_rows*im_size ) )
    axs = axs.ravel()
    plt.tight_layout()

    for i in range(len(x_hat)):
        axs[i].imshow(x_hat[im_labels[i]], 'gray',vmin=vmin, vmax = vmax)
        axs[i].set_title( label + str(im_labels[i])+
                               '\n PSNR '+ str(round(peak_signal_noise_ratio(x.numpy(),x_hat[im_labels[i]].numpy(), data_range=1),3)) +
                      '\n distance from clean: ' + str( round(normalized_distance_np(x.numpy(), x_hat[im_labels[i]].numpy()) ,3)),
                                                      fontsize = im_size*5)

    for i in range(len(axs)):
        axs[i].axis('off')


def show_im_set(dataset , N=None , im_size=3,vmin=None, vmax=None, label=None, colorbar=False,n_columns=10, sub_labels=None, colormap='gray',
               norm=None, font_size=15):
    if N is None:
        N = dataset.shape[0]
    device = dataset.device
    if device.type == 'cuda':
        dataset = dataset.cpu()

    if dataset.shape[1] != 1:
        dataset = dataset.permute(0,2,3,1)


    n_rows = int(N/n_columns)
    if N%n_columns != 0:
        n_rows = n_rows+1


    f, axs = plt.subplots(n_rows, n_columns, figsize = (im_size * n_columns , im_size * n_rows))
    axs = axs.ravel()
    if label is not None:
        f.suptitle( label , fontsize = font_size*1.5 )
    plt.tight_layout()
    for i in range(N ):
        if dataset.shape[1] == 1:
            fig = axs[i].imshow(dataset[i,0], colormap,norm=norm, vmin=vmin, vmax = vmax)
        else:
            fig = axs[i].imshow(rescale_image_range(dataset[i],1), colormap,norm=norm)

        if colorbar:
            plt.colorbar(fig, ax=axs[i], fraction=.05)
        if sub_labels is not None:
            axs[i].set_title( str(sub_labels[i]), fontsize = font_size)

        else:
            axs[i].set_title(str(i))
    for i in range(len(axs)):
        axs[i].axis('off')
    plt.show()

    dataset = dataset.to(device)
    
def show_im_grid(dataset , N=None , im_size=2,vmin=None, vmax=None, label=None,
                 n_columns=10, colormap='gray',
               norm=None, font_size=15, save_name=None, dpi= 100, grid_lines = False, return_image=False):
    if N is None:
        N = dataset.shape[0]
    device = dataset.device
    if device.type == 'cuda':
        dataset = dataset.cpu()

    H = dataset.shape[2]
    W = dataset.shape[3]
    
    n_rows = N//n_columns
    
    if N%n_columns != 0:
        n_rows = n_rows+1
        
    extras = N%n_rows

    ones = torch.ones((n_columns - extras, dataset.shape[1], dataset.shape[2], dataset.shape[3] ))
    dataset= torch.cat([dataset, ones])
    
    temp1 = []
    temp2 = []
    for j in range(n_rows):
        
        temp1 = torch.cat([dataset[j*n_columns+i] for i in range(n_columns)], dim=2)
        temp2.append(temp1)
    image = torch.cat(temp2, dim = 1).permute(1,2,0).squeeze()
    f, axs = plt.subplots(1, 1, figsize = (im_size*n_columns  , im_size*n_rows ))
    if label is not None:
        f.suptitle( label , fontsize = font_size*1.5 )

    fig = axs.imshow(image, colormap,norm=norm, vmin=vmin, vmax = vmax)

    if grid_lines:
        axs.set_xticks( np.array(list(range(0, image.shape[1], H))) -.5 )
        axs.set_yticks( np.array(list(range(0, image.shape[0], W))) -.5 )    
        axs.grid(color='r', linestyle='-', linewidth=2)
        axs.set_xticklabels([])
        axs.set_yticklabels([])        
    else: 
        axs.axis('off')
        
    # plt.show()
    if save_name is not None: 
        plt.savefig(save_name,bbox_inches='tight', dpi=dpi)    
    plt.show()
    dataset = dataset.to(device)
    if return_image:
        return image


def plot_single_im(x, size=(2,2), vmin=None, vmax=None, colorbar = False, label = None):

    if x.device.type == 'cuda':
        x = x.cpu().squeeze()
    else:
        x = x.squeeze()

    if len(x.shape)==3:
        x = x.permute(1,2,0)
        
    plt.figure(figsize = size)
    if len(x.shape)==3:
        plt.imshow(rescale_image_range(x,1) )
    else: 
        plt.imshow(x, 'gray' , vmin=vmin, vmax=vmax )
        
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    if label is not None:
        plt.title(label)
       
