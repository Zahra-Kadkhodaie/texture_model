import numpy as np
import torch
from dataloader_func import rescale_image_range
from noise import add_noise_torch
######################################################################################
################################ quality metrics #####################################
######################################################################################

def MSE(ref_im, im ):
    return ((ref_im - im)**2).mean()


def sig_to_psnr(std, I_max = 255): 
    '''std for im in range 0 to 255'''
    return -10*np.log10( (std/255)**2  )   

def batch_psnr_numpy(ref_ims, images ,max_I):
    '''
    batch of ref_im and im are tensors of dims N, W, H
    '''
    if len(ref_ims.shape)==3:
        mse = ((ref_ims - images)**2).mean(axis=(1,2))
    elif len(ref_ims.shape)==4:
        mse = ((ref_ims - images)**2).mean(axis=(1,2,3))
    psnr_all =  10*(np.log10(max_I**2) - np.log10(mse))
    return psnr_all


def batch_ave_psnr_torch(ref_ims, images ,max_I):
    '''
    batch of ref_im and im are tensors of dims N,C, W, H
    returns ave psnr of a batch of images 
    '''
    mse = ((ref_ims - images)**2).mean(dim=(1,2,3))
    psnr_all =  10*(np.log10(max_I**2) - torch.log10(mse))
    return psnr_all.mean()


def psnr(denoiser,loader, sigma,device,skip=True): 
    '''
    Takes denoiser, clean data, noise level, 
    and returns ave psnr of denoised images for that specific sigma
    '''
    psnr_sum = 0
    for i, batch in enumerate(loader,0):
        denoiser.eval()
        clean = batch.to(device)
        noisy , noise, _ = add_noise_torch(all_patches=clean, noise_level=sigma)
        with torch.no_grad():
            denoised = denoiser(noisy)
        if skip: 
            denoised = noisy - denoised
            
        psnr_sum += batch_ave_psnr_torch(clean, denoised ,1.).item()
    return psnr_sum/(i+1)
 



def calc_psnr(denoiser,loader, sigma_range, device, skip=True, max_I=1., rescale=False, loader_cond=None):
    '''
    Takes denoiser, clean data, and a range of noise, 
    returns ave psnr for all sigma in the noise range
    '''    
    if loader_cond is not None: 
        loader_cond_list = list(enumerate(loader_cond))
        
    psnr_range = {}
    for sigma in sigma_range:
        psnr_range[str(round(sigma.item(),5))] = []
        psnr = 0
        for i, batch in enumerate(loader,0):
            denoiser.eval()
            clean = batch.to(device)
            _,C,_,_ = clean.shape
            if rescale:
                clean = rescale_image_range(clean, 1.,0.)
            noisy , noise , _= add_noise_torch(all_patches=clean, noise_level=sigma)
            with torch.no_grad():
                if loader_cond is None: 
                    output = denoiser(noisy)
                else: 
                    output = denoiser(noisy, loader_cond_list[i][1].to(device))
                    
                if skip: 
                    denoised = noisy - output
                else: 
                    denoised = output
                    
            psnr += batch_ave_psnr_torch(clean, denoised ,max_I).item()
            
        psnr_range[str(round(sigma.item(),5))] = psnr/(i+1)
    return psnr_range









def normalize_im_set_l1(im_set):
    '''
    im_set: tensor of size N,C,H, W
    this function normalizes intensity variations for images
    '''
    if len(im_set.shape) != 4:
        raise ValueError('Input shape error')

    return im_set/im_set.norm(dim = (2,3),p=1, keepdim=True)

def normalized_distance_np(x, y):
    '''
    Euclidean distance / sqrt(im size). Normalization is applied to make it comparible to sigma.
    Since radius of sphere in concentration of measure theorem is sigma/sqrt(im size)
    '''

    return np.linalg.norm(x - y) /(np.sqrt(np.prod(x.shape)))

def normalized_distance_torch(x, y):
    '''
    Euclidean distance / sqrt(im size). Normalization is applied to make it comparible to sigma.
    Since radius of sphere in concentration of measure theorem is sigma/sqrt(im size)
    '''
    if x.device.type == 'cuda':
        x = x.cpu()
    if y.device.type == 'cuda':
        y = y.cpu()
    return torch.norm(x - y) /(np.sqrt(np.prod(x.shape)))

def cos_similarity(im1,im2):
    return torch.matmul(((im1/im1.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)).flatten(start_dim=1)),
                 (im2/im2.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)).flatten(start_dim=1).T)




remove_im_mean = lambda data : data - data.mean(dim=(1,2,3),keepdims=True )

def im_set_corr(set1, set2, remove_mean=True):
    '''
    im_set: tensor of size N,C,H, W
    '''

    if len(set1.shape) != 4 or len(set2.shape) != 4 :
        raise ValueError('Input shape error')
    if remove_mean: 
        set1 = remove_im_mean(set1)
        set2 = remove_im_mean(set2)

    norms1 = set1.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)
    norms1[norms1 == 0 ] = .001 # to avoid dividing by 0 for blank images 
    norms2 = set2.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)
    norms2[norms2 == 0 ] = .001 # to avoid dividing by 0 for blank images 
    
    return torch.matmul(((set1/norms1).flatten(start_dim=1)),
             (set2/norms2).flatten(start_dim=1).T)

