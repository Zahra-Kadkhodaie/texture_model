import torch
import math

###########################################################################

def add_noise_torch(all_patches, noise_level, sigma_dist='uniform', alpha = 7, coarse=True):
    '''
    Gets images in the form of torch  tensors of size (B, C, H, W)
    @sigma_dist: str:  'uniform' , 'inv_sqrt', 'inv', 'arccos', 'power_law'
    @alpha sets the power in power law dist. If alpha =2 => dist=inv_sqrt
    '''
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = all_patches.device
    N, C, H, W = all_patches.size()

    ####### get the sigmas 
    
    #for blind denoising
    if isinstance(noise_level, (list, tuple)):
        low, high = float(noise_level[0]), float(noise_level[1])
        u = torch.rand(size=(N,1,1,1), device=device)
        eps = 1e-12
        low_pos  = max(low,  eps)
        high_pos = max(high, eps)
        
        if sigma_dist=='uniform':  
            std = u *(high- low)  +low
        elif sigma_dist =='inv': 
            # log-uniform sampling (p(sigma) ∝ 1/sigma)
            log_min, log_max = math.log(low), math.log(high)
            std = torch.exp(u * (log_max - log_min) + log_min)

        elif sigma_dist =='arccos': 
            std = torch.arccos(1.0 - 2.0 * u) / math.pi
            std = std *(high- low)  + low

            
        elif sigma_dist =='inv_sqrt': 
            std = u *( math.sqrt(high)- math.sqrt(low) )  + math.sqrt(low)
            std = torch.square(std)
        
        elif sigma_dist =='power_law': 
            #inv_sqrt is a special cases of this with alpha = 2
            #uniform is a special cases of this with alpha = 1
            std = u *( high**(1/alpha) - low**(1/alpha) )  + low**(1/alpha)
            std = std**alpha
        
        elif sigma_dist in ('inv_sq', 'inverse_square'):
            # p(sigma) ∝ 1/sigma^2 on [a,b]
            # CDF F(s) = (1/a - 1/s) / (1/a - 1/b)
            # Inverse: s = 1 / ( 1/a - u*(1/a - 1/b) )
            a, b = low_pos, high_pos
            denom = (1.0/a - 1.0/b)
            # guard against numerical issues if a≈b
            denom = denom if abs(denom) > 1e-18 else 1e-18
            inv_s = (1.0/a) - u * denom
            # avoid divide-by-zero
            inv_s = torch.clamp(inv_s, min=1e-18)
            std = 1.0 / inv_s
        
        elif sigma_dist in ('inv_1p5', 'inverse_3_2'):
            # p(s) ∝ 1/s^(3/2); inverse CDF: s = [ a^{-1/2} + u*(b^{-1/2} - a^{-1/2}) ]^{-2}
            a, b = low_pos, high_pos            
            a_mh = a**(-0.5)
            b_mh = b**(-0.5)
            lin = a_mh + u * (b_mh - a_mh)
            lin = torch.clamp(lin, min=1e-12)
            std = lin.pow(-2.0)
            
    #for specific noise
    else:
        std = torch.ones(N,1,1,1,  device = device) * noise_level

    # put in range 
    std = std/255 

    ######## add noise
    if coarse:
        noise_samples = torch.randn(size = all_patches.size() , device = device) * std            
        noisy = noise_samples+ all_patches
            
    else:
        noise_samples = torch.randn(size = (N,3,H,W) , device = device) * std
        noisy = torch.cat((all_patches[:, 0:1, :,:], all_patches[:, 1::, :,:] + noise_samples), dim=1)

    
    return noisy, noise_samples, std

###########################################################################

def add_noise_torch_range(im, noise_range, device=None, coarse=True):
    '''
    For images and wave coeffs
    Gets image- torch  tensors of size (1, H, W)
    @noise_rangel: sigmas for images in range 0-255 -tensor
    '''
    C,H,W = im.shape
    images = torch.stack([im]*noise_range.shape[0])
    noise_range = noise_range/255
    noise_samples = torch.randn(size = images.size() , device = device) * noise_range

    if coarse:
        noisy = images + noise_samples

    else:
        noise_samples = noise_samples[:,0:3]
        noisy = torch.cat((images[:, 0:1, :,:], images[:, 1::, :,:] + noise_samples), dim=1)

    return noisy , noise_samples    

###########################################################################

def add_noise_coeffs_torch_range(im, noise_range, device=None):
    '''
    Gets image- torch  tensors of size (1, H, W)
    @noise_rangel: sigmas for images in range 0-255 -tensor
    '''
    images = torch.stack([im]*noise_range.shape[0])
    noise_range = noise_range/255
    noise_samples = torch.randn(size = images.size() , device = device) * noise_range
    noise_samples = noise_samples[:,0:3]
    noisy = torch.cat((images[:, 0:1, :,:], images[:, 1::, :,:] + noise_samples), dim=1)
    return noisy , noise_samples

