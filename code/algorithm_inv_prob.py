import numpy as np
import torch
import time
from dataloader_func import rescale_image_range
from torch.optim import Adam
import torch.nn as nn 
from model_loader_func import get_channel_means
### Takes a tensor of size (n_ch, im_d1, im_d2)
### and returns a tensor of size (n_ch, im_d1, im_d2)



###############################################################

#########################################################
############### self conditional synthesis ##############
#########################################################


def backward_sampling(model, x, x_c , sigmas, sig_c=None, max_iter=200, seed=None, K=None , average_phi=False):
    '''
    This is the old algorithm. Will be retired
    '''
    all_x = {}
    all_losses = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for sig in sigmas:
        x_noisy, all_loss = generate_metamer(model, x, x_c , sig, sig_c, max_iter, seed, K,average_phi)
        # pass the metamer through the entire net (encoder and decoder)
        x = model(x_noisy.to(device)).detach()
        all_x[sig] = x
        all_losses[sig] = all_loss
        if seed is not None: 
            seed = seed+ 1        
    return x, all_x, all_losses


def generate_metamer(model, x, x_c , sig, sig_c=None, max_iter=200, seed=None, K= None , average_phi=False): 
    all_loss = []  
    
    ## no grad for model params 
    for param in model.parameters():
        param.requires_grad = False

    ## chooose the loss function 
    criterion =  nn.MSELoss(reduction='none')
    
    # set the seed for noise added to input 
    if seed is not None:
        torch.manual_seed(seed)        

    if len(x_c.shape) == 4:
        # add noise to target (if not specified, same noise level to both input and conditioner)
        if sig_c is None:
            x_c_noisy = x_c + torch.randn_like(x_c)* sig        
        else: 
            x_c_noisy = x_c + torch.randn_like(x_c)* sig_c
        
        x_c_noisy.requires_grad = False
    
        phi_c = get_channel_means(model, x1 = x_c_noisy, average_phi=average_phi, noGrad=True )
        b = int(len(phi_c)/2) #pick the mid block
        target_phi = phi_c[b]
        if average_phi: 
            target_phi = torch.tile(target_phi, (x.shape[0],1,1,1) )
        

    else:
        target_phi = x_c.unsqueeze(2).unsqueeze(3)
        b = 3
    
    # if K, only keep the top K values of phi        
    if K is not None:
        topk_ids = target_phi.topk(dim =1, k=K)[1] 
        mask = torch.zeros_like(target_phi, dtype=torch.bool).scatter_(dim=1, index=topk_ids, value=True)
        target_phi = target_phi * mask 

         
    # add noise to init image 
    x_noisy = x + torch.randn_like(x) * sig
    x_noisy.requires_grad = True
    
    optimizer = Adam( [x_noisy], lr = 0.01 )
    
    ## backprop to match the phi
    iter = 0
    loss = torch.ones(2)
    loss_thresh = torch.ones(2)*.1
    while (loss > loss_thresh).any():
        optimizer.zero_grad()
        phi_inp = get_channel_means(model, x1 = x_noisy , noGrad=False)         
        loss = criterion( phi_inp[b], target_phi ).mean(dim=(1,2,3)) 
        loss.mean().backward()
        optimizer.step()
        all_loss.append(loss.detach())
        
        if iter ==0: 
            loss_thresh = loss.detach()/100
            # print(loss_thresh)
        iter += 1
        if iter >= max_iter: 
            # print('not converged')
            break
                    
    # print(f"sig {sig}, iter {iter}, Loss: {loss.mean().detach() }")

    return x_noisy.detach(),torch.stack(all_loss)    






#########################################################
############### self conditional synthesis ##############
#########################################################



def get_activations_unconditional(unet, x, noGrad=True): 
    
    '''
    returns activations of unconditinoal UNet
    @x: a batch of images. tensor of size (N, C, H, W)
    out: a list of activations.  The length is equal to number of blocks - 1  
    The list consists of tensors of size (N, C', H',W') where C' is the number of channels in that layer
    '''    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set network in saving mode
    unet.save_activations = True

    #compute the activations  
    with torch.no_grad() if noGrad else torch.enable_grad():
        out = unet(x.to(device))
        activations = unet.stored_x

    unet.save_activations = False

    return activations
    
def get_activation_means_unconditional(unet, x, noGrad=True): 
    
    '''
    returns means of activations (phi) of unconditinoal UNet
    @x: a batch of images. tensor of size (N, C, H, W)
    @centroids: if True, the average phis are returned 
    out: a list of phis. The length is equal to number of blocks - 1  
    The list consists of tensors of size (N, C') where C' is the number of channels in that layer
    '''    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set network in saving mode
    unet.save_activations_means = True    
    
    #compute the activation means
    with torch.no_grad() if noGrad else torch.enable_grad():
        out = unet(x.to(device))
        phi = unet.stored_x_means
    # turn off save mdoe
    unet.save_activations_means = False
    return phi

def get_phi_centroids(phi, max=False):
    '''
    get a list of phis and return the centroids 
    input size: list of tensors of size (N, C')
    output size:list of tensors of size (1, C')
    '''
    phi_centroids= []
    for i in range(len(phi)): 
        if max: 
            means = phi[i].max(0,keepdim=True)[0]
        else: 
            means = phi[i].mean(0,keepdim=True)
        phi_centroids.append(means)
               
    return phi_centroids


def repeat_phi(phi, repeats): 
    '''
    @phi: a list that contains one phi of size (1, C')
    '''
    for i in range(len(phi)): 
        phi[i] = torch.tile(phi[i],(repeats,1 ))
    return phi



def keep_topK_phi(target_phi):
    # if K, only keep the top K values of phi        
    if K is not None:
        topk_ids = target_phi.topk(dim =1, k=K)[1] 
        mask = torch.zeros_like(target_phi, dtype=torch.bool).scatter_(dim=1, index=topk_ids, value=True)
        target_phi = target_phi * mask 
    return target_phi    

def phi_c_noise_averaged(model, x_c, sigs, n_noise ) : 
    '''
    phi from noisy conditioner. For each image in x_c, averaged over n_noise realizations 
    @x_c: tensor of size (N_c, C_c, H_c, W_c)
    @sigs:  tensor of size (N_c,1,1,1) that contains noise level to be added in (0,1)
    @n_noise: number of noise repeats to be averaged over 
    @output: phi averaged over noise realizations
    '''
    N_c, C_c, H_c, W_c = x_c.shape
    
    #repeat the clean conditioners
    x_c_repeated = torch.tile(x_c, (n_noise, 1,1,1))
    sigs_repeated = torch.tile(sigs, (n_noise, 1,1,1))
    ## add noise 
    x_c_repeated_noisy = x_c_repeated + torch.randn_like(x_c_repeated) * sigs_repeated
    
    # get the phi for conditioner 
    x_c_repeated_noisy.requires_grad = False
    phi_c = get_activation_means_unconditional(model, x= x_c_repeated_noisy )                                                

    phi_c_ave = []
    for i in range(len(phi_c)):
        # phi_c_mid = phi_c[int(len(phi_c)/2)] # N_c*n_noise, 512 
        phi_c_scale = phi_c[i]
        phi_c_scale_ave = phi_c_scale.view( n_noise ,N_c, phi_c_scale.shape[1]).mean(dim=0) # avergae over noise realizations 
        phi_c_ave.append(phi_c_scale_ave)
    return phi_c_ave



def phi_c_many_noise(model, x_c, sigs, n_noise ) : 
    '''
    phi from noisy conditioner. For each image in x_c, averaged over n_noise realizations 
    @x_c: tensor of size (N_c, C_c, H_c, W_c)
    @sigs: tensor of size (N_c,1,1,1) contains noise level to be added in (0,1)
    @n_noise: number of noise repeats to be averaged over 
    @output: phi in the midfle block averaged over noise realizations
    '''
    N_c, C_c, H_c, W_c = x_c.shape
    #repeat the clean conditioners
    x_c_repeated = torch.tile(x_c, (n_noise, 1,1,1))
    sigs_repeated = torch.tile(sigs, (n_noise, 1,1,1))
    ## add noise 
    x_c_repeated_noisy = x_c_repeated + torch.randn_like(x_c_repeated) * sigs_repeated
    
    # get the phi for conditioner 
    x_c_repeated_noisy.requires_grad = False
    phi_c = get_activation_means_unconditional(model, x= x_c_repeated_noisy )                                                

    phi_c_all = []
    for i in range(len(phi_c)):
        # phi_c_mid = phi_c[int(len(phi_c)/2)] # N_c*n_noise, 512 
        phi_c_scale = phi_c[i].view( n_noise ,N_c, phi_c[i].shape[1]) 
        phi_c_all.append(phi_c_scale)
    return phi_c_all


def match_phis_mid_block(model, x_noisy, target_phi ,  max_iter=200,  lr= 0.01, phi_c_mask=None): 
    '''
    takes a model and a noisy input and changes the x_noisy to match its phi with target_phi in the mid block. 
    x_noisy: tensor of size (N, C, H, W). images.     
    target_phi: Pre-computed from a conditioner image. A list of len (n_blocks - 1). Each tensor in the list of size (N,C').
    @b: the blocks to be matched 
    The mid layer will be picked up and the size of that should be (N, C') where C' is normally 512.
    '''

    if phi_c_mask is not None: 
         target_phi[int(len(target_phi)/2)] = target_phi[int(len(target_phi)/2)] * phi_c_mask
    ## no grad for model params 
    for param in model.parameters():
        param.requires_grad = False

    ## chooose the loss function 
    criterion =  nn.MSELoss(reduction='none')    
    x_noisy.requires_grad = True
    optimizer = Adam( [x_noisy], lr = lr )
    
    ## backprop to match the phi
    all_loss = [] 
    iter = 0
    loss = torch.ones(2)
    loss_thresh = torch.ones(2)* .5
    while (loss > loss_thresh).any():
        optimizer.zero_grad()
        phi_inp = get_activation_means_unconditional(model, x= x_noisy , noGrad=False)#requires grad     
        b = int(len(phi_inp)/2) # pick the mid block
        loss = criterion( phi_inp[b], target_phi[b] ).mean(dim=(1)) 
        loss.mean().backward()
        optimizer.step()
        all_loss.append(loss.detach())
        
        if iter ==0: 
            loss_thresh = loss.detach()/100 #set the loss threshold to 1% of the loss in 1st step
            if phi_inp[b].shape != target_phi[b].shape: 
                raise ValueError('target size is wrong!')
        iter += 1
        if iter >= max_iter: 
            break
                    
    return x_noisy.detach(),torch.stack(all_loss)    


    




def self_conditional_sampling(model, 
                              x, 
                              x_c , 
                              sigmas, 
                              sig_c=None, 
                              max_iter=200, 
                              seed= None, 
                              skip=False,
                              n_noise=1,                               
                              centroids=False,
                              phi_mask = None,                              
                              max=False, 
                              phi_c_mask=None ,
                              lr = .01, 
                              return_lists = True
                              ):

    '''
    Takes a model and a conditioner and a sigma schedule. Generates images conditioned on representation of the conditioner.
    @x: input of size (N,C,H,W). Output will be the same size. This is normally a constant tensor set to distribution mean (no noise). 
    @x_c:conditioner: clean conditioner images of size (N_c,C,H_c,W_c)
        Permitted size settings:
        a) N_c --> N  (N_c separate conditioners, N samples): N_c and N must be equal.
        b) 1 --> N  (1 conditioner, N samples): conditioner expands to size N. 
        c) N_c --> 1 centroid --> N tiled centroid--> N (N_c conditioners, to collapse to one centroid, N samples)
        In all cases, size of initial sample of noise and size of generated sample should match.
    @sigmas: schedule for noise levels 
    @seed 
    @skip                                                            
    @n_noise: number of noise realizations on each conditioner image in x_c                              
    @centroids
    @K
    @max_iter                            
    @lr:learning rate of the mathcing algorithm
    @return_lists
    '''    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # N, C, H, W = x.shape
    # N_c = x_c.shape[0]
    # ### if 1 conditioner to batch sampling, repeat the conditioner 
    # if (N_c == 1) & (N !=1): 
    #     x_c = torch.tile(x_c, (N, 1,1,1))
    
    x = x.to(device)
    x_c = x_c.to(device)
    N, C, H, W = x.shape
    N_c, C_c, H_c, W_c = x_c.shape
    if N_c != N: 
        if N_c == 1: # if 1 conditioner to batch sampling, repeat the conditioner
            x_c = torch.tile(x_c, (N, 1,1,1))
        else: 
            if centroids: 
                pass 
            else: 
                raise ValueError('sizes of x and x_c are not among the permitted settings!')

    if phi_mask is not None: 
        phi_mask = torch.tile(phi_mask, (N,1) )
        
    #### first get the target_phi
    ## if phi_c computed only at one noise level:
    phi_x_c_seq =[]
    if sig_c is not None: 
        phi_x_c = phi_c_noise_averaged(model=model, x_c=x_c, sigs=torch.tile(torch.tensor(sig_c, device=device), (x_c.shape[0],1,1,1)), n_noise=n_noise )
        if centroids: 
            phi_x_c = get_phi_centroids(phi_x_c, max)
            phi_x_c = repeat_phi(phi_x_c, repeats= N )
        phi_x_c_seq.append(phi_x_c) 
        phi_x_c_seq = phi_x_c_seq * len(sigmas) # repeat same phi_c for all noise levels
        
    ## if phi_c computed at all sigmas:
    else: 
        for sig in sigmas:
            phi_x_c = phi_c_noise_averaged(model=model, x_c=x_c, sigs=torch.tile(sig.to(device), (x_c.shape[0],1,1,1)), n_noise=n_noise )
            if centroids: 
                phi_x_c = get_phi_centroids(phi_x_c, max)
                phi_x_c = repeat_phi(phi_x_c, repeats= N )
            phi_x_c_seq.append(phi_x_c)                 
        
        if phi_mask is not None: 
            phi_x_c[int(len(phi_x_c)/2) ] = phi_x_c[int(len(phi_x_c)/2)] * phi_mask
    #### Alternate between matching phi and score step
    all_x = []
    all_losses = []
    i = 0
    for sig in sigmas:
        # match phi
        print('-------', sig)
        if seed is not None:
            torch.manual_seed(seed + i)
        x_noisy = x + torch.randn_like(x) * sig
        x_noisy, loss = match_phis_mid_block(model=model, 
                                                 x_noisy=x_noisy, 
                                                 target_phi=phi_x_c_seq[i] , 
                                                 max_iter=max_iter,  
                                                 lr=lr, 
                                                  phi_c_mask=phi_c_mask)
        
        # pass the metamer through the entire net (encoder and decoder)
        if skip:
            x = x_noisy- model(x_noisy.to(device)).detach()
        else:     
            x =  model(x_noisy.to(device)).detach()
  
                
        all_x.append(x)
        all_losses.append(loss)
        i += 1
    
    # return x, all_x, all_losses
    if return_lists:
        return x, all_x, all_losses
    else: 
        return x
        

#########################################################
############### self conditional synthesis ##############
#########################################################


def self_conditional_sampling_adaptive(model, 
                              x, 
                              x_c, 
                              sig_0=1, 
                              sig_L=.01, 
                              h0=.01, 
                              beta=.01, 
                              freq=0,
                              fixed_h = True,
                              max_T=None, 
                              seed=None, 
                              skip=True,                                                            
                              n_noise=1,                               
                              centroids=False,
                              phi_mask = None,
                              K=None ,
                              max_iter=200,                               
                              lr = .01, 
                              return_lists= False
):
    
    '''
    Takes a model and a conditioner. Generates images conditioned on representation of the conditioner.
    @x: input of size (N,C,H,W). Output will be the same size. This is normally a constant tensor set to distribution mean (no noise). 
    @x_c:conditioner: clean conditioner images of size (N_c,C,H_c,W_c)
        Permitted size settings:
        a) N_c --> N  (N_c separate conditioners, N samples): N_c and N must be equal.
        b) 1 --> N  (1 conditioner, N samples): conditioner expands to size N. 
        c) N_c --> 1 centroid --> N tiled centroid--> N (N_c conditioners, to collapse to one centroid, N samples)
        In all cases, size of initial sample of noise and size of generated sample should match.
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @freq: frequency at witch to retain outputs
    @fixed_h:
    @max_T 
    @seed 
    @skip                                                            
    @n_noise: number of noise realizations on each conditioner image in x_c                              
    @centroids
    @K
    @max_iter                            
    @lr:learning rate of the mathcing algorithm    
    '''    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start_time_total = time.time()     
    ###### lists to collect stuff ######
    intermed_Ys=[]
    sigmas = []
    means = []
    all_losses = []

    ###### verify sizes  ######
    x = x.to(device)
    x_c = x_c.to(device)
    N, C, H, W = x.shape
    dim =  C*H*W # ambient dimensionality 
    N_c, C_c, H_c, W_c = x_c.shape
    if N_c != N: 
        if (N_c == 1): # if 1 conditioner to batch sampling, repeat the conditioner
            x_c = torch.tile(x_c, (N, 1,1,1))
        else: 
            if centroids: 
                pass 
            else: 
                raise ValueError('sizes of x and x_c are not among the permitted settings!')
                
    if phi_mask is not None: 
        phi_mask = torch.tile(phi_mask, (N,1) )

    ###### initializations ######
    t=1
    sigma = torch.ones((N,1,1,1), device=device)*sig_0 #one scalar sigma per input image
    update_mask = torch.ones((N,1,1,1), device=device) #mask for stopping updates on individual images in x    
    x = torch.normal(x , sigma ) # Add noise to initial images 
    if freq > 0:
        intermed_Ys.append(x)
        
    ###### iterate until convergence     
    while sigma.max() > sig_L :  
        
        h = h0        
        if fixed_h is False:
            h = h0*t/(1+ (h0*(t-1)) )

        ### Matching step ###
        phi_x_c = phi_c_noise_averaged(model=model, x_c=x_c[ update_mask[:,0,0,0] ==1] , sigs=sigma[ update_mask[:,0,0,0] ==1] , n_noise=n_noise ) # compute phi of conditioner
        
        if phi_mask is not None: 
            phi_x_c[3] = phi_x_c[3] * phi_mask
            
        if centroids: 
            phi_x_c = get_phi_centroids(phi_x_c)
            phi_x_c = repeat_phi(phi_x_c, repeats= int(update_mask.sum().item() )  )
        
        x_updated , loss = match_phis_mid_block(model=model, x_noisy=x[ update_mask[:,0,0,0] ==1]  , target_phi=phi_x_c, max_iter=max_iter,  lr=lr)
        x[update_mask[:,0,0,0] == 1] = x_updated
        all_losses.append(loss)
        
        ### score step ###
        with torch.no_grad():
            if skip:            
                f_x = model(x) 
            else: 
                f_x = x - model(x)                  
        
        #estimate sigma                
        sigma = torch.norm(f_x, dim=(2,3),keepdim=True).norm(dim=1,keepdim=True)/np.sqrt(dim)
        sigmas.append(sigma)
        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))
        noise = torch.randn(N,C, H, W, device=device) 
        update_mask[sigma<sig_L] = 0 

        #take the score step 
        x = x -  (h*f_x + gamma*noise ) * update_mask
        means.append(x.mean(dim=(2,3)) )        

        
        ### estimate sigma after score step ###
        with torch.no_grad():
            if skip:            
                f_x = model(x) 
            else: 
                f_x = x - model(x)                  
        #estimate sigma                
        sigma = torch.norm(f_x, dim=(2,3),keepdim=True).norm(dim=1,keepdim=True)/np.sqrt(dim)
        update_mask[sigma<sig_L] = 0 

        
        t +=1
        if max_T is not None and t>max_T:
            print('max T surpassed')
            break
        if sigma.max() > 5:
            print('not converging')
            break
            
        if freq > 0 and t%freq== 0:
            print('-----------------------------', t)
            print('max sigma ' , sigma.max().item() )
            print('mean ', x.mean().item() )
            # print(torch.hstack((update_mask, sigma )).squeeze() )
            if skip: 
                intermed_Ys.append( (x-f_x.detach() ))
            else: 
                intermed_Ys.append( x.detach())
                
    
    print('-------- total number of iterations: ', t)
    print("-------- final max sigma, " , sigma.max().item() )
    print('-------- final mean ', x.mean(dim=(2,3)).mean().item() )
    print("-------- final snr, " , 20*torch.log10((x.std()/sigma)).mean().item() )

    if skip:
        denoised_x = x - model(x)  
    else: 
        denoised_x = model(x)  

    if return_lists:
        return denoised_x.detach(), intermed_Ys, sigmas, means, all_losses
    else: 
        return denoised_x.detach()
    


