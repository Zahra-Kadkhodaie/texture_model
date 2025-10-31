import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import pickle
from network import *



##################################################################################################

def initialize_network(network_name, args):
    '''
    Function to dynamically initialize a neural network by class name
    '''
    if network_name in globals() and issubclass(globals()[network_name], nn.Module):
        return globals()[network_name](args)
    else:
        raise ValueError(f"Network {network_name} not found or not a subclass of nn.Module")

        
def load_learned_model(folder_path, print_args=False, new_arch=None, return_args=False): 
    '''
    Loads dictionary of all args used to define the model for training and then loads the saved trained model with the specified parameters.
    This can only be used if the network parameters were saved in advanced.
    '''
    with (open(folder_path +'exp_arguments.pkl' , "rb")) as openfile:
        arguments = pickle.load(openfile)    
    
    if print_args: 
        print('*************** saved arguments:*************')
        for key,v in arguments.items(): 
            print(key, v)
    parser = argparse.ArgumentParser(description='set CNN args')

    for k,v in arguments.items(): 
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args('')

    if new_arch is not None: 
        args.arch_name = new_arch
    model = initialize_network(args.arch_name, args)
    if torch.cuda.is_available():
        model = model.cuda()

    if new_arch is None: 
        model = read_trained_params(model, folder_path + '/model.pt')
    else: 
        model = read_trained_params(model, folder_path + '/model.pt', strict=False)
        
    print('******************************************************')
    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.eval()
    print('train mode:', model.training )
    if return_args: 
        return model, args
    else: 
        return model      


##################################################################################################
def read_trained_params(model, path, strict=True): 
    '''reads parametres of saved models into an initialized network'''
    if torch.cuda.is_available():
        learned_params =torch.load(path, weights_only=True)
    else:
        learned_params =torch.load(path, map_location='cpu' , weights_only=True)
        
    ## unwrap if in Dataparallel 
    new_state_dict = {}
    for key,value in learned_params.items(): 
        if key.split('.')[0] == 'module': 
            new_key = '.'.join(key.split('.')[1::])
            new_state_dict[new_key] = value

        else: 
            new_state_dict[key] = value
        

    model.load_state_dict(new_state_dict, strict=strict)        
    model.eval();

    return model

################################################UNet_flex##################################################


##################################################################################################

def compute_UNet_RF(num_enc_conv,  num_mid_conv, kernel_size):
    '''
    RF is the size of the neighborhood from which one pixel in the last layer of mid block is computed.
    returns a scalar value which is the size of the RF in one dimension
    Assuming all the kernels and strides are squares, not rectangles 
    '''
    num_blocks = len(num_enc_conv)
    r = 0
    ## RF at the end of the last encoder block 
    for b in range(num_blocks):
        s = 2**b #effective stride
        r += num_enc_conv[b] * ((kernel_size-1) * s) + ((2-1) * s) #hard-coded 2 because of 2x2 pooling. Change if different 

    ## RF at the end of the last layer of the mid block 
    s = 2**(b+1)
    r += num_mid_conv * ((kernel_size-1) * s)

    r = r+1
    return r







##################################################################################################



def get_channel_means(unet, x1, x2=None, average_phi=False, with_params=True, return_activations = False, noGrad=True): 
    '''
    returns means of channels (phi) of a conditional or unconditinoal UNet
    if conditional unet: returns means of channels for either input or conditioner image, or both
    if unconditional unet: returns means of channels for input
    @x1:
    @x2:
    @with_param: important: set to False if phi is computed to be given to the network (instead of x)
    '''    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set network in saving mode
    unet.save_activations_means = True    
    if return_activations: 
        unet.save_activations = True

    #compute the responses 
    with torch.no_grad() if noGrad else torch.enable_grad():
        try: #for conditional model
            if x2 is None: #if only conditioner image is given 
                out = unet(x1.to(device), x1.to(device), average_phi)
            else: # if a noisy input is also given
                out = unet(x2.to(device), x1.to(device), average_phi)
                x_means = unet.stored_x_means
                
            x_c_means = unet.stored_x_c_means
            x_c = unet.stored_x_c
            
        except TypeError: # for the unconditional model
            out = unet(x1.to(device))
            x_c_means = unet.stored_x_means
            x_c = unet.stored_x
            if average_phi: 
                for i in range(len(x_c_means)): 
                    x_c_means[i] = x_c_means[i].mean(0,keepdim=True)
    unet.save_activations_means = False
    unet.save_activations = False


    if with_params:   
        try:
            params = get_matching_params(unet)
            for i in range(len(params)): 
                x_c_means[i] = x_c_means[i] * params[i]
        except AttributeError: 
            pass

    if (x2 is None) & (return_activations is False): 
        return x_c_means
    elif (x2 is None) & (return_activations is True): 
        return x_c_means, x_c
    elif (x2 is not None) & (return_activations is False): 
        return x_c_means, x_means
    else: 
        return x_c_means, x_means, x_c

def get_matching_params(unet): 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    params = []
    for l in range(len(unet.encoder_matching_params)):
        params.append(unet.encoder_matching_params[str(l)].detach().to(device))
    params.append(unet.mid_matching_params.detach().to(device))
    for l in range(len(unet.decoder_matching_params)-1, 0,-1):
        params.append(unet.decoder_matching_params[str(l)].detach().to(device))
    return params







