import numpy as np
import torch.nn as nn
import torch


##################################################################################################################
############################################### Normalization classes ############################################
##################################################################################################################

class normalization(nn.Module): 
    '''
    make normliazation layer
    norm_type: LayerNorm, InstanceNorm, BatchNorm
    In all of them, only divide bt standard deviation. We do not remove mean.
    '''
    def __init__(self, num_kernels, norm_type, epsilon=1e-5):
        super(normalization, self).__init__()                    
        self.norm_type = norm_type.lower()
        self.epsilon = epsilon
        g = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)
        if self.norm_type=='batchnorm':
            self.register_buffer("running_sd", torch.ones(1,num_kernels,1,1))   

    
    def forward(self, x):
        if self.norm_type == "batchnorm":
            return self._batch_norm(x)
        elif self.norm_type == "layernorm":
            return self._layer_norm(x)
        elif self.norm_type == "instancenorm":
            return self._instance_norm(x)
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")
    
    def _layer_norm(self,x):
        sd_x = x.std(dim=(1,2,3) ,keepdim = True, unbiased=False)+ self.epsilon                        
        x = x / sd_x.expand_as(x)
        x = x * self.gammas.expand_as(x)
        return x   
    
    def _instance_norm(self, x):
        sd_x = x.std(dim=(2,3) ,keepdim = True, unbiased=False)+ self.epsilon            
        x = x / sd_x.expand_as(x)
        x = x * self.gammas.expand_as(x)
        return x  
        
    def _batch_norm(self, x): 
        training_mode = self.training       
        sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ self.epsilon )
        if training_mode:
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x) 
            x = x * self.gammas.expand_as(x)
        else:
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)        
        return x   





class BF_batchNorm(nn.Module): 
    def __init__(self, num_kernels, instanceNorm=False):
        super(BF_batchNorm, self).__init__()            
        g = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)
        self.instanceNorm = instanceNorm
        if instanceNorm==False:
            self.register_buffer("running_sd", torch.ones(1,num_kernels,1,1))        
        
    def forward(self, x):
        if self.instanceNorm==False: # do batch norm 
            training_mode = self.training       
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)
            if training_mode:
                x = x / sd_x.expand_as(x)
                with torch.no_grad():
                    self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x) 

                x = x * self.gammas.expand_as(x)

            else:
                x = x / self.running_sd.expand_as(x)
                x = x * self.gammas.expand_as(x)
                
        else: ## do instance norm 
            sd_x = x.std(dim=(2,3) ,keepdim = True, unbiased=False)+ 1e-05
                
            x = x / sd_x.expand_as(x)
            x = x * self.gammas.expand_as(x)

        return x



class LayerNorm(nn.Module):
    '''
    This is both LayerNorm and InstanceNorm. 
    '''
    def __init__(self, num_kernels):
        super(LayerNorm, self).__init__()            
        g1 = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g1, requires_grad=True)
        
    def forward(self, x ):
        sd_x = x.std(dim=(1,2,3) ,keepdim = True, unbiased=False)+ 1e-05                      
        x = x / sd_x.expand_as(x)
        x = x * self.gammas.expand_as(x)
        return x      
        

        
##################################################################################################################
##################### UNet Family - Unconditional  ###############################################################   
##################################################################################################################

 



 
################################################# UNet_flex #################################################   
# reminder: re-write the code for this, using V2 version which is cleaner 
    
class UNet_flex(nn.Module): 
    def __init__(self, args): 
        super(UNet_flex,self).__init__()
        
        args.num_blocks = len(args.num_kernels)-1
        self.num_blocks = args.num_blocks
        self.RF = self.compute_RF(args)
        self.inp_channels = args.num_channels
        self.save_activations_means = False 
        self.save_activations = False
        try: 
            self.dilations = args.dilations
        except AttributeError: 
            self.dilations = None
            
        try:        
            self.NormType = args.NormType 
        except AttributeError: 
            self.NormType = 'batchNorm'            
        
        try:
            self.inter_skip = args.inter_skip
        except AttributeError: 
            self.inter_skip = True
        
        # try: 
            # self.sparsify_phi = args.sparsify_phi
        # except AttributeError: 
            # self.sparsify_phi = False
            
        # try:
        #     self.normalize_phi = args.normalize_phi
        # except AttributeError: 
        #     self.normalize_phi = False
            
        try: #this is here to fix future runs. Had bias in upsampling layers
            self.upsample_with_bias=args.upsample_with_bias
        except AttributeError: 
            self.upsample_with_bias=args.bias
            
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b,args)
                                
        ########## Mid-layers ##########
        self.mid = self.init_mid_block(b,args)
                                    
        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        for b in range(self.num_blocks-1,-1,-1):
            self.upsample[str(b)], self.decoder[str(b)] = self.init_decoder_block(b,args)        

                         
        
    def forward(self, x):
        self.stored_x_means = []
        self.stored_x = []        
        
        pool =  nn.AvgPool2d(kernel_size=2, stride=2, padding=0 )  
        ########## Encoder ##########
        unpooled = []
        for b in range(self.num_blocks): 
            x = self.encoder[str(b)](x)
            # if self.normalize_phi: 
                # x = x/x.mean(dim = (2,3), keepdims= True).norm(dim = 1,keepdim= True)
            if self.inter_skip:
                unpooled.append(x)
            # save the last layer means of channels if needed 
            if self.save_activations_means is True: 
                mean_x = x.mean(dim=(2,3))       
                self.stored_x_means.append(mean_x)     
            if self.save_activations is True:                 
                self.stored_x.append(x)     
                   
            x = pool(x)           


        ########## Mid-layers ##########
        x = self.mid(x)
        # if self.normalize_phi: 
            # x = x/x.mean(dim = (2,3), keepdims= True).norm(dim = 1,keepdim= True)        
            
        # save the last layer means of channels if needed         
        if self.save_activations_means is True: 
            mean_x = x.mean(dim=(2,3))        
            self.stored_x_means.append(mean_x) 
        if self.save_activations is True:            
            self.stored_x.append(x)                
                
        ######### Decoder ##########
        for b in range(self.num_blocks-1, -1, -1):
            x = self.upsample[str(b)](x)
            if self.inter_skip:
                x = torch.cat([x, unpooled[b]], dim = 1)
            x = self.decoder[str(b)](x)    
            # if self.normalize_phi: 
                # if b!=0:
                    # x = x/x.mean(dim = (2,3), keepdims= True).norm(dim = 1,keepdim= True)            
            # save the last layer means of channels if needed 
            if self.save_activations_means is True: 
                if b >0:
                    mean_x = x.mean(dim=(2,3))     
                    self.stored_x_means.append(mean_x)  
            if self.save_activations is True:      
                if b >0:
                    self.stored_x.append(x)   
        
        # if self.sparsify_phi and self.training: 
            # return x, self.save_activations_means[int(len(self.save_activations_means)/2 )]
        # else: 
        return x    
        
        
    def init_encoder_block(self, b, args):
        enc_layers = nn.ModuleList([])
        if b==0: #first layer of first block no normalization 
            enc_layers.append(nn.Conv2d(self.inp_channels ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
            enc_layers.append(nn.ReLU(inplace=True))
            for l in range(1,args.num_enc_conv[b]): 
                enc_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
                enc_layers.append(normalization(args.num_kernels[b],self.NormType)) #added this instead of old normalization layer
                enc_layers.append(nn.ReLU(inplace=True))
        else: 
            for l in range(args.num_enc_conv[b]): 
                if l==0:
                    enc_layers.append(nn.Conv2d(args.num_kernels[b-1] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
                else: 
                    enc_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))                    
                enc_layers.append(normalization(args.num_kernels[b],self.NormType)) #added this instead of old normalization layer            
                enc_layers.append(nn.ReLU(inplace=True))
                                
        return nn.Sequential(*enc_layers)

    def init_mid_block(self, b, args): 
        mid_block = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            if l==0:
                mid_block.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b+1], args.kernel_size, padding=args.padding , bias=args.bias))
            else: 
                mid_block.append(nn.Conv2d(args.num_kernels[b+1] ,args.num_kernels[b+1], args.kernel_size, padding=args.padding , bias=args.bias))    
            
            mid_block.append(normalization(args.num_kernels[b+1],self.NormType)) #added this instead of old normalization layer                       
            mid_block.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*mid_block)
        
    def init_decoder_block(self, b, args):
        dec_layers = nn.ModuleList([])
        
        #initiate the last block:
        if b==0:
            for l in range(args.num_dec_conv[b]-1): 
                if l==0:   # first layer              
                    upsample = nn.ConvTranspose2d(args.num_kernels[b+1], args.num_kernels[b], kernel_size=2, stride=2,bias=False)
                    if self.inter_skip:
                        if self.dilations is None:
                            dec_layers.append(nn.Conv2d(args.num_kernels[b]*2, args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))                                    
                        else: 
                            dec_layers.append(MultiDilationconv(args.num_kernels[b]*2, args.num_kernels[b], kernel_size=args.kernel_size, bias=args.bias,dilations=self.dilations ) )
                    else: 
                        if self.dilations is None:
                            dec_layers.append(nn.Conv2d(args.num_kernels[b], args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))                                                            
                        else: 
                            dec_layers.append(MultiDilationconv(args.num_kernels[b], args.num_kernels[b], kernel_size=args.kernel_size, bias=args.bias,dilations=self.dilations ) )
                            
                else: # other layers
                    if self.dilations is None:
                        dec_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding, bias=args.bias))                 
                    else: 
                        dec_layers.append(MultiDilationconv(args.num_kernels[b], args.num_kernels[b], kernel_size=args.kernel_size, bias=args.bias,dilations=self.dilations ) )
                    
                dec_layers.append(normalization(args.num_kernels[b],self.NormType)) #added this instead of old normalization layer                     
                dec_layers.append(nn.ReLU(inplace=True))
            # the vary last layer of the network (output) 
            dec_layers.append(nn.Conv2d(args.num_kernels[b], args.num_channels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))
            
        #other blocks
        else: 
            for l in range(args.num_dec_conv[b]): 
                if l==0: # first layer    
                    upsample= nn.ConvTranspose2d(args.num_kernels[b+1], args.num_kernels[b], kernel_size=2, stride=2,bias=self.upsample_with_bias) #here
                    if self.inter_skip:     
                        if self.dilations is None:
                            dec_layers.append(nn.Conv2d(args.num_kernels[b]*2, args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))                                    
                        else: 
                            dec_layers.append(MultiDilationconv(args.num_kernels[b]*2, args.num_kernels[b], kernel_size=args.kernel_size, bias=args.bias,dilations=self.dilations ) )
                    else: 
                        if self.dilations is None:                        
                            dec_layers.append(nn.Conv2d(args.num_kernels[b], args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))                                    
                        else: 
                            dec_layers.append(MultiDilationconv(args.num_kernels[b], args.num_kernels[b], kernel_size=args.kernel_size, bias=args.bias,dilations=self.dilations ) )
                            
                else:  # other layers
                    if self.dilations is None:                                        
                        dec_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding, bias=args.bias))
                    else: 
                        dec_layers.append(MultiDilationconv(args.num_kernels[b], args.num_kernels[b], kernel_size=args.kernel_size, bias=args.bias,dilations=self.dilations ) )
                        
                dec_layers.append(normalization(args.num_kernels[b],self.NormType)) #added this instead of old normalization layer                
                dec_layers.append(nn.ReLU(inplace=True))
        return upsample, nn.Sequential(*dec_layers)

    def compute_RF(self,args): 
        '''
        RF is the size of the neighborhood from which one pixel in the last layer of mid block is computed.
        returns a scalar value which is the size of the RF in one dimension
        Assuming all the kernels and strides are squares, not rectangles 
        '''
        r = 0
        ## RF at the end of the last encoder block 
        for b in range(args.num_blocks):
            s = 2**b #effective stride
            r += args.num_enc_conv[b] * ((args.kernel_size-1) * s) + ((2-1) * s) #hard-coded 2 because of 2x2 pooling. Change if different 

        ## RF at the end of the last layer of the mid block 
        s = 2**(b+1)
        r += args.num_mid_conv * ((args.kernel_size-1) * s) 

        r = r+1    
        return r    


   