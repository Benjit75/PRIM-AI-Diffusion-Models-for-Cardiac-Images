#from torch import nn, einsum



"""
def zero_module(module):
    
            #Zero out the parameters of a module and return it.
        
            for p in module.parameters():  ##Est-ce que j'ai cette fonction au moins ?
                p.detach().zero_()
            return module




def make_zero_conv(self, channels):
        return self.time_mlp(self.zero_module(conv_nd(self, channels, channels, 1, padding=0)))



def conv_nd(dims, *args, **kwargs):
    
   #Create a 1D, 2D, or 3D convolution module.
        
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

"""