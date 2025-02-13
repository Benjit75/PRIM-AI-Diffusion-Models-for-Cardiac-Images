import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import math
from einops import rearrange

from models.model import * #exists, default, Residual, Downsample, Upsample, SinusoidalPositionEmbeddings, ConvNextBlock, Attention, LinearAttention, PreNorm

#from ControlNet.util import zero_module, make_zero_conv, conv_nd

## input controlnet = input Zt Unet + zero conv(condition Cf) ???

## Since we send the ith down layer of the controlNet to a zero-convolution 
## and then to the block of the up sampling of the unet ,we have that we need
## somewhere to put the output of the zero_conv somewhere either in the output
## of the control net and his modules, or in the "control" entity which is
## a list of the blocks which we applied zero_convolution to 


## HOW TO IMPLEMENT THE ZERO CONVOLUTION ????

## HOW TO MAKE SURE DIMENSIONS ARE OK FOR CONDITION AND INPUT (No latent space yet)

## HOW TO MAKE SURE I GET CONTROL AND GET THE DOWNSAMPLING PART ONLY ???
def f(x):
    return x+10


def g(x):
    return x



class ControlNet(nn.Module):

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        hint_channels=1,
        with_time_emb=True,
        convnext_mult=2,
    ):
        super().__init__()

        print(1)

        print(2)
        # determine dimensions
        self.channels = channels
        self.hint_channels = hint_channels ##à vérifier

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        
        
        block_klass = partial(ConvNextBlock, mult=convnext_mult)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None


        

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        #self.zero_convs = nn.ModuleList([self.make_zero_conv(32)])
        self.zero_convs = nn.ModuleList()
        self.dims = []

        """
        Ici vu que j'ai enlevé le UPSAMPLING est-ce que cela change le make_zero_conv ?
        nn.ModuleList([self.make_zero_conv(model_channels)]) à voir si c'est bon ça 
        me paraît un peu simple
        """


        self.input_hint_block = self.zero_module(self.conv_nd(2, self.hint_channels, 32, 3, padding=1))
        ## Problème : on a time_dim qui vaut 4*dim alors que self.

        #print(self.input_hint_block.shape)

        num_resolutions = len(in_out)

        """
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
        """

        
        

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            
            self.dims.append(dim_out)

            self.zero_convs.append(self.make_zero_conv(dim_out))
            ### Est-ce que cela marche ???? J'ai remplacé ch par dim_out


            ##self.zero_convs.append(self.make_zero_conv(ch))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.middle_block_out= self.make_zero_conv(mid_dim)


        
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )
        





        #self.zero_convs = 

        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        

        

    


               ## self.time_mlp(zero_module(conv)) ???

               ##TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
## self.time_mlp(zero_module(conv???))

    


    """
    Je dois définir  CONV_ND et ZERO_MODULE dans leur github...

    DONE EN BAS DU CODE EN COMMENTAIRE
    """


    def make_zero_conv(self, channels):
        return self.zero_module(self.conv_nd(2, channels, channels, 1, padding=0))
    
    #return self.time_mlp(self.zero_module(self.conv_nd(self, channels, channels, 1, padding=0)))
    
    def conv_nd(self,dims, *args, **kwargs):
    
   #Create a 1D, 2D, or 3D convolution module.
        
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        else:
            return nn.Conv3d(*args, **kwargs)
        ##raise ValueError(f"unsupported dimensions: {dims}")
    
    def zero_module(self,module):
    
            #Zero out the parameters of a module and return it.
        
        for p in module.parameters():  ##Est-ce que j'ai cette fonction au moins ?
            p.detach().zero_()
        return module


    def forward(self, x, hint, time):

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None


        outs = []


        guided_hint = self.input_hint_block(hint)

        #print(guided_hint.shape)

        ##Guided_hint dans la bonne dimension ??

        # downsample
        a = 0
        for block1, block2, attn, downsample in self.downs:

            zero_conv = self.zero_convs[a]
            a += 1

            #print(f"zero_conv weight shape: {zero_conv.weight.shape}")  # Print weight shape
            #print(f"zero_conv bias shape: {zero_conv.bias.shape if zero_conv.bias is not None else 'No bias'}")  # Print bias shape if exists
            #print(f"Step {a}: Before block1, x shape: {x.shape}")  # Print x shape before block1

            

            x = block1(x, t)
            #print(f"Step {a}: After block1, x shape: {x.shape}")  # Print x shape after block1
        
            x = block2(x, t)
            #print(f"Step {a}: After block2, x shape: {x.shape}")  # Print x shape after block2
        
            x = attn(x)
            #print(f"Step {a}: After attn, x shape: {x.shape}")  # Print x shape after attn


            if guided_hint is not None:
                x += guided_hint
                #print(f"Step {a}: After adding guided_hint, x shape: {x.shape}")  # Print x shape after guided_hint
                guided_hint = None  # Reset guided_hint


            outs.append(zero_conv(x))  # Apply zero_conv
            #print(f"Step {a}: After zero_conv, x shape: {outs[-1].shape}")  # Print x shape after zero_conv
            
            x = downsample(x)  # Apply downsampling
            #print(f"Step {a}: After downsampling, x shape: {x.shape}")

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        ##x += guided_hint

        outs.append(zero_conv(x))

        return outs



        """
        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
"""

## Je crois que je renvoie h auquel j'ai ajouté le moddile block et auquel 
## j'ai appliqué dans chaque terme une ZERO_CONV

        #return self.final_conv(x)     #EST CE QUE JE RENVOIE X OU CA ??


"""
 def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)


## Ici on a EMB ça veut dire qu'on doit ajouter le time embedding à ce block ???
## Et pourquoi est-ce qu'on a pas de x ??? Où est la première zero_conv ???
## J'ai l'impression que ce n'est pas le même schéma que ce que j'ai vu
### CHECKER INPUT HINT BLOCK


        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint ###D'OU VIENT CETTE ADDITION ???
                guided_hint = None ###On alterne guided-hint et non guided hiny ??
                ## voir dans self.input_hint_block
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context)) ## On a bien la zero conv

        h = self.middle_block(h, emb, context) 

        outs.append(self.middle_block_out(h, emb, context)) ## Pourquuoi pas de zero conv ni de hint?

           ### MIDDLE BLOCK OUT ????

        return outs
        """


"""
 def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
"""


## Ils ajoutent progressivement des zeros convs pour chaque layer (car chaque
## layer a une taille diff, avec comme param les channels)

## Sauf pour l'input où ils utilisent directement 
##zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)) ln 162

"""
self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
self.zero_convs.append(self.make_zero_conv(ch))

Ainsi que pour le middle_block 

self.middle_block_out = self.make_zero_conv(ch)

Ici ils ont utilisé ch = out_ch = out_channels pour chaque couche permettant
d'avoir la bonne dimension pour la zero_convolution

J'ai juste à faire la même chose dans la définition du Unet pour faire
en parallèle de mes blocks une liste de zero_convs

La seule incertitude c'est quelle conv est-ce que j'utilise ??

"""



"""
GITHUB : Conv_nd ET Zero_module


def zero_module(module):
    
    #Zero out the parameters of a module and return it.
    
    for p in module.parameters():  ##Est-ce que j'ai cette fonction au moins ?
        p.detach().zero_()
    return module


Il suffit donc au final de faire zero(module) de n'importe quel 


"""