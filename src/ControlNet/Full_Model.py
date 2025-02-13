import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import math
from einops import rearrange

from models.model import *

from ControlNet.util import *

from ControlNet.ControledUNet import *
from ControlNet.ControlNET import *



class Full_Model(nn.Module):
    def __init__(
        

        self,
        controlledunetmodel,
        controlnet
    ):
        super().__init__()  
        
        #self.modules = nn.ModuleList([])
        self.controlledunetmodel = controlledunetmodel
        self.controlnet = controlnet

        #self.modules.append(
         #       nn.ModuleList(
             #       [
          #              controlledunetmodel,
           #             controlnet
            #        ]
             #   )
            #)
        
        

    def forward(self, x, hint, time):
        control = self.controlnet(x,hint,time)
        x = self.controlledunetmodel(x, time, control)
        return x