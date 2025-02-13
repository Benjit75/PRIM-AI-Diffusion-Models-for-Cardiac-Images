from models.model import exists

import torch as torch

from models.model import *



"""class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
                
            h = self.middle_block(h, emb, context)

            


            





        if control is not None:
            h += control.pop()

    
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)
"""

    
class ControlledUnetModel(Unet):

        def forward(self, x, time, control=None):
            x = self.init_conv(x)

            t = self.time_mlp(time) if exists(self.time_mlp) else None

            h = []

            # downsample
            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

            # bottleneck
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            if control is not None:
                x+= control.pop()

            # upsample
            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()+control.pop()), dim=1)
                x = block1(x, t)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)

            return self.final_conv(x)    