import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NosieDivider(nn.Module):


    def __init__(self,config):
        super(NosieDivider,self).__init__()
    

    def filtering(self,x , x_abs, i,thres):
        sample = torch.where(x_abs > thres, x, 0.)
        mask = torch.where(sample != 0.,1.,0)
        noise_mask = torch.where(sample == 0.,1.,0.)
        return sample ,mask, noise_mask

    #-> [bs, fs, len]
    

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)


    def forward(self,x):
        B,C,S = x.shape
        
        out =  torch.fft.rfft(x,dim=-1) # len(x)//2 + 1
        noise = out.clone()
        x_abs = torch.abs(out)
        x_md = torch.median(x_abs,dim=-1).values

        out,mask,noise_mask = self.filtering(out,x_abs,0,x_md.unsqueeze(-1))

        # if config.single:
        noise =  noise * noise_mask
        noise = torch.fft.irfft(noise,dim=-1)
        out = torch.fft.irfft(out,dim=-1)
        return out,noise, mask, noise_mask


if __name__ =='__main__':
    config = None
    layer = NosieDivider2(config)
    x = torch.randn([2,3,10])
    out,noise, mask, noise_mask= layer(x)
    print(out.shape)
    # print(torch.masked_select(out, mask))