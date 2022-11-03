import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class NosieDivider(nn.Module):


    def __init__(self,config):
        super(NosieDivider,self).__init__()
    

    def filtering(self,x , x_abs, i,thres):
        sample = torch.where(x_abs > thres, x, 0.)
        mask = torch.where(sample != 0.,0.,1.)
        noise_mask = torch.where(sample == 0.,1.,0.)
        return sample ,mask, noise_mask

     #-> [bs, fs, len]
    

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)


    def forward(self,x):

        out =  torch.fft.rfft(x,dim=-1) # len(x)//2 + 1
        noise = out.clone()
        x_abs = torch.abs(out).squeeze()
        x_md = torch.median(x_abs,dim=-1).values.squeeze(0)

        # if config.single:
        out[0,0],mask,noise_mask = self.filtering(out.squeeze(0),x_abs,0,x_md)
        noise[0,0] =  noise[0,0] * noise_mask
        noise = torch.fft.irfft(noise)
        out = torch.fft.irfft(out)
        return out,noise, mask, noise_mask

if __name__ =='__main__':
    config = None
    layer = NosieDivider(config)
    x = torch.randn([1,1,10])
    out, mask = layer(x)
    print(out)
    print(mask)
    exit()
    mask = mask.expand_as(out)
    print(torch.index_select(mask,dim=-1))
    print(out[mask])
    # print(torch.masked_select(out, mask))