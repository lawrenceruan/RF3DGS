"""Evaluate the existing sampled SSIM objective only at its sampled centers."""
import torch
import torch.nn.functional as F
from utils.loss_utils import create_window, ssim_sample


def ssim_sample_patches(img1, img2, window_size=11, size_average=True, num_samples=100):
    if not size_average or img1.ndim!=3 or window_size%2!=1:
        return ssim_sample(img1,img2,window_size,size_average,num_samples)
    c,h,w=img1.shape
    # Same CPU draws, count and order as sample_pixels in the reference loss.
    row=torch.randint(0,h,(num_samples,)).to(img1.device)
    col=torch.randint(0,w,(num_samples,)).to(img1.device)
    delta=torch.arange(window_size,device=img1.device)-window_size//2
    y=row[:,None,None]+delta[None,:,None]
    x=col[:,None,None]+delta[None,None,:]
    valid=((y>=0)&(y<h)&(x>=0)&(x<w))[None]
    def patches(image):
        return (image[:,y.clamp(0,h-1),x.clamp(0,w-1)]*valid).permute(1,0,2,3)
    a,b=patches(img1),patches(img2)
    window=create_window(window_size,c).to(device=img1.device,dtype=img1.dtype)
    conv=lambda value:F.conv2d(value,window,groups=c)
    ma,mb=conv(a),conv(b)
    aa,bb=ma.square(),mb.square();ab=ma*mb
    va,vb,cov=conv(a*a)-aa,conv(b*b)-bb,conv(a*b)-ab
    value=((2*ab+.01**2)*(2*cov+.03**2))/((aa+bb+.01**2)*(va+vb+.03**2))
    return value.mean()
