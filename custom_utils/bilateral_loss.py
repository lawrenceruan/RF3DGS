import torch
import torch.nn as nn

class BilateralLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / self.gamma)
    
    def forward(self, inputs, weights, gt_image=None):
        # [1, H, W, C]
        if gt_image is not None:
            w1 = self.bilateral_filter(weights[:,:,:-1] - weights[:,:,1:]) * gt_image[:,:,:-1]
            w2 = self.bilateral_filter(weights[:,:-1,:] - weights[:,1:,:]) * gt_image[:,:-1,:]
            w3 = self.bilateral_filter(weights[:,:-1,:-1] - weights[:,1:,1:]) * gt_image[:,:-1,:-1]
            w4 = self.bilateral_filter(weights[:,1:,:-1] - weights[:,:-1,1:]) * gt_image[:,1:,:-1]
        else:
            w1 = self.bilateral_filter(weights[:,:,:-1] - weights[:,:,1:])
            w2 = self.bilateral_filter(weights[:,:-1,:] - weights[:,1:,:])
            w3 = self.bilateral_filter(weights[:,:-1,:-1] - weights[:,1:,1:])
            w4 = self.bilateral_filter(weights[:,1:,:-1] - weights[:,:-1,1:])

        L1 = self.loss(w1 * (inputs[:,:,:-1] - inputs[:,:,1:]))
        L2 = self.loss(w2 * (inputs[:,:-1,:] - inputs[:,1:,:]))
        L3 = self.loss(w3 * (inputs[:,:-1,:-1] - inputs[:,1:,1:]))
        L4 = self.loss(w4 * (inputs[:,1:,:-1] - inputs[:,:-1,1:]))
        return (L1 + L2 + L3 + L4) / 4  