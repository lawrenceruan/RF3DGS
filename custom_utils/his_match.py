import torch
import torch.nn as nn

class HistogramMatchingLoss(nn.Module):
    def __init__(self):
        super(HistogramMatchingLoss, self).__init__()

    def forward(self, input, target):
        """
        Compute the histogram matching loss between input and target images.
        
        Parameters:
        input (torch.Tensor): The input image tensor of shape (N, C, H, W).
        target (torch.Tensor): The target image tensor of shape (N, C, H, W).
        
        Returns:
        torch.Tensor: The computed histogram matching loss.
        """
        # Ensure input is in the range [0, 1]
        input = torch.clamp(input, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Flatten the images to 1D arrays
        input_flat = input.reshape(input.size(0), -1)  # Shape (N, H*W*C)
        target_flat = target.reshape(target.size(0), -1)  # Shape (N, H*W*C)

        # Calculate the histogram and cumulative distribution function (CDF)
        input_hist = torch.histc(input_flat, bins=256, min=0, max=1)  # Shape (256,)
        target_hist = torch.histc(target_flat, bins=256, min=0, max=1)  # Shape (256,)

        # Calculate the cumulative distribution function (CDF)
        input_cdf = torch.cumsum(input_hist, dim=0)  # Shape (256,)
        target_cdf = torch.cumsum(target_hist, dim=0)  # Shape (256,)

        # Normalize the CDFs
        input_cdf = input_cdf / input_cdf[-1]  # Normalize
        target_cdf = target_cdf / target_cdf[-1]  # Normalize

        # Compute the histogram matching loss
        loss = torch.mean((input_cdf - target_cdf) ** 2)

        return loss
