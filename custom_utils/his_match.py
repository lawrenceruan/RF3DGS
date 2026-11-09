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


class DifferentiableHistogramMatchingLoss(nn.Module):
    """Squared CDF distance with differentiable linear bin assignment."""

    def __init__(self, bins=256):
        super().__init__()
        if bins < 2:
            raise ValueError("Histogram matching needs at least two bins")
        self.bins = bins

    def cdf(self, image):
        position = image.clamp(0, 1).reshape(-1) * (self.bins - 1)
        lower = position.floor().long()
        fraction = position - lower
        histogram = position.new_zeros(self.bins)
        histogram.scatter_add_(0, lower, 1 - fraction)
        histogram.scatter_add_(0, (lower + 1).clamp_max(self.bins - 1), fraction)
        return histogram.cumsum(0) / position.numel()

    def forward(self, input, target):
        return (self.cdf(input) - self.cdf(target)).square().mean()
