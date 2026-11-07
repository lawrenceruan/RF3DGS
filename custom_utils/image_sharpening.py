import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# Function to create a Gaussian kernel
def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size) - kernel_size // 2
    x = x.float()
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()  # Normalize
    kernel_1d = gauss.view(1, 1, -1)
    kernel_2d = kernel_1d @ kernel_1d.transpose(1, 2)  # Outer product to create 2D kernel
    return kernel_2d

# Unsharp masking function
def unsharp_mask(image_tensor, kernel_size=5, sigma=1.0, amount=1.0):
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma).to(image_tensor.device)
    kernel = kernel.expand(3, 1, kernel_size, kernel_size)  # Expand for channels
    
    # Apply Gaussian blur (convolution)
    blurred = F.conv2d(image_tensor.unsqueeze(0), kernel, padding=kernel_size // 2, groups=image_tensor.size(0))
    blurred = blurred.squeeze(0)
    
    # Calculate high-frequency details
    high_freq_details = image_tensor - blurred
    
    # Add high-frequency details back to the original
    sharpened = image_tensor + amount * high_freq_details
    
    # Clip values to maintain valid image range
    sharpened = torch.clamp(sharpened, 0, 1)
    return sharpened