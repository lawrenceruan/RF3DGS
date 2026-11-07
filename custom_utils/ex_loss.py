import torch
import torch.nn as nn

def sample_pixels(image, num_samples):
    """
    Randomly sample a set of pixels from an image.
    
    Args:
        image (torch.Tensor): The input image with shape [C, H, W].
        num_samples (int): The number of pixels to sample.
        
    Returns:
        torch.Tensor: A tensor of sampled pixel values with shape [num_samples, C].
    """
    # Get the image dimensions
    C, H, W = image.shape
    
    # Generate random row and column indices
    row_idx = torch.randint(0, H, (num_samples,))
    col_idx = torch.randint(0, W, (num_samples,))
    
    # Extract the sampled pixel values
    sampled_pixels = image[:, row_idx, col_idx]
    
    return sampled_pixels

class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):

        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
            grady1_s = (torch.sigmoid(grady1) * 2) - 1
            gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)

            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            # img1 = self.avg_pool(img1)
            # img2 = self.avg_pool(img2)

        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):        
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

class ExclusionLoss_sample(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss_sample, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2, num_samples=500):

        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
            grady1_s = (torch.sigmoid(grady1) * 2) - 1
            gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1

            # Get the image dimensions
            B, C, H, W = gradx1_s.shape
            
            # Generate random row and column indices
            row_idx = torch.randint(0, H, (num_samples,))
            col_idx = torch.randint(0, W, (num_samples,))

            gradx_loss += self._all_comb(gradx1_s[:, :, row_idx, :][:, :, :, col_idx], gradx2_s[:, :, row_idx, :][:, :, :, col_idx])
            grady_loss += self._all_comb(grady1_s[:, :, row_idx, :][:, :, :, col_idx], grady2_s[:, :, row_idx, :][:, :, :, col_idx])

        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):        
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady