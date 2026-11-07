import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

##################################################

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

#Initialize VGG16 with pretrained weight on ImageNet
def vgg_init():
    vgg_model = torchvision.models.vgg16(pretrained = True).cuda()
    trainable(vgg_model, False)
    return vgg_model

#Extract features from internal layers for perceptual loss
class vgg(nn.Module):
    def __init__(self, vgg_model):
        super(vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            # '1': "relu1_1",
            # '3': "relu1_2",
            # '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                return x

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = vgg(vgg_init()).cuda()
        self.criterion = nn.MSELoss()
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        # x = self.normalize(x)
        # y = self.normalize(y)
        x_output, y_output = self.vgg(x), self.vgg(y)
        loss = (x_output-y_output)**2
        return loss.mean()
    
class VGGLoss_min(nn.Module):
    def __init__(self):
        super(VGGLoss_min, self).__init__()        
        self.vgg = vgg(vgg_init()).cuda()
        self.criterion = nn.MSELoss()
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x, y, z):
        
        x_output, y_output, z_output = self.vgg(x), self.vgg(y), self.vgg(z)
        loss_gt = (x_output-y_output)**2
        loss_gt_pseudo = (x_output-z_output)**2
        loss = torch.minimum(loss_gt, loss_gt_pseudo)
        return loss.mean()