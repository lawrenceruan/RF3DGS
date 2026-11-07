import torch
import numpy as np
import torch.nn as nn

class ImageGradient(nn.Module):
    def __init__(self):
        super(ImageGradient, self).__init__()

        a = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)

        b = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float(), requires_grad=False)

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, input):
        G_x = self.conv1(input)
        G_y = self.conv2(input)
        
        return torch.cat([G_x, G_y], dim=0)