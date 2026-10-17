
import torch.nn as nn

import torch



class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # create a 1D Gaussian kernel
        kernel_1D = torch.exp(-torch.arange(self.kernel_size).float() ** 2 / (2 * self.sigma ** 2))
        kernel_1D = kernel_1D / torch.sum(kernel_1D)

        # create a 2D Gaussian kernel by outer product of 1D kernel
        kernel_2D = torch.mm(kernel_1D.unsqueeze(1), kernel_1D.unsqueeze(0))

        # create a convolutional layer with the Gaussian kernel as weights
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, bias=False)

        # Set the weight data and disable gradient computation
        self.conv.weight.data[0, 0, :, :] = kernel_2D
        self.conv.weight.requires_grad = False  # Disable gradient computation for the convolutional layer weights
        for p in self.conv.parameters():
            p.requires_grad = False
    def forward(self, x):
        # apply the Gaussian filter to the input image
        return self.conv(x)

