import torch
import torch.nn as nn

import torch
# import torch.nn as nn


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


# import matplotlib.pyplot as plt
# import numpy as np

# npfile=np.load(r"/Users/a1/PycharmProjects/98second/00001.npy")
# print(npfile.shape)
# img=torch.tensor(npfile).float()

# img=img.unsqueeze(0)
# img=img.unsqueeze(0)
# print(img.shape)

# gaussian_filter = GaussianBlur(kernel_size=3, sigma=0.5)

# low_freq_img = gaussian_filter(img)
# print(low_freq_img.shape)
# high_freq_img = img - low_freq_img
# print(high_freq_img.shape)


# low_freq_img = low_freq_img.detach().numpy().squeeze()
# high_freq_img = high_freq_img.detach().numpy().squeeze()

# # plot the original, low frequency, and high frequency images
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
# ax[0].imshow(img.squeeze(), cmap='gray')
# ax[0].set_title('Original image')
# ax[1].imshow(low_freq_img, cmap='gray')
# ax[1].set_title('Low frequencyimage')
# ax[2].imshow(high_freq_img, cmap='gray')
# ax[2].set_title('High frequency image')
# plt.show()
#
#
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # # create a random 256x256 grayscale image
# # img = np.random.rand(1, 1,256, 256)
# #
# # # convert the image to a PyTorch tensor
# # img_tensor = torch.from_numpy(img).float()
# #
# # # create an instance of the GaussianBlur module with kernel size 5 and sigma 1.0
# # gaussian_filter = GaussianBlur(kernel_size=5, sigma=1.0)
# #
# # # apply the Gaussian filter to the input image
# # low_freq_img = gaussian_filter(img_tensor)
# # print(low_freq_img.shape)
# # high_freq_img = img_tensor - low_freq_img
# # print(high_freq_img.shape)
# #
# #
# #
# # # compute the high frequency image by subtracting the low frequency image from the original image
# #
# #
# # # convert the low frequency and high frequency images to numpy arrays
# #
# # low_freq_img = low_freq_img.detach().numpy().squeeze()
# # high_freq_img = high_freq_img.detach().numpy().squeeze()
# #
# # # plot the original, low frequency, and high frequency images
# # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
# # ax[0].imshow(img.squeeze(), cmap='gray')
# # ax[0].set_title('Original image')
# # ax[1].imshow(low_freq_img, cmap='gray')
# # ax[1].set_title('Low frequencyimage')
# # ax[2].imshow(high_freq_img, cmap='gray')
# # ax[2].set_title('High frequency image')
# # plt.show()


# relv_1 = self.relve1(low_group[4], hight_group[4], V_list[3])
# relv_2 = self.relve2(low_group[3], hight_group[3], V_list[2])

# relv_3 = self.relve3(low_group[2], hight_group[2], V_list[1])
# relv_4 = self.relve4(low_group[1], hight_group[1], V_list[0])

# relv_1 = self.relve1(V_list[3], hight_group[4], low_group[4])
# relv_2 = self.relve2(V_list[2], hight_group[3], low_group[3])
#
# relv_3 = self.relve3(V_list[1], hight_group[2], low_group[2])
# relv_4 = self.relve4(V_list[0], hight_group[1], low_group[1])