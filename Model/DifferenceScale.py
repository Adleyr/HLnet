
import torch

import torch
import torch.nn as nn

class downbl(nn.Module):
    def __init__(self,inc,outc):
        super(downbl,self).__init__()
        self.features=nn.Sequential(nn.Conv2d(inc,outc,kernel_size=3,stride=2,padding=1),
                                    nn.Conv2d(outc,outc,kernel_size=3,stride=1,padding=1),
                                    nn.InstanceNorm2d(outc),
                                    nn.ReLU(inplace=True))
    def forward(self,x):

        return self.features(x)

class DownSampleModule(nn.Module):
    def __init__(self):
        super(DownSampleModule, self).__init__()


        self.dow1=downbl(16,32)
        self.dow2 = downbl(32, 64)
        self.dow3 = downbl(64, 128)
        self.dow4 = downbl(128, 256)



    def forward(self, x):
        outputs = []
        x=self.dow1(x)
        outputs.append(x)

        x=self.dow2(x)
        outputs.append(x)

        x=self.dow3(x)
        outputs.append(x)

        x=self.dow4(x)
        outputs.append(x)

        return outputs

# 测试代码
# input_image = torch.randn(1, 16, 256, 256)  # 输入图像为(1, 1, 256, 256)
# downsample_module = DownSampleModule()
# outputs = downsample_module(input_image)
# print(len(outputs))
# 
# # 输出每次下采样后的尺寸
# for i, output in enumerate(outputs):
#     print(f"Downsample {i + 1} shape: {output.shape}")



# Downsample 1 shape: torch.Size([1, 32, 128, 128])
# Downsample 2 shape: torch.Size([1, 64, 64, 64])
# Downsample 3 shape: torch.Size([1, 128, 32, 32])
# Downsample 4 shape: torch.Size([1, 256, 16, 16])
