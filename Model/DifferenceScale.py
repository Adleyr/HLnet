
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

