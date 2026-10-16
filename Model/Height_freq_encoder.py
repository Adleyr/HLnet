import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features, momentum=0.1),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features, momentum=0.1)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,block=ResidualBlock,num_block=3):
        super(DownsampleBlock, self).__init__()

        self.block=block
        self.num_block=num_block
        self.in_channels=in_channels
        self.res = self.create_layer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def create_layer(self):
        branch_layer = []
        for i in range( self.num_block):
            branch_layer.append(self.block(self.in_channels))
        branch_layer = nn.Sequential(*branch_layer)
        return branch_layer


    def forward(self, x):
        middle=self.res(x)
        out = self.conv(middle)
        out = self.bn(out)
        out = self.relu(out)
        return out,middle



class Econd_b(nn.Module):
    def __init__(self,in_channels):
        """

        :type in_channels: list-->[32,64,128,256]
                      shape      128,64,32,16
        """
        super(Econd_b,self).__init__()
        self.inchannel=in_channels

        self.to_channel=ConvNet()



        self.down_1=DownsampleBlock(16,self.inchannel[0])
        self.down_2 = DownsampleBlock(self.inchannel[0], self.inchannel[1])
        self.down_3=DownsampleBlock(self.inchannel[1],in_channels[2])

        self.down_4 = DownsampleBlock(self.inchannel[2], self.inchannel[3])

        self.down_5= DownsampleBlock(self.inchannel[3],self.inchannel[4])



    def forward(self,x):
        out=[]
        x=self.to_channel(x)
        next,middle=self.down_1(x)
        out.append(middle)

        next,middle=self.down_2(next)
        out.append(middle)

        next,middle=self.down_3(next)
        out.append(middle)

        next,middle=self.down_4(next)
        out.append(middle)

        next,middle=self.down_5(next)
        out.append(middle)

        return  next,out




# img=torch.randn(1,1,256,256)
# G=Econd_b(in_channels=[32,64,128,256,512])
#
#
# next,group=G(img)
# print(next.shape)
# # print(len(group))
# print("第一个交叉：",group[0].shape)
# print("第二个交叉：",group[1].shape)
# print("第三个交叉：",group[2].shape)
# print("第四个交叉：",group[3].shape)
# print("第五个交叉：",group[4].shape)

# torch.Size([1, 256, 16, 16])
# 第一个交叉： torch.Size([1, 16, 256, 256])
# 第二个交叉： torch.Size([1, 32, 128, 128])
# 第三个交叉： torch.Size([1, 64, 64, 64])
# 第三个交叉： torch.Size([1, 128, 32, 32])