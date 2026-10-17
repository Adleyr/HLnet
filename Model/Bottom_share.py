



import torch.nn as nn
import torch



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



class ShareNet(nn.Module):
    def __init__(self,num_blocks=3,block=ResidualBlock):
        super(ShareNet, self).__init__()
        self.num_blocks=num_blocks
        self.block=block
        self.resnet = self.make_layers()

        self.channel_pres_1=nn.Conv2d(1024,512,1,1)

        self.channel_pres_2=nn.Sequential(nn.Conv2d(1024,512,3,1,1),
                                              nn.Conv2d(512,512,3,1,1))

    def make_layers(self):
        branch_layer = []
        for i in range(self.num_blocks):
            branch_layer.append(self.block(512))
        branch_layer = nn.Sequential(*branch_layer)
        return branch_layer

    def forward(self, x1,x2):

        flag=torch.cat((x1,x2),dim=1)

        x_1=self.channel_pres_1(flag)
        x_1_ =self.channel_pres_2(flag)

        bridge=x_1 + x_1_
        x=self.resnet(bridge) + bridge


        return x

