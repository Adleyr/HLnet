import torch.nn as nn
import torch
import torch.nn.functional as F
from .DifferenceScale import DownSampleModule
from .Gaussi import GaussianBlur
from .Height_freq_encoder import Econd_b
from .Low_freq_encoder import Econd_a
from .Bottom_share import ShareNet
from .Decoder import Swin_Decoder,Swin_DecoderL
from .Special_transformer import Transformer
class HLnet(nn.Module):
    def __init__(self,inc,depths,num_heads):
        super(HLnet,self).__init__()
        self.inc=inc   ##[32,]
        self.depths=depths
        self.num_heads=num_heads

        self.toC=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.outLayer=nn.Sequential(*[nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,padding=1),
                                      nn.InstanceNorm2d(8),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=8,out_channels=1,kernel_size=3,stride=1,padding=1),
                                      nn.Tanh()])

        self.gaussi=GaussianBlur(sigma=1)

        self.Hencoder=Econd_b(in_channels=self.inc)
        self.Lencoder=Econd_a(in_channels=self.inc)

        self.bottom_share=ShareNet()

        self.relve1 = Transformer(256)
        self.relve2 = Transformer(128)
        self.relve3 = Transformer(64)
        self.relve4 = Transformer(32)

        self.differenceS=DownSampleModule()

        self.up_1=Swin_Decoder(512,self.depths,self.num_heads)

        self.up_2=Swin_Decoder(256,self.depths,self.num_heads)

        self.up_3 = Swin_Decoder(128, self.depths, self.num_heads)

        self.up_4 = Swin_Decoder(64, self.depths, self.num_heads)

        self.up_5= Swin_DecoderL(32,self.depths,self.num_heads)


    def forward(self,x):
        toc=self.toC(x)
        V_list=self.differenceS(toc)

        low_freq_img=self.gaussi(x)
        high_freq_img=x-low_freq_img

        high_next, hight_group=self.Hencoder(high_freq_img)

        low_next, low_group =self.Lencoder(low_freq_img)

        bottom=self.bottom_share(high_next,low_next)

        relv_1 = self.relve1(low_group[4],  V_list[3], hight_group[4])
        relv_2 = self.relve2(low_group[3],  V_list[2], hight_group[3])

        relv_3 = self.relve3(low_group[2],  V_list[1], hight_group[2])
        relv_4 = self.relve4(low_group[1],  V_list[0], hight_group[1])

        out, bridge=self.up_1(bottom,relv_1)
        out, bridge=self.up_2(out, relv_2, bridge)
        out, bridge=self.up_3(out, relv_3, bridge)
        out, bridge=self.up_4(out,  relv_4,bridge)
        out=self.up_5(out, hight_group[0]+ low_group[0]) + bridge
        out=self.outLayer(out)

        return out


def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    para_size = para_num * 4 / 1024 / 1024
    return para_size
    

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def create_sobel_layer():
    grayscale = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    grayscale.weight.data.fill_(1.0 / 2.2)
    grayscale.bias.data.zero_()
    sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
    sobel_filter.weight.data[0, 0].copy_(
        torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    )
    sobel_filter.weight.data[1, 0].copy_(
        torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    )
    sobel_filter.bias.data.zero_()
    sobel = nn.Sequential(grayscale, sobel_filter)
    for p in sobel.parameters():
        p.requires_grad = False


    return sobel.type(torch.cuda.FloatTensor)



