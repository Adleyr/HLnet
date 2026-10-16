import itertools

from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

from datasets import ImageDataset
from utils import Resize, ToTensor
from utils import Logger
from torchvision.transforms import RandomAffine, ToPILImage
import torchvision
import numpy as np

level=4
transforms_1 = [ToPILImage(),
                RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                             scale=[1 - 0.02 * level, 1 + 0.02 * level], fillcolor=-1),
                ToTensor(),
                Resize(size_tuple=(256, 256))]

transforms_2 = [ToPILImage(),
                RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                             scale=[1 - 0.02 * level, 1 + 0.02 * level], fillcolor=-1),  #

                ToTensor(),
                Resize(size_tuple=(256, 256))]



tran=torchvision.transforms.Compose(transforms_1)

tran2=torchvision.transforms.Compose(transforms_2)

img=np.load(r"D:\python_project2\pGAN\data1\T1T2\train2D\00046.npy").astype(np.float32)
img=tran(img)
img=img.unsqueeze(0)

img2=np.load(r"D:\python_project2\pGAN\data1\T1T2\val2D\00046.npy").astype(np.float32)
img2=tran2(img2)
img2=img2.unsqueeze(0)

loss_D_B=1

logger = Logger('test', 8097, 1, 2)

# self.logger.log({'loss_D_B': loss_D, },
#                 learning_rate={self.optimizerG.state_dict()['param_groups'][0]['lr']},
#                 images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})  # ,'SR':SysRegist_A2B

logger.log(losses=None,learning_rate=0.002,images={'real_A': img, 'real_B': img2, 'fake_B': img})