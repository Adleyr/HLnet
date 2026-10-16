import itertools

from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR, Logger, ReplayBuffer

from .datasets import ImageDataset, ValDataset
from .utils import Resize, ToTensor, smooothing_loss
from .utils import Logger
from torchvision.transforms import RandomAffine, ToPILImage


from skimage import metrics
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, ExponentialLR

from Model.net import HLnet, create_sobel_layer
import torch

from Model.networks import *


# /root/workspace/lxj/test3/pGAN/Model
class Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.netG_A2B = HLnet(inc=[32, 64, 128, 256, 512], depths=2, num_heads=16).cuda()

        self.bestmae = 1
        self.bestpsnr = 30
        self.bestssim = 0.80
        self.netD_B = define_D(2, 64, 'basic', 'vit', 256,
                               4, 'instance', False, 'normal', [0])




        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.fake_AB_pool = ImagePool(4)

        self.criterionL1 = torch.nn.L1Loss()
        self.sobel = create_sobel_layer()

        self.optimizerG = torch.optim.Adam(self.netG_A2B.parameters(), lr=self.config['lr_G'], betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.netD_B.parameters(), lr=self.config['lr_D'], betas=(0.5, 0.999))

        Tensor = torch.cuda.FloatTensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])

        self.n_generator_updates = 5
        # Dataset loader
        level = config['noise_level']  # set noise level

        transforms_1 = [ToPILImage(),

                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [ToPILImage(),

                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs'], len(self.dataloader))

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizerG,
                                                                lr_lambda=LambdaLR(self.config['n_epochs'], 0,
                                                                                   30).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizerD,
                                                                lr_lambda=LambdaLR(self.config['n_epochs'], 0,
                                                                                   50).step)
        # 50).step)

    def train(self):
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):

                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))


                fake_B = self.netG_A2B(real_A)

                self.optimizerD.zero_grad()
                    ##fake
                fake_AB = self.fake_AB_pool.query(torch.cat((real_A, fake_B), 1).data)
                pred_fake = self.netD_B(fake_AB.detach())
                loss_D_fake = self.criterionGAN(pred_fake, False)

                    ##real
                real_AB = torch.cat((real_A, real_B), 1)
                pred_real = self.netD_B(real_AB)

                loss_D_real = self.criterionGAN(pred_real, True)

                loss_GP = self.calculate_gradient_penalty(self.netD_B, fake_AB.detach())

                loss_D = (loss_D_fake + loss_D_real) * self.config['lambda_adv'] + loss_GP


                loss_D.backward()
                self.optimizerD.step()

                    ####  G
                self.optimizerG.zero_grad()

                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = self.netD_B(fake_AB)
                loss_G_GAN = self.criterionGAN(pred_fake, True) * self.config['lambda_A']
                loss_G_L1 = self.criterionL1(fake_B, real_B) * self.config['lambda_A']


                real_e = self.sobel(real_B)
                fake_e = self.sobel(fake_B)
                loss_e = self.criterionL1(fake_e, real_e) * 4
                    ###

                loss_G = loss_G_GAN + loss_G_L1 + loss_e
                loss_G.backward()
                self.optimizerG.step()



                self.logger.log({'loss_D_B': loss_D, },
                                learning_rate={self.optimizerG.state_dict()['param_groups'][0]['lr']},
                                images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})  # ,'SR':SysRegist_A2B

            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')

            torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'netD_B.pth')

            # ------------------------------------------
            #############val###############
            with torch.no_grad():
                MAE = 0
                SSIM = 0
                PSNR = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B, real_B)
                    psnr = self.PSNR(fake_B, real_B)

                    ssim = metrics.structural_similarity(fake_B, real_B)
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim

                    num += 1

                print('Val MAE:', MAE / num)
                if (MAE / num) < self.bestmae:
                    torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'bestnetG_A2B.pth')
                    self.bestmae = MAE / num

                if (PSNR / num) > self.bestpsnr:
                    torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'psnrbestnetG_A2B.pth')
                    self.bestpsnr = PSNR / num

                if (SSIM / num) > self.bestssim:
                    torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'ssimbestnetG_A2B.pth')
                    self.bestssim = SSIM / num

    def calculate_gradient_penalty(self, discriminator, combined_data, lambda_gp=0.01):
        batch_size = 1
        alpha = torch.rand(batch_size, 1, 1, 1).cuda()
        interpolates = alpha * combined_data + (1 - alpha) * combined_data
        interpolates.requires_grad = True

        d_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return lambda_gp * gradient_penalty

    def test(self):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'bestnetG_A2B.pth'))
        # self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
            MAE = 0
            PSNR = 0
            SSIM = 0
            num = 0
            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()

                fake_B = self.netG_A2B(real_A)
                fake_B = fake_B.detach().cpu().numpy().squeeze()
                mae = self.MAE(fake_B, real_B)
                psnr = self.PSNR(fake_B, real_B)

                ssim = metrics.structural_similarity(fake_B, real_B)
                MAE += mae
                PSNR += psnr
                SSIM += ssim
                num += 1
            print('MAE:', MAE / num)
            print('PSNR:', PSNR / num)
            print('SSIM:', SSIM / num)

    def PSNR(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mse = np.mean(((fake[x][y] + 1) / 2. - (real[x][y] + 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def MAE(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mae = np.abs(fake[x, y] - real[x, y]).mean()
        return mae / 2  # from (-1,1) normaliz  to (0,1)
