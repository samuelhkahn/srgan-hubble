from torchvision.models import vgg19
import torch.nn as nn
import torch.nn.functional as F
import torch

from neuralnet_pytorch.metrics import emd_loss
from neuralnet_pytorch.metrics import ssim

class Loss(nn.Module):
    '''
    Loss Class
    Implements composite content+adversarial loss for SRGAN
    Values:
        device: 'cuda' or 'cpu' hardware to put VGG network on, a string
    '''

    def __init__(self, device='cuda'):
        super().__init__()

        vgg = vgg19(pretrained=True).to(device)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    @staticmethod
    def img_loss(x_real, x_fake):
        return F.mse_loss(x_real, x_fake)

    @staticmethod
    def img_loss_with_mask(x_real, x_fake,seg_map_real):
        return torch.sum(((x_real-x_fake)*seg_map_real)**2.0)/torch.sum(seg_map_real)


    @staticmethod
    def ssim_loss_with_mask(x_real, x_fake,seg_map_real):

        # seg_map_real = seg_map_real.squeeze(1)
        # x_real = x_real*seg_map_real
        # x_fake = x_fake*seg_map_real
        return ssim(x_real,x_fake)
    @staticmethod
    def emd(x_real, x_fake,seg_map_real):

        seg_map_real = seg_map_real.squeeze(1)
        
        x_real = x_real*seg_map_real
        x_fake = x_fake*seg_map_real

        x_real = x_real.squeeze(1)
        x_fake = x_fake.squeeze(1)

        return emd_loss(x_real,x_fake,sinkhorn=True)

    def adv_loss(self, x, is_real):
        # If fake we want "convince" that it is real
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)

    def vgg_loss(self, x_real, x_fake):
        #Copy across channel diension because VGG expects 3 channels
        x_real = torch.repeat_interleave(x_real, 3, dim=1)
        x_fake = torch.repeat_interleave(x_fake, 3, dim=1)
        return F.mse_loss(self.vgg(x_real), self.vgg(x_fake))

    def forward(self, generator, discriminator, hr_real, lr_real,seg_map_real):
        ''' Performs forward pass and returns total losses for G and D '''
        hr_fake = generator(lr_real)
        fake_preds_for_g = discriminator(hr_fake)
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())

        vgg_loss = 0.006 * self.vgg_loss(hr_real, hr_fake)

        img_loss_with_mask = self.img_loss_with_mask(hr_real, hr_fake,seg_map_real)
        g_loss = (
            0.1 * self.adv_loss(fake_preds_for_g, False) + \
           # vgg_loss + \
             img_loss_with_mask + \
            self.img_loss(hr_real, hr_fake)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )

        return g_loss, d_loss,vgg_loss, hr_fake
