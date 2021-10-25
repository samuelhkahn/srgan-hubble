from torchvision.models import vgg19
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms

# from neuralnet_pytorch.metrics import emd_loss
# from neuralnet_pytorch.metrics import ssim

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

        self.preprocess = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

    @staticmethod
    def img_loss(x_real, x_fake):
        return F.mse_loss(x_real, x_fake)

    @staticmethod
    def img_loss_with_mask(x_real, x_fake,seg_map_real):
        return torch.sum(((x_real-x_fake)*seg_map_real)**2.0)/torch.sum(seg_map_real)
    @staticmethod
    def l1_loss(x_real, x_fake):
        return F.l1_loss(x_real, x_fake)

    @staticmethod
    def l1_loss_with_mask(x_real, x_fake,seg_map_real):
        return torch.sum(((torch.abs(x_real-x_fake))*seg_map_real))/torch.sum(seg_map_real)

    @staticmethod
    def ssim_loss_with_mask(x_real, x_fake,seg_map_real):

        seg_map_real = seg_map_real.squeeze(1)

        x_real = x_real*seg_map_real
        x_fake = x_fake*seg_map_real

        x_real = torch.clip(x_real,0.0001)
        x_fake = torch.clip(x_fake,0.0001)

        return 1-ssim(x_real,x_fake)
    @staticmethod
    def emd(x_real, x_fake,seg_map_real):

        seg_map_real = seg_map_real.squeeze(1)

        x_real = x_real*seg_map_real
        x_fake = x_fake*seg_map_real

        x_real = x_real.squeeze(1)
        x_fake = x_fake.squeeze(1)

        x_real = torch.clip(x_real,0.0001)
        x_real = torch.clip(x_real,0.0001)

        return emd_loss(x_real,x_fake,sinkhorn=True)
    @staticmethod
    def tensor_zero_one_transform(tensor):
        tensor = (tensor+1)/2
        return tensor
    def adv_loss(self, x, is_real):
        # If fake we want "convince" that it is real
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)

    def adv_wasserstein_gen_loss(self, fake_preds_for_d):
        # loss = -torch.mean(fake_preds_for_d)
        return -torch.mean(fake_preds_for_d)

    def adv_wasserstein_disc_loss(self, real_preds_for_d,fake_preds_for_d):
        # loss = -torch.mean(real_preds_for_d) + torch.mean(fake_preds_for_d)
        return -torch.mean(real_preds_for_d) + torch.mean(fake_preds_for_d)       

    def vgg_loss(self, x_real, x_fake,zero_one_transform = True,mean_std_transform = True):
        if zero_one_transform == True:
            x_real = self.tensor_zero_one_transform(x_real)
            x_fake = self.tensor_zero_one_transform(x_fake)


        #Copy across channel diension because VGG expects 3 channels
        x_real = torch.repeat_interleave(x_real, 3, dim=1)
        x_fake = torch.repeat_interleave(x_fake, 3, dim=1)

        if mean_std_transform== True:
            x_real = self.preprocess(x_real)
            x_fake = self.preprocess(x_fake)


        return F.mse_loss(self.vgg(x_real), self.vgg(x_fake))



    def forward(self, generator, discriminator, hr_real, lr_real,seg_map_real):
        ''' Performs forward pass and returns total losses for G and D '''
        hr_fake = generator(lr_real)

        fake_preds_for_g = discriminator(hr_fake)
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())

        # vgg_loss = 0.006 * self.vgg_loss(hr_real, hr_fake)

        img_loss_with_mask = self.img_loss_with_mask(hr_real, hr_fake,seg_map_real)

        g_loss =  self.adv_wasserstein_gen_loss(fake_preds_for_g)

        d_loss = 0.2*self.adv_wasserstein_disc_loss(real_preds_for_d,fake_preds_for_d)
        print(g_loss)
        print(d_loss)

        vgg_loss = 0
        # g_loss = (
        #     self.adv_loss(fake_preds_for_g, False) + \
        #    # vgg_loss + \
        #      img_loss_with_mask + \
        #     self.img_loss(hr_real, hr_fake)
        # )
        # # d_loss = 0.5 * (
        # #     self.adv_loss(real_preds_for_d, True) + \
        # #     self.adv_loss(fake_preds_for_d, False)
        # # )

        return g_loss, d_loss,vgg_loss, hr_fake
