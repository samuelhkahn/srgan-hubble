from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from srresnet_loss import Loss
import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from torch import autograd
import numpy as np
# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def show_tensor_images(image_tensor):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:4], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def train_srresnet(srresnet, dataloader, device, experiment, lr=1e-4, total_steps=1e6, display_step=500):
    srresnet = srresnet.to(device).train()
    optimizer = torch.optim.Adam(srresnet.parameters(), lr=lr)

    # every 5000th step decrease the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1) 

    cur_step = 0
    mean_loss = 0.0

    while cur_step < total_steps:
        for hr_real, lr_real, hr_segs in tqdm(dataloader, position=0):
            # Conv2d expects (n_samples, channels, height, width)
            # So add the channel dimension
            hr_real = hr_real.unsqueeze(1).to(device)
            lr_real = lr_real.unsqueeze(1).to(device)
            hr_segs = hr_segs.unsqueeze(1).to(device)
            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    hr_fake = srresnet(lr_real)
                    mse_loss = Loss.l1_loss(hr_real, hr_fake)
                    mse_loss_mask = Loss.l1_loss_with_mask(hr_real, hr_fake,hr_segs)
                    # ssim_loss = Loss.ssim_loss_with_mask(hr_real, hr_fake,hr_segs)
                    # emd_loss = Loss.emd(hr_real, hr_fake,hr_segs)
                    # print(emd_loss)

                    # emd_loss = torch.mean(emd_loss)

            else:

                hr_fake = srresnet(lr_real)
                mse_loss = Loss.l1_loss(hr_real, hr_fake)
                mse_loss_mask = Loss.l1_loss_with_mask(hr_real, hr_fake,hr_segs)
            # ssim_loss = Loss.ssim_loss_with_mask(hr_real, hr_fake,hr_segs)
            # emd_loss = Loss.emd(hr_real, hr_fake,hr_segs)
            # print(emd_loss)

            # emd_loss = torch.mean(emd_loss)

            # loss=0.001*mse_loss+0.001*mse_loss_mask+ssim_loss
            loss = mse_loss+ mse_loss_mask
            # print(mse_loss)
            # print(mse_loss_mask)
            # print(ssim_loss)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(srresnet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            mean_loss += loss.item() / display_step

            # Log to Comet ML



            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                experiment.log_image(lr_real[0,:,:,:].cpu(),"Low Resolution")
                experiment.log_image(hr_fake[0,:,:,:].cpu(),"Super Resolution")
                experiment.log_image(hr_real[0,:,:,:].cpu(),"High Resolution")

                img_diff = (hr_fake[0,:,:,:] - hr_real[0,:,:,:]).cpu()

                experiment.log_image(img_diff,"Image Difference")

                mean_loss = 0.0
            # show_tensor_images(lr_real * 2 - 1)
            # show_tensor_images(hr_fake.to(hr_real.dtype))
            # show_tensor_images(hr_real)


            experiment.log_metric("SRResNet MSE Loss",mean_loss)
            experiment.log_metric("Learning Rate",optimizer.param_groups[0]['lr'])




            if cur_step%20000==0:
                torch.save(srresnet, f'srresnet_checkpoint_median_scale_{cur_step}.pt')

            cur_step += 1
            if cur_step == total_steps:
                break
    return srresnet

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_srgan(generator, discriminator, dataloader, device,experiment, lr=1e-4, total_steps=2e5, display_step=500,lambda_gp=1):
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    loss_fn = Loss(device=device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lambda _: 0.1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lambda _: 0.1)

    lr_step = total_steps // 2
    cur_step = 0

    mean_g_loss = 0.0
    mean_d_loss = 0.0
    mean_vgg_loss = 0.0

    while cur_step < total_steps:
        for hr_real, lr_real, hr_segs in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)
            
            hr_real = hr_real.unsqueeze(1).to(device)
            lr_real = lr_real.unsqueeze(1).to(device)
            hr_segs = hr_segs.unsqueeze(1).to(device)

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            # if has_autocast:
            #     with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            #         g_loss, d_loss,vgg_loss, hr_fake = loss_fn(
            #             generator, discriminator, hr_real, lr_real, hr_segs
            #         )
            # else:
            #     g_loss, d_loss,vgg_loss, hr_fake = loss_fn(
            #         generator, discriminator, hr_real, lr_real, hr_segs
            #     )

            hr_fake = generator(lr_real).detach()
            gradient_penalty = compute_gradient_penalty(discriminator, hr_real, hr_fake)
            d_loss = -torch.mean(discriminator(hr_real)) +\
                     torch.mean(discriminator(hr_fake))+ \
                     lambda_gp*gradient_penalty

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            hr_fake = generator(lr_real)
            # Adversarial loss
            g_loss = -torch.mean(discriminator(hr_fake))



            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()



            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step
            # mean_vgg_loss += vgg_loss.item() / display_step


            experiment.log_metric("Generator Loss",mean_g_loss)
            experiment.log_metric("Discriminator Loss",mean_d_loss)
            # experiment.log_metric("VGG Loss",vgg_loss)


            # if cur_step == lr_step:
            #     g_scheduler.step()
            #     d_scheduler.step()
            #     print('Decayed learning rate by 10x.')

            if cur_step%50000==0:
                torch.save(generator, f'srgenerator_checkpoint_median_scale_{cur_step}.pt')
                torch.save(discriminator, f'srdiscriminator_median_scale_{cur_step}.pt')


            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step, mean_g_loss, mean_d_loss))
#                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                experiment.log_image(lr_real[0,:,:,:].cpu(),"Low Resolution")
                experiment.log_image(hr_fake[0,:,:,:].cpu(),"Super Resolution")
                experiment.log_image(hr_real[0,:,:,:].cpu(),"High Resolution")
                img_diff = (hr_fake[0,:,:,:] - hr_real[0,:,:,:]).cpu()
                experiment.log_image(img_diff,"Image Difference")

                mean_g_loss = 0.0
                mean_d_loss = 0.0
                vgg_loss = 0.0


            cur_step += 1
            if cur_step == total_steps:
                break

    return generator,discriminator
