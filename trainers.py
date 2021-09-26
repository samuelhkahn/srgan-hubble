from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from srresnet_loss import Loss
import torch
from torch.nn.utils import clip_grad_norm_,clip_grad_value_
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

def invert_min_max_normalization(tensor:np.ndarray, min_val:float, max_val:float) -> np.ndarray:
    denominator = max_val-min_val
    unnormalized=tensor*denominator+min_val
    return unnormalized

def train_srresnet(srresnet, dataloader, device, experiment,model_name, lr=1e-4, total_steps=1e6, display_step=500 ):
    srresnet = srresnet.to(device).train()
    optimizer = torch.optim.Adam(srresnet.parameters(), lr=lr)

    # every 5000th step decrease the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1) 

    cur_step = 0
    mean_loss = 0.0

    # # HST clip range - (0,99.996)
    hst_min, hst_max = (-4.656636714935303, 36.228747035183915)

    # HSC clip range - (0,99.9)
    hsc_min, hsc_max = (-0.4692089855670929, 12.432257350922441)

    while cur_step < total_steps:
        for hst_lr,hst_hr,lr_real, seg_map_real in tqdm(dataloader, position=0):
            # Conv2d expects (n_samples, channels, height, width)
            # So add the channel dimension
            hst_lr = hst_lr.to(device)
            hst_hr = hst_hr.to(device)


            lr_real = lr_real.unsqueeze(1).to(device)
            seg_map_real = seg_map_real.to(device)


            # print("HST LR:",hst_lr.shape)
            # print("HST HR:",hst_hr.shape)
            # print("HSC LR:",lr_real.shape)
            # print("Seg Map:",seg_map_real.shape)
            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    # hr_fake = batch, channels, height, width

                    hr_fake = srresnet(lr_real)

                    ## Force Inductive Channel Bias -> 1=HST_HR, Channel 2=HST_LR
                    mse_loss_hr = Loss.img_loss(hst_hr, hr_fake[:,0,:,:])
                    mse_loss_lr = Loss.img_loss(hst_lr, hr_fake[:,1,:,:])

                    mse_loss_mask_hr = Loss.img_loss_with_mask(hst_hr, hr_fake[:,0,:,:],seg_map_real)
                    mse_loss_mask_lr = Loss.img_loss_with_mask(hst_lr, hr_fake[:,1,:,:],seg_map_real)

            else:
                hr_fake = srresnet(lr_real)

                mse_loss_hr = Loss.img_loss(hst_hr, hr_fake[:,0,:,:])
                mse_loss_lr = Loss.img_loss(hst_lr, hr_fake[:,1,:,:])

                mse_loss_mask_hr = Loss.img_loss_with_mask(hst_hr, hr_fake[:,0,:,:],seg_map_real)
                mse_loss_mask_lr = Loss.img_loss_with_mask(hst_lr, hr_fake[:,1,:,:],seg_map_real)

            loss = mse_loss_mask_hr + mse_loss_mask_lr + mse_loss_lr

            optimizer.zero_grad()
            loss.backward()
            clip_grad_value_(srresnet.parameters(), 10000)
            optimizer.step()
            scheduler.step()

            mean_loss += loss.item() / display_step

            # Log to Comet ML



            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                # lr_image = invert_min_max_normalization(lr_real[0,:,:,:].cpu(),hsc_min,hsc_max)
                # sr_image = invert_min_max_normalization(hr_fake[0,:,:,:].cpu(),hst_min,hst_max)
                # hr_image = invert_min_max_normalization(hr_real[0,:,:,:].cpu(),hst_min,hst_max)

                lr_image = lr_real[0,:,:,:].cpu()

                sr_image_hr = hr_fake[0,0,:,:].cpu()
                sr_image_lr = hr_fake[0,1,:,:].cpu()

                hst_lr_image = hst_lr[0,:,:].cpu()
                hst_hr_image = hst_hr[0,:,:].cpu()  

                seg_image = seg_map_real[0,:,:].cpu()


         
                experiment.log_image(lr_image,"Low Resolution")
                experiment.log_image(sr_image_hr,"Super Resolution - HR")
                experiment.log_image(sr_image_lr,"Super Resolution - LR")
                experiment.log_image(hst_hr_image,"High Resolution - HR")
                experiment.log_image(hst_lr_image,"High Resolution - LR")
                # experiment.log_image(seg_image,"Segmentation Map")#,image_minmax=(0,1),cmap='gray')
                # experiment.log_image(hr_image,"High Resolution")#,image_minmax=(0,1),cmap='gray')

                # img_diff_lr = (sr_image_lr - hst_lr_image).cpu()
                # img_diff_hr = (sr_image_hr - hst_hr_image).cpu()

                # experiment.log_image(img_diff_lr,"Image Difference - LR")
                # experiment.log_image(img_diff_hr,"Image Difference - HR")


                mean_loss = 0.0
            # show_tensor_images(lr_real * 2 - 1)
            # show_tensor_images(hr_fake.to(hr_real.dtype))
            # show_tensor_images(hr_real)

            experiment.log_metric("SRResNet MSE MASKED HR Loss",mse_loss_mask_hr.item()/display_step)
            experiment.log_metric("SRResNet MSE MASKED LR Loss",mse_loss_mask_lr.item()/display_step)
            experiment.log_metric("SRResNet MSE LR Loss",mse_loss_lr.item()/display_step)
            experiment.log_metric("SRResNet MSE MASKED HR/LR LossRatio ",mse_loss_mask_hr.item()/mse_loss_mask_lr.item())

            experiment.log_metric("SRResNet Total MSE Loss",mean_loss)
            experiment.log_metric("Learning Rate",optimizer.param_groups[0]['lr'])




            if cur_step%10000==0:
                torch.save(srresnet, f'srresnet_{model_name}_checkpoint_{cur_step}.pt')

            cur_step += 1
            if cur_step == total_steps:
                break
    return srresnet

def compute_gradient_penalty(discriminator, real_samples, fake_samples,device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = discriminator(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1).to(device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean().to(device)
    return gradient_penalty

def train_srgan(generator, discriminator, dataloader, device,experiment, model_name,lr=1e-4, total_steps=2e5, display_step=100,lambda_gp=1):
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

    # # HST clip range - (0,99.996)
    hst_min, hst_max = (-4.656636714935303, 36.228747035183915)

    # HSC clip range - (0,99.9)
    hsc_min, hsc_max = (-0.4692089855670929, 12.432257350922441)

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
            gradient_penalty = compute_gradient_penalty(discriminator, hr_real, hr_fake,device)

            real_disc_loss = torch.mean(discriminator(hr_real))
            fake_disc_loss = torch.mean(discriminator(hr_fake))
            gradient_penalty = lambda_gp*gradient_penalty
            d_loss = fake_disc_loss + \
                     - real_disc_loss+ \
                     gradient_penalty

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            mean_d_loss += d_loss.item() / display_step

            # hr_fake = generator(lr_real)
            # Adversarial loss
            
            if display_step %5 ==0 and display_step!=0:
                g_loss = -torch.mean(discriminator(hr_fake))
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                mean_g_loss += g_loss.item() / display_step
            # mean_vgg_loss += vgg_loss.item() / display_step


            experiment.log_metric("Generator Loss",mean_g_loss)
            experiment.log_metric("Discriminator Loss",mean_d_loss)
            experiment.log_metric("Real Disc Loss Component",real_disc_loss)
            experiment.log_metric("Fake Disc Loss Component",fake_disc_loss)
            experiment.log_metric("Gradient Penalty",gradient_penalty)

            # experiment.log_metric("VGG Loss",vgg_loss)


            # if cur_step == lr_step:
            #     g_scheduler.step()
            #     d_scheduler.step()
            #     print('Decayed learning rate by 10x.')

            if cur_step%50000==0:
                torch.save(generator, f'srgenerator_{model_name}_checkpoint_{cur_step}.pt')
                torch.save(discriminator, f'srdiscriminator_{model_name}_checkpoint_{cur_step}.pt')


            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step, mean_g_loss, mean_d_loss))
#                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                # lr_image = invert_min_max_normalization(lr_real[0,:,:,:].cpu(),hsc_min,hsc_max)
                # sr_image = invert_min_max_normalization(hr_fake[0,:,:,:].cpu(),hst_min,hst_max)
                # hr_image = invert_min_max_normalization(hr_real[0,:,:,:].cpu(),hst_min,hst_max)


                lr_image = lr_real[0,:,:,:].cpu()
                sr_image = hr_fake[0,:,:,:].cpu()
                hr_image = hr_real[0,:,:,:].cpu()  

                # lr_image = invert_min_max_normalization(lr_image,hsc_min,hsc_max)
                # sr_image = invert_min_max_normalization(sr_image,hst_min,hst_max)               
                # hr_image = invert_min_max_normalization(hr_image,hst_min,hst_max)

                experiment.log_image(lr_image,"Low Resolution")
                experiment.log_image(sr_image,"Super Resolution")
                experiment.log_image(hr_image,"High Resolution")
                img_diff = (sr_image - hr_image).cpu()

                experiment.log_image(img_diff,"Image Difference")

            mean_g_loss = 0.0
            mean_d_loss = 0.0
            vgg_loss = 0.0


            cur_step += 1
            if cur_step == total_steps:
                break

    return generator,discriminator
