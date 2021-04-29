from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from srresnet_loss import Loss
import torch

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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1) 

    cur_step = 0
    mean_loss = 0.0

    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            # Conv2d expects (n_samples, channels, height, width)
            # So add the channel dimension
            hr_real = hr_real.unsqueeze(1).to(device)
            lr_real = lr_real.unsqueeze(1).to(device)
            
            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    hr_fake = srresnet(lr_real)
                    loss = Loss.img_loss(hr_real, hr_fake)
            else:

                hr_fake = srresnet(lr_real)
                loss = Loss.img_loss(hr_real, hr_fake)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            mean_loss += loss.item() / display_step

            # Log to Comet ML



            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                experiment.log_image(lr_real[0,:,:,:].cpu(),"Low Resolution")
                experiment.log_image(hr_fake[0,:,:,:].cpu(),"Super Resolution")
                experiment.log_image(hr_real[0,:,:,:].cpu(),"High Resolution")
                mean_loss = 0.0
            # show_tensor_images(lr_real * 2 - 1)
            # show_tensor_images(hr_fake.to(hr_real.dtype))
            # show_tensor_images(hr_real)


            experiment.log_metric("SRResNet MSE Loss",mean_loss)
            experiment.log_metric("Learning Rate",optimizer.param_groups[0]['lr'])




            if cur_step%20000==0:
                torch.save(srresnet, f'srresnet_checkpoint_{cur_step}_no_clip.pt')

            cur_step += 1
            if cur_step == total_steps:
                break
    return srresnet

def train_srgan(generator, discriminator, dataloader, device,experiment, lr=1e-4, total_steps=2e5, display_step=500):
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
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)
            
            hr_real = hr_real.unsqueeze(1).to(device)
            lr_real = lr_real.unsqueeze(1).to(device)
            
            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss,vgg_loss, hr_fake = loss_fn(
                        generator, discriminator, hr_real, lr_real,
                    )
            else:
                g_loss, d_loss,vgg_loss, hr_fake = loss_fn(
                    generator, discriminator, hr_real, lr_real,
                )


            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step
            mean_vgg_loss += vgg_loss.item() / display_step


            experiment.log_metric("Generator Loss",mean_g_loss)
            experiment.log_metric("Discriminator Loss",mean_d_loss)
            experiment.log_metric("VGG Loss",vgg_loss)


            if cur_step == lr_step:
                g_scheduler.step()
                d_scheduler.step()
                print('Decayed learning rate by 10x.')

            if cur_step%10000==0:
                torch.save(generator, f'srgenerator_checkpoint_log_scale_{cur_step}.pt')
                torch.save(discriminator, f'srdiscriminator_checkpoint_log_scale_{cur_step}.pt')


            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step, mean_g_loss, mean_d_loss))
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                experiment.log_image(lr_real[0,:,:,:].cpu(),"Low Resolution")
                experiment.log_image(hr_fake[0,:,:,:].cpu(),"Super Resolution")
                experiment.log_image(hr_real[0,:,:,:].cpu(),"High Resolution")
                mean_g_loss = 0.0
                mean_d_loss = 0.0
                vgg_loss = 0.0


            cur_step += 1
            if cur_step == total_steps:
                break

    return generator,discriminator