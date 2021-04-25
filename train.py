import os
from comet_ml import Experiment
import torch
from srresnet_generator import Generator
from discriminator import Discriminator
from trainers import train_srresnet,train_srgan
from dataset import SR_HST_HSC_Dataset

def collate_fn(batch):
    hrs, lrs = [], []

    for hr, lr in batch:
        if hr.shape == (600,600) and lr.shape == (100,100):
            hrs.append(hr)
            lrs.append(lr)
    return torch.stack(hrs, dim=0), torch.stack(lrs, dim=0)

def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	generator = Generator(n_res_blocks=16, n_ps_blocks=2)

	hst_path = "../data/samples/hst/filtered"
	hsc_path = "../data/samples/hsc/filtered"

	
	api_key = os.environ['COMET_ML_ASTRO_API_KEY']
	# Create an experiment with your api key


	experiment = Experiment(
	    api_key=api_key,
	    project_name="Super Resolution GAN: HSC->HST",
	    workspace="samkahn-astro",
	)
	dataloader = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[600, 600], lr_size=[100, 100]), 
	    batch_size=6, pin_memory=True, shuffle=True, collate_fn = collate_fn
	)

	generator = train_srresnet(generator, dataloader, device, experiment, lr=1e-4, total_steps=5e4, display_step=50)

	torch.save(generator, 'srresnet_no_vgg.pt')

	generator = torch.load('srresnet_no_vgg.pt')
	discriminator = Discriminator(n_blocks=1, base_channels=8)

	generator,discriminator = train_srgan(generator, discriminator, dataloader, device, experiment, lr=1e-4, total_steps=1e5, display_step=1000)
	torch.save(generator, 'srgenerator_no_vgg.pt')
	torch.save(discriminator, 'srdiscriminator_no_vgg.pt')


if __name__=="__main__":
    main()




