import os
from comet_ml import Experiment
import torch
from srresnet_generator import Generator
from discriminator import Discriminator
from trainers import train_srresnet,train_srgan
from dataset import SR_HST_HSC_Dataset
import numpy as np
import configparser 
def collate_fn(batch):
	hrs, lrs = [], []


	for hr, lr in batch:
		hr_nan = torch.isnan(hr).any()
		lr_nan = torch.isnan(lr).any()
		hr_inf = torch.isinf(hr).any()
		lr_inf = torch.isinf(lr).any()
		good_vals = [hr_nan,lr_nan,hr_inf,lr_inf]
		if hr.shape == (600,600) and lr.shape == (100,100) and True not in good_vals:
			hrs.append(hr)
			lrs.append(lr)
	return torch.stack(hrs, dim=0), torch.stack(lrs, dim=0)


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load file paths from config
	config = configparser.ConfigParser()
	config.read('paths.config')
	hst_path = config["DEFAULT"]["hst_path"]
	hsc_path = config["DEFAULT"]["hsc_path"]
	
	# Adding Comet Logging
	api_key = os.environ['COMET_ML_ASTRO_API_KEY']
	experiment = Experiment(
	    api_key=api_key,
	    project_name="Super Resolution GAN: HSC->HST",
	    workspace="samkahn-astro",
	)
	experiment.add_tag("median_scaling - grad clipping - lr 1e-6 - PReLU final layer ")

	# Create Dataloader
	dataloader = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[600, 600], lr_size=[100, 100], transform_type = "median_scale"), 
	    batch_size=1, pin_memory=True, shuffle=True, collate_fn = collate_fn
	)

	# Define Generator
	generator = Generator(n_res_blocks=16, n_ps_blocks=2)

	# Pretrain 
	generator = train_srresnet(generator, dataloader, device, experiment, lr=1e-6, total_steps=1, display_step=50)

	torch.save(generator, 'srresnet_median_scale.pt')

	generator = torch.load('srresnet_median_scale.pt')
	discriminator = Discriminator(n_blocks=1, base_channels=8)

	generator,discriminator = train_srgan(generator, discriminator, dataloader, device, experiment, lr=1e-6, total_steps=1, display_step=1000)
	
	torch.save(generator, 'srresnet_median_scale.pt')
	torch.save(discriminator, 'srdiscriminator_median_scale.pt')


if __name__=="__main__":
    main()
