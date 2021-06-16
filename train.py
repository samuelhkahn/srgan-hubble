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
	hrs, lrs, hr_segs = [], [], []


	for hr, lr,hr_seg in batch:
		hr_nan = torch.isnan(hr).any()
		lr_nan = torch.isnan(lr).any()
		hr_inf = torch.isinf(hr).any()
		lr_inf = torch.isinf(lr).any()
		good_vals = [hr_nan,lr_nan,hr_inf,lr_inf]
		if hr.shape == (600,600) and lr.shape == (100,100) and True not in good_vals:
			hrs.append(hr)
			lrs.append(lr)
			hr_segs.append(hr_seg)
	hrs = torch.stack(hrs, dim=0)
	lrs = torch.stack(lrs, dim=0)
	hr_segs = torch.stack(hr_segs, dim=0)
	return hrs, lrs, hr_segs


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load file paths from config
	config = configparser.ConfigParser()
	config.read('paths.config')

	# Configuration Information
	hst_path = config["DEFAULT"]["hst_path"]
	hsc_path = config["DEFAULT"]["hsc_path"]
	model_name = config["MODEL_NAME"]["model_name"]

	comet_tag = config["COMET_TAG"]["comet_tag"]

	batch_size = int(config["BATCH_SIZE"]["batch_size"])
	srresnet_steps = int(config["SRRESENET_STEPS"]["srresnet_steps"])
	gan_steps = int(config["GAN_STEPS"]["gan_steps"])


	# Adding Comet Logging
	api_key = os.environ['COMET_ML_ASTRO_API_KEY']

	experiment = Experiment(
	    api_key=api_key,
	    project_name="Super Resolution GAN: HSC->HST",
	    workspace="samkahn-astro",
	)

	experiment.add_tag(comet_tag)


	# Create Dataloader
	dataloader = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[600, 600], 
	    	lr_size=[100, 100], transform_type = "global_median_scale",data_aug = False), 
	    batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn
	)

	# Define Generator
	generator = Generator(n_res_blocks=16, n_ps_blocks=2,pix_shuffle=True)

	# Pretrain 
	generator = train_srresnet(generator, dataloader, device, experiment, lr=1e-6, total_steps=srresnet_steps, display_step=250)

	torch.save(generator, f'srresnet_{model_name}.pt')

	generator = torch.load(f'srresnet_{model_name}.pt')
	discriminator = Discriminator(n_blocks=1, base_channels=8)

	generator,discriminator = train_srgan(generator, discriminator, dataloader, device, experiment, lr=1e-5, total_steps=gan_steps, display_step=1000)
	
	torch.save(generator, f'srresnet_{model_name}.pt')
	torch.save(discriminator, f'srdiscriminator_{model_name}.pt')


if __name__=="__main__":
    main()
