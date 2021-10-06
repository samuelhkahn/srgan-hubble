import os
from comet_ml import Experiment
import torch
from srresnet_generator import Generator
from discriminator import Discriminator
from trainers import train_srresnet,train_srgan
from dataset import SR_HST_HSC_Dataset
import numpy as np
import configparser


# Load file paths from config
config = configparser.ConfigParser()
config.read('paths.config')
hst_dim = int(config["HST_DIM"]["hst_dim"])
hsc_dim = int(config["HSC_DIM"]["hsc_dim"])


def collate_fn(batch):
	hst_lrs,hst_hrs,lrs, hr_segs = [], [], [], []


	for hst_lr,hst_hr,lr, hr_seg in batch:
		hst_lr_nan = torch.isnan(hst_lr).any()
		hst_hr_nan = torch.isnan(hst_hr).any()
		lr_nan = torch.isnan(lr).any()

		hst_lr_inf = torch.isnan(hst_lr).any()
		hst_hr_inf = torch.isnan(hst_hr).any()
		lr_inf = torch.isinf(lr).any()


		good_vals = [hst_lr_nan,hst_hr_nan,hst_lr_inf,hst_hr_inf,lr_nan,lr_inf]

		if hst_hr.shape == (hst_dim,hst_dim) \
		and hst_lr.shape == (hst_dim,hst_dim) \
		and lr.shape == (hsc_dim,hsc_dim) \
		and True not in good_vals:
			hst_lrs.append(hst_lr)
			hst_hrs.append(hst_hr)
			lrs.append(lr)
			hr_segs.append(hr_seg)

	hst_lrs = torch.stack(hst_lrs, dim=0)
	hst_hrs = torch.stack(hst_hrs, dim=0)
	lrs = torch.stack(lrs, dim=0)
	hr_segs = torch.stack(hr_segs, dim=0)

	return hst_lrs,hst_hrs,lrs, hr_segs


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load file paths from config
	config = configparser.ConfigParser()
	config.read('paths.config')

	# Configuration Information
	hst_path = config["DEFAULT"]["hst_path"]
	hsc_path = config["DEFAULT"]["hsc_path"]

	srresnet_model_name = config["SRRESNET_MODEL_NAME"]["model_name"]
	gan_model_name = config["GAN_MODEL_NAME"]["model_name"]

	comet_tag = config["COMET_TAG"]["comet_tag"]

	batch_size = int(config["BATCH_SIZE"]["batch_size"])
	srresnet_steps = int(config["SRRESENET_STEPS"]["srresnet_steps"])
	gan_steps = int(config["GAN_STEPS"]["gan_steps"])

	hst_dim = int(config["HST_DIM"]["hst_dim"])
	hsc_dim = int(config["HSC_DIM"]["hsc_dim"])

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
	    SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[hst_dim, hst_dim], 
	    	lr_size=[hsc_dim, hsc_dim], transform_type = "global_median_scale",data_aug = True), 
	    batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn
	)

	# Define Generator
	generator = Generator(n_res_blocks=16, n_ps_blocks=2,pix_shuffle=True)

	generator = train_srresnet(generator, dataloader, device, experiment,srresnet_model_name, lr=1e-4, total_steps=srresnet_steps, display_step=1)

	torch.save(generator, f'srresnet_{srresnet_model_name}.pt')

	generator = torch.load(f'srresnet_{srresnet_model_name}.pt')
	discriminator = Discriminator(n_blocks=1, base_channels=8)

	generator,discriminator = train_srgan(generator, discriminator, dataloader, device, experiment,gan_model_name, lr=1e-4, total_steps=gan_steps, display_step=1,lambda_gp=10)
	
	torch.save(generator, f'srgenerator_{gan_model_name}.pt')
	torch.save(discriminator, f'srdiscriminator_{gan_model_name}.pt')


if __name__=="__main__":
    main()
