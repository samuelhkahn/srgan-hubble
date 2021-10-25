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
	
	srresnet_lr = float(config["SRRESNET_LR"]["srresnet_lr"])
	gan_lr = float(config["GAN_LR"]["gan_lr"])

	pretrained = eval(config["PRETRAINED"]["pretrained"])
	pretrained_model = config["PRETRAINED_MODEL"]["pretrained_model"]

	pretrained_disc = eval(config["PRETRAINED_DISC"]["pretrained_disc"])
	pretrained_disc_model = config["PRETRAINED_DISC_MODEL"]["pretrained_disc_model"]

	data_aug = eval(config["DATA_AUG"]["data_aug"])

	n_res_blocks = eval(config["N_RES_BLOCKS"]["n_res_blocks"])

	display_steps = eval(config["DISPLAY_STEPS"]["display_steps"])


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
	    	lr_size=[hsc_dim, hsc_dim], transform_type = "global_median_scale",data_aug = data_aug), 
	    batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn
	)

	# Define (or load) Generator

	if pretrained==True:
		print(f"Loading Pretrained Model: {pretrained_model}")
		generator = torch.load(pretrained_model)
	else:
		generator = Generator(n_res_blocks=n_res_blocks, n_ps_blocks=2,pix_shuffle=True)
	
	generator = train_srresnet(generator, dataloader, device, experiment,srresnet_model_name, lr=srresnet_lr, total_steps=srresnet_steps, display_step=display_steps)

	torch.save(generator, f'srresnet_{srresnet_model_name}.pt')

	if pretrained_disc==True:
		print(f"Loading Pretrained Discriminator Model: {pretrained_disc_model}")
		discriminator = torch.load(pretrained_disc_model)
	else:
		discriminator = Discriminator(n_blocks=1, base_channels=8)

	generator,discriminator = train_srgan(generator, discriminator, dataloader, device, experiment,gan_model_name, lr=gan_lr, total_steps=gan_steps, display_step=display_steps,lambda_gp=10)
	
	torch.save(generator, f'srgenerator_{gan_model_name}.pt')
	torch.save(discriminator, f'srdiscriminator_{gan_model_name}.pt')

if __name__=="__main__":
    main()
