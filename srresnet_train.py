import os
from comet_ml import Experiment
import torch
from srresnet_generator import Generator
from discriminator import Discriminator
from trainers import train_srresnet,train_srgan
from dataset import SR_HST_HSC_Dataset

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
	    batch_size=3, pin_memory=True, shuffle=True,
	)

	generator = train_srresnet(generator, dataloader, device, experiment, lr=1e-4, total_steps=1, display_step=50)
	
	torch.save(generator, 'srresnet.pt')

if __name__=="__main__":
    main()