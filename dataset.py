from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader
import os
import torch
import random
import torchvision.transforms.functional as TF
import sep
from torchvision.transforms import CenterCrop

class SR_HST_HSC_Dataset(Dataset):
    '''
    Dataset Class
    Values:
        hr_size: spatial size of high-resolution image, a list/tuple
        lr_size: spatial size of low-resolution image, a list/tuple
        *args/**kwargs: all other arguments for subclassed torchvision dataset
    '''

    def __init__(self, hst_path: str, hsc_path:str, hr_size: list, lr_size: list, transform_type: str, data_aug: bool ) -> None:
        super().__init__()

        sep.set_extract_pixstack(1000000)

        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == 6 * lr_size[0]
            assert hr_size[1] == 6 * lr_size[1]


        self.hsc_std = 0.04180176637927356
        self.hst_std = 0.0010912614529011736

        self.median_scale = 0.32154497558051215
        self.mean_scale = 0.31601302214882165

        self.hst_median = 2.696401406865334e-05
        self.hsc_median = 1.4194287359714508e-05

        self.hst_path = hst_path
        self.hsc_path = hsc_path

        self.transform_type = transform_type
        self.data_aug = data_aug

        self.filenames = os.listdir(hst_path)


        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        # HST clip range - (0,99.996)
        self.hst_min,self.hst_max = (-4.656636714935303, 0.11904790546745403)

        # HSC clip range - (0,99.9)
        self.hsc_min,self.hsc_max = (-0.4692089855670929, 12.432257434845326)
        
    def load_fits(self, file_path: str) -> np.ndarray:
        cutout = fits.open(file_path)
        array = cutout[0].data
        array = array.astype(np.float32) # Big->little endian
        return array

    def sigmoid_array(self,x:np.ndarray) -> np.ndarray:                                        
        return 1 / (1 + np.exp(-x))

    def sigmoid_transformation(self,x:np.ndarray) -> np.ndarray:
        x = self.sigmoid_array(x) #shift to make noise more apparent
        x = 2*(x-0.5)
        return x
    def sigmoid_rms_transformation(self,x:np.ndarray,std_scale:float) -> np.ndarray:
        x = self.scale_tensor(x,std_scale,"div")
        x = self.sigmoid_array(x)
        return x

    def scale_tensor(self,tensor:np.ndarray, scale:float,scale_type: str) -> np.ndarray:
        if scale_type == "prod":
            return scale*tensor
        elif scale_type == "div":
            return scale/tensor

    def log_transformation(self,tensor:np.ndarray,min_pix:float,eps:float) -> np.ndarray:
        transformed = tensor+np.abs(min_pix)+eps
        transformed = np.log10(transformed)
        return transformed

    # Local (image level) median tansformation
    def median_transformation(self,tensor:np.ndarray) -> np.ndarray:
        y = tensor - np.median(tensor)
        y_std = np.std(y)
        normalized = y/y_std
        return normalized

    # Global Median Transformation
    def global_median_transformation(self,tensor:np.ndarray,median: float, std:float) -> np.ndarray:
        y = tensor - median
        normalized = y/std
        return normalized

    # Min max normalization with clipping
    @staticmethod
    def min_max_normalization(tensor:np.ndarray, min_val:float, max_val:float) -> np.ndarray:
        tensor =  np.clip(tensor, min_val, max_val)
        numerator = tensor-min_val
        denominator = max_val-min_val
        tensor = numerator/denominator
        return tensor

    @staticmethod
    def invert_min_max_normalization(tensor:np.ndarray, min_val:float, max_val:float) -> np.ndarray:
        denominator = max_val-min_val
        unnormalized=tensor*denominator+min_val
        return unnormalized

    # segmentation map
    def get_segmentation_map(self,pixels:np.ndarray) -> np.ndarray:
            # pixels = pixels.byteswap().newbyteorder()
            bkg = sep.Background(pixels)
            mask = sep.extract(pixels, 3, 
                                err=bkg.globalrms,
                                segmentation_map=True)[1]
            mask[mask>0]=1
            return  mask

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple:


        hst_image = os.path.join(self.hst_path,self.filenames[idx])
        hsc_image = os.path.join(self.hsc_path,self.filenames[idx])
        
        hst_array = self.load_fits(hst_image)
        hsc_array = self.load_fits(hsc_image)

        hst_seg_map = self.get_segmentation_map(hst_array)

        # scale LR image with median scale factor
        #hsc_array = self.scale_tensor(hsc_array,self.median_scale)
        if self.transform_type == "sigmoid":
            hst_transformation = self.sigmoid_transformation(hst_array)
            hsc_transformation = self.sigmoid_transformation(hsc_array)

        elif self.transform_type == "log_scale":
            hst_transformation = self.log_transformation(hst_array,self.hst_min,1e-6)
            hsc_transformation = self.log_transformation(hsc_array,self.hst_min,1e-6)

        elif self.transform_type == "median_scale":
            hst_transformation = self.median_transformation(hst_array)
            hsc_transformation = self.median_transformation(hsc_array)
            
        elif self.transform_type == "sigmoid_rms":
            hst_transformation = self.sigmoid_rms_transformation(hst_array,self.hst_std)
            hsc_transformation = self.sigmoid_rms_transformation(hsc_array,self.hsc_std)
        elif self.transform_type == "global_median_scale":
            hst_transformation = self.global_median_transformation(hst_array,self.hst_median,self.hst_std)
            hsc_transformation = self.global_median_transformation(hsc_array,self.hsc_median,self.hsc_std)
        elif self.transform_type == "clip_min_max_norm":
            hst_transformation = self.min_max_normalization(hst_array,self.hst_min,self.hst_max)
            hsc_transformation = self.min_max_normalization(hsc_array,self.hsc_min,self.hsc_max)
        # Add Segmap to second channel to ensure proper augmentations
        hst_seg_stack = np.dstack((hst_transformation,hst_seg_map))
        hst_seg_stack = self.to_tensor(hst_seg_stack)


        hsc_transformation = self.to_tensor(hsc_transformation)

        if self.data_aug == True:
            # Rotate 
            rotation = random.randint(0,359)
            hsc_transformation  = TF.rotate(hsc_transformation,rotation)
            hst_seg_stack = TF.rotate(hst_seg_stack,rotation)

            #Center Crop 
            hsc_transformation = TF.center_crop(hsc_transformation,[100,100])
            hst_seg_stack = TF.center_crop(hst_seg_stack,[600,600])

        ## Flip Augmentations
            if random.random() > 0.5:
                hsc_transformation  = TF.vflip(hsc_transformation)
                hst_seg_stack  = TF.vflip(hst_seg_stack)
                
            if random.random() >0.5:
                hsc_transformation  = TF.hflip(hsc_transformation)
                hst_seg_stack  = TF.vflip(hst_seg_stack)


        # Collapse First Dimension and extract hst/seg_map
        hst_tensor = hst_seg_stack[0,:,:].squeeze(0)
        hst_seg_map = hst_seg_stack[1,:,:].squeeze(0)
        hsc_tensor = hsc_transformation.squeeze(0)


        return hst_tensor,hsc_tensor,hst_seg_map
