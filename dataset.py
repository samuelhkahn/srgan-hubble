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


class SR_HST_HSC_Dataset(Dataset):
    '''
    Dataset Class
    Values:
        hr_size: spatial size of high-resolution image, a list/tuple
        lr_size: spatial size of low-resolution image, a list/tuple
        *args/**kwargs: all other arguments for subclassed torchvision dataset
    '''

    def __init__(self, hst_path: str, hsc_path:str, hr_size: list, lr_size: list, transform_type: str ) -> None:
        super().__init__()

        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == 6 * lr_size[0]
            assert hr_size[1] == 6 * lr_size[1]
        
#         # High-res images are cropped and scaled to [-1, 1]
#         self.hr_transforms = transforms.Compose([
#             transforms.RandomCrop(hr_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.Lambda(lambda img: np.array(img)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

        self.median_scale = 0.32154497558051215
        self.mean_scale = 0.31601302214882165
        self.hst_min = -2.318
        self.hsc_min = -0.168
        
        self.hst_path = hst_path
        self.hsc_path = hsc_path
        self.transform_type = transform_type

        self.filenames = os.listdir(hst_path)


        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        
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

    def scale_tensor(self,tensor:np.ndarray, scale:float) -> np.ndarray:
        return scale*tensor

    def log_transformation(self,tensor:np.ndarray,min_pix:float,eps:float) -> np.ndarray:
        transformed = tensor+np.abs(min_pix)+eps
        transformed = np.log10(transformed)
        return transformed
    def median_transformation(self,tensor:np.ndarray) -> np.ndarray:
        y = tensor - np.median(tensor)
        y_std = np.std(y)
        normalized = y/y_std
        #max_val = np.max(normalized)
        #min_val = np.min(normalized)
        #denominator = max_val-min_val
        #normalized = (normalized-min_val)/denominator
        ##print(f"Normalized Max:{np.max(normalized)}")
        return normalized
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple:


        hst_image = os.path.join(self.hst_path,self.filenames[idx])
        hsc_image = os.path.join(self.hsc_path,self.filenames[idx])
        
        hst_array = self.load_fits(hst_image)
        hsc_array = self.load_fits(hsc_image)

        # scale LR image with median scale factor
        #hsc_array = self.scale_tensor(hsc_array,self.median_scale)
        print(type(hsc_array))
        if self.transform_type == "sigmoid":
            hst_transformation = self.sigmoid_transformation(hst_array)
            hsc_transformation = self.sigmoid_transformation(hsc_array)

        elif self.transform_type == "log_scale":
            hst_transformation = self.log_transformation(hst_array,self.hst_min,1e-6)
            hsc_transformation = self.log_transformation(hsc_array,self.hst_min,1e-6)

        elif self.transform_type == "median_scale":
            hst_transformation = self.median_transformation(hst_array)
            hsc_transformation = self.median_transformation(hsc_array)
            
        else:
            hst_transformation = hst_array
            hsc_transformation = hsc_array
        # print(type(hst_transformation))
        # hst_transformation = torch.from_numpy(hst_transformation)
        # hsc_transformation = torch.from_numpy(hsc_transformation)
        hst_transformation = self.to_pil(hst_transformation)
        hsc_transformation = self.to_pil(hsc_transformation)

        ## Flip Augmentations
        if random.random() > 0.5:
            hst_transformation = TF.vflip(hst_transformation)
            hsc_transformation  = TF.vflip(hsc_transformation)

        if random.random() >0.5:
            hst_transformation = TF.hflip(hst_transformation)
            hsc_transformation  = TF.hflip(hsc_transformation)

        # Convert to Tensor
        hst_tensor = self.to_tensor(hst_transformation)
        hsc_tensor = self.to_tensor(hsc_transformation)

        # Collapse First Dimension
        hst_tensor = hst_tensor.squeeze(0)
        hsc_tensor = hsc_tensor.squeeze(0)

        return hst_tensor,hsc_tensor
