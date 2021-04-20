from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader
import os
import torch

class SR_HST_HSC_Dataset(Dataset):
    '''
    Dataset Class
    Values:
        hr_size: spatial size of high-resolution image, a list/tuple
        lr_size: spatial size of low-resolution image, a list/tuple
        *args/**kwargs: all other arguments for subclassed torchvision dataset
    '''

    def __init__(self, hst_path: str, hsc_path:str, hr_size: list, lr_size: list ) -> None:
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

        
        self.hst_path = hst_path
        self.hsc_path = hsc_path 
        self.filenames = os.listdir(hst_path)


        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        
    def load_fits(self, file_path: str) -> np.ndarray:
        cutout = fits.open(file_path)
        array = cutout[0].data
        array = array.astype(np.float32) # Big->little endian
        return array

    def sigmoid_array(self,x):                                        
        return 1 / (1 + np.exp(-x))

    def sigmoid_transformation(self,x:np.ndarray) -> np.ndarray:
        x = self.sigmoid_array(x) #shift to make noise more apparent
        x = 2*(x-0.5)
        return x


    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple:


        hst_image = os.path.join(self.hst_path,self.filenames[idx])
        hsc_image = os.path.join(self.hsc_path,self.filenames[idx])
        
        hst_array = self.load_fits(hst_image)
        hsc_array = self.load_fits(hsc_image)

        hst_transformation = self.sigmoid_transformation(hst_array)
        hsc_transformation = self.sigmoid_transformation(hsc_array)

        hst_tensor = torch.from_numpy(hst_transformation)
        hsc_tensor = torch.from_numpy(hsc_transformation)
        
        return hst_tensor,hsc_tensor

    @staticmethod
    def collate_fn(batch):
        hrs, lrs = [], []

        for hr, lr in batch:
            hrs.append(hr)
            lrs.append(lr)

        return torch.stack(hrs, dim=0), torch.stack(lrs, dim=0)