import torchvision.transforms.functional as TF
class SquarePad:
    def __init__(self,padding,padding_mode):
        self.padding = padding
        self.padding_mode = padding_mode
        
    def __call__(self, image):
        return TF.pad(image, padding = self.padding,padding_mode = self.padding_mode)