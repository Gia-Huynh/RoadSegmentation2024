from torchvision.datasets import Cityscapes
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
import numpy as np

mean = [0.46617496, 0.36034706, 0.33016744]
std = [0.23478602, 0.21425594, 0.20241965]
from torchvision.models.segmentation import FCN_ResNet50_Weights
weights = FCN_ResNet50_Weights.DEFAULT
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
#dataset = Cityscapes('D:/DOWNLOAD/Dataset/mmsegmentation/data/cityscapes', split='train', mode='fine',
#                     target_type='semantic')
#img, smnt = dataset[0]
mapping_20 = {  0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                9: 0,
                10: 0,
                14: 0,
                15: 0,
                16: 0,
                18: 0,
                29: 0,
                30: 0,
                -1: 0,
        24: 15,
        26: 7,
        32: 14,
        33: 2,
        21: 16,
        #Lez fucking gooooooo
        7: 1,
        8: 3,
        11: 4,
        12: 5,
        13: 6,
        17: 8,
        19: 9,
        20: 10,
        22: 11,
        23: 12,
        25: 13,
        27: 17,
        28: 18,
        31: 19,
    }
class Cityscapes_ToPascal(Cityscapes):
    def __init__(self, *args, __transforms=None, augmented = False, resize = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__transforms = __transforms
        self.augmented = augmented
        self.resize = resize
        if self.__transforms is None:
            self.__transforms = v2.Compose([    
                #v2.ToDtype(torch.float32, scale=True), #Auto scale from 0-255 to 0.0-1.0
                v2.Normalize(mean=mean, std=std)
            ])
        if augmented == True:
            self.augmentPosition = v2.Compose([
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.1),
                v2.RandomRotation(degrees = 5),
                v2.RandomAffine(degrees = 5),
                v2.RandomPerspective(distortion_scale=0.15, p=0.80)
            ])
            self.augmentColour = v2.Compose([
                v2.ColorJitter(brightness=0.35, contrast = 0.25, saturation = 0.1, hue = 0.04),
                v2.GaussianNoise(mean = 0, sigma = 0.01) #default sigma 0.1
            ])
        
    def replaceMask(self, mask_OG):
        mask = np.zeros_like(mask_OG)
        for k in mapping_20:
            mask[mask_OG == k] = mapping_20[k]
        return mask
    
    # Override the __getitem__ method
    def __getitem__(self, idx):
        image, mask_OG = super().__getitem__(idx)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)
        image = v2.functional.to_dtype (image, scale = True)
        #mask_OG = np.transpose(np.array(mask_OG), (2, 0, 1))
        mask = self.replaceMask(np.array(mask_OG))
        mask = torch.from_numpy(mask)
        if self.augmented == True:
            image = self.augmentColour(tv_tensors.Image(image))
            image, mask = self.augmentPosition (tv_tensors.Image(image), tv_tensors.Mask(mask)) 
        if self.resize is not None:
            image = v2.functional.resize (tv_tensors.Image(image), self.resize)
            mask = v2.functional.resize (tv_tensors.Mask(mask), self.resize)
        if self.__transforms is not None:
            image = self.__transforms(image)
        return image, mask
if __name__ == "__main__":
    nigga = Cityscapes_ToPascal ('G:/cityscapes',
                                 transforms=None, augmented = True, resize = (512),
                                 split='train', mode='fine', target_type='semantic')
    a,b = nigga[0]


#class to index of output array.
#from torchvision.models.segmentation import FCN_ResNet50_Weights
#weights = FCN_ResNet50_Weights.DEFAULT
#class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

