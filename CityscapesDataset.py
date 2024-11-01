from torchvision.datasets import Cityscapes
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
import numpy as np

mean = [0.46617496, 0.36034706, 0.33016744]
std = [0.23478602, 0.21425594, 0.20241965]
#dataset = Cityscapes('D:/DOWNLOAD/Dataset/mmsegmentation/data/cityscapes', split='train', mode='fine',
#                     target_type='semantic')
#img, smnt = dataset[0]
mapping_20 = {  0: 0,                1: 0,                2: 0,                3: 0,                4: 0,
                5: 0,                6: 0,                9: 0,                10: 0,
                14: 0,                15: 0,                16: 0,                18: 0,
                29: 0,                30: 0,                -1: 0,
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
        if self.resize is not None:
            image = v2.functional.resize (tv_tensors.Image(image), self.resize)
            mask = v2.functional.resize (tv_tensors.Mask(mask), self.resize)
        if self.augmented == True:
            image = self.augmentColour(tv_tensors.Image(image))
            image, mask = self.augmentPosition (tv_tensors.Image(image), tv_tensors.Mask(mask)) 
        if self.__transforms is not None:
            image = self.__transforms(image)
        return image, mask
#if __name__ == "__main__":
    #nigga = Cityscapes_ToPascal ('G:/cityscapes',
    #                             transforms=None, augmented = True, resize = (512),
    #                             split='train', mode='fine', target_type='semantic')
    #a,b = nigga[0]


from collections import namedtuple
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

class_to_idx = {labels[i][0]:mapping_20[labels[i][1]] for i in range (0, len(labels))}
#print (LabelBinding)
#class to index of output array.
#from torchvision.models.segmentation import FCN_ResNet50_Weights
#weights = FCN_ResNet50_Weights.DEFAULT
#class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

