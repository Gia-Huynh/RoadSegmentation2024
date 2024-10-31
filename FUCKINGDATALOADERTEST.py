import torch
from torch.utils.data import DataLoader
from CityscapesDataset import Cityscapes_ToPascal
#This class is just a cityscape dataset wrapper
dataset = Cityscapes_ToPascal ('G:/cityscapes',
                                 transforms=None, augmented = False, resize = (256),
                                 split='train', mode='fine', target_type='semantic')
#                               # pin_memory = True, persistent_workers=True)
import time
    
def test_dataloader(num_workers):
    print(f"num_workers = {num_workers}")
    try:
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=num_workers)
        
        print (len (dataloader))
        cum = 0
        start_time = time.time()
        
        for images, masks in dataloader:
            cum = cum+1
            print (cum,' ',time.time() - start_time)
            start_time = time.time()
            if (cum == 5):
                break
        print(f"Success with num_workers = {num_workers}")
        globals().update(locals())
    except Exception as e:
        print(f"Failed with num_workers = {num_workers}: {e}")

for num_workers in range(1,5):
    test_dataloader(num_workers)
print("\nDone running!")

#train_loader = DataLoader(Dataset, batch_size=16, shuffle=False,
#                          num_workers = 2)
# pin_memory = True, persistent_workers=True)
print (f"Dataset Loaded")
#print (len (train_loader))
cum = 0
start_time = time.time()
for images, masks in zip(range (0, 15), range (0, 15)):
#for a in range (0, 15):
    cum = cum+1
    print (cum,' ',time.time() - start_time)
    start_time = time.time()
    if (cum == 5):
        break
#for images, masks in dataloader:
#    cum = cum+1
#    print (cum,' ',time.time() - start_time)
#    start_time = time.time()
#    if (cum == 5):
#        break