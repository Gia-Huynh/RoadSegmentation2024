import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from FUCKINGEXTERNALDATALOADERFILE import test_dataloader


for num_workers in range(0, 5):
    test_dataloader(num_workers)
print("\nDone running!")

#cum = 0
#This will work
#if __name__ == '__main__':
#    for a in range (0, 15):
#        cum = cum+1
    
#This will not work
for a in range (0, 15):
    cum = cum+1
