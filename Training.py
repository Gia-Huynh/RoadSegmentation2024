import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np
import random
#from torchvision.transforms.v2.functional import InterpolationMode
mean = [0.46617496, 0.36034706, 0.33016744]
std = [0.23478602, 0.21425594, 0.20241965]
num_epochs = 50
test_frequency = 10 #1 for debug, 10 for real world
LearningRate = 0.001
batch_size = 10
num_workers =  8

print (f"Going to train for {num_epochs} epoch")
        
#Dataset, dataloader
print (f"Dataset Loading...")
from torch.utils.data import DataLoader
from CityscapesDataset import Cityscapes_ToPascal
import torch.optim as optim
Dataset = Cityscapes_ToPascal ('G:/cityscapes',
                                 transforms=None, augmented = False, resize = (384),
                                 split='train', mode='fine', target_type='semantic')
train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=False,
                          num_workers = num_workers, persistent_workers=True, pin_memory = True)
print (f"Dataset Loaded")
            
print (f"Model Loading...")
#model, weight, training hyperparameter
if __name__ == '__main__':
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LearningRate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    #class to index of output array.
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    
print (f"Model Loaded")


print (f"Start training...")
def startTraining (num_epochs):
    best_loss = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        #Saving a few prediction images during training to visualize training process
        #Skip this part
        for images, masks in train_loader:
            break
        globals().update(locals())
        if ((epoch%test_frequency) == 0) or (epoch < 15):
            for imgIdx in range (0, min(batch_size,4)):
                #Original image
                fucking_image = ((images[imgIdx]*torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)))*255
                fucking_image = to_pil_image(fucking_image.to(torch.uint8))
                if epoch==0:
                    fucking_image.save("TrainingProgress/Begin/" + str(imgIdx) + '_' + str(int(epoch)) + "_OG.png")
                    #Mask (Sanity check)
                    sexmask = to_pil_image(torch.cat([torch.zeros((2,*images[imgIdx].shape[1:])),
                                                      (masks[0]==(class_to_idx["car"])).detach().cpu().unsqueeze(0)],
                                                     dim=0
                                                     ).to(torch.uint8)*255)
                    #combine (Sanity check)
                    fucking_image.paste(sexmask, (0, 0), to_pil_image((masks[imgIdx]).detach().cpu().unsqueeze(0).to(torch.uint8)*150))
                    fucking_image.save("TrainingProgress/Begin/" + str(imgIdx) + '_' + str(int(epoch)) + "_MASK.png")
                    return 0
                #Model Output
                model_output = model(images[imgIdx].unsqueeze(0).to(device))['out'].softmax(dim=1)[:,class_to_idx["car"],:,:].detach().cpu()
                model_image = to_pil_image(torch.cat([model_output, torch.zeros((2,*model_output.shape[1:]))], dim=0))
                #combine
                fucking_image.paste(model_image, (0, 0), to_pil_image(model_output[0]))
                fucking_image.save("TrainingProgress/" + str(imgIdx) + '_' + str(int(epoch)) + ".png")            
            globals().update(locals())
        #Actual training code
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)['out']
                loss = criterion(outputs, masks.long()) #*class_to_idx["person"])
            # Scales the loss, and calls backward()
            # to create scaled gradients
            # (FP16) helps reducing vram
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            #updating global variable
            globals().update(locals())

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        if (running_loss/len(train_loader)) < best_loss:
            best_loss = running_loss
            torch.save (model, "models/bestmodel.pt")
    torch.cuda.empty_cache()
    torch.save (model, "models/fuckmodel.pt")
    globals().update(locals()) #STORE ALL VARIABLE TO GLOBAL, the stackoverflow answer hates this though...
    
if __name__ == '__main__':
    startTraining(num_epochs)

