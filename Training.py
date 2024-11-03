import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import numpy as np
import random
#from torchvision.transforms.v2.functional import InterpolationMode
mean = [0.46617496, 0.36034706, 0.33016744]
std = [0.23478602, 0.21425594, 0.20241965]
num_epochs = 500
test_frequency = 10 #1 for debug, 10 for real world
LearningRate = 0.001
batch_size = 5
test_batch_size = 3
num_workers =  2

print (f"Going to train for {num_epochs} epoch")
        
#Dataset, dataloader
print (f"Dataset Loading...")
from torch.utils.data import DataLoader
import CityscapesDataset
import torch.optim as optim

if __name__ == '__main__':
    Dataset = CityscapesDataset.Cityscapes_ToPascal ('G:/cityscapes',
                                     transforms=None, augmented = True, resize = (384),
                                     split='train', mode='fine', target_type='semantic')

    TestDataset = CityscapesDataset.Cityscapes_ToPascal ('G:/cityscapes',
                                     transforms=None, augmented = False, resize = (384),
                                     split='val', mode='fine', target_type='semantic')
    class_to_idx = CityscapesDataset.class_to_idx
    train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=False,
                              num_workers = num_workers, persistent_workers=True, pin_memory = True)
    test_loader = DataLoader(TestDataset, batch_size=test_batch_size, shuffle=False,
                              num_workers = num_workers, persistent_workers=True, pin_memory = True)
print (f"Dataset Loaded")
            
print (f"Model Loading...")
#model, weight, training hyperparameter
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LearningRate)
    #model = torch.load("models/bestmodel.pt", weights_only=False) 
    model.to(device)
    model.train()
    scaler = torch.amp.GradScaler('cuda')
print (f"Model Loaded")


print (f"Start training...")
import SupportFunc
def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
def startTraining (num_epochs):
    def LogImg ():
        for imgIdx in range (0, min(test_batch_size,4)):
            #Original image
            OGimage = ((images[imgIdx]*torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)))*255
            fucking_image = to_pil_image(OGimage.to(torch.uint8))

            #sanity check
            if epoch==0:
                fileName = "TrainingProgress/Begin/" + str(imgIdx) + '_' + str(int(epoch))
                fucking_image.save( fileName + "_OG.png")

                
                sexmask = to_pil_image(torch.cat([torch.zeros((2,*images[imgIdx].shape[1:])),
                                                  (masks[imgIdx]==(class_to_idx["car"])).detach().cpu().unsqueeze(0)],
                                                 dim=0).to(torch.uint8)*255)
                
                fucking_image.paste(sexmask, (0, 0), to_pil_image((masks[imgIdx]).detach().cpu().unsqueeze(0).to(torch.uint8)*150))
                fucking_image.save( fileName + "_MASK.png")
                
                    
            #Model Output
            model_OG = model(images[imgIdx].unsqueeze(0).to(device))['out'].softmax(dim=1)
            model_output = model_OG[:,class_to_idx["car"],:,:].detach().cpu()
            model_image = to_pil_image(torch.cat([model_output, torch.zeros((2,*model_output.shape[1:]))], dim=0))
            #combine
            fucking_image.paste(model_image, (0, 0), to_pil_image(model_output[0]))
            fucking_image.save("TrainingProgress/" + str(imgIdx) + '_' + str(int(epoch)) + ".png")
            
            if (epoch%(test_frequency*3) == 0):
                folderStr = "TrainingProgress/"+str(epoch)+"/"
                SupportFunc.create_folder_if_not_exists (folderStr)
                for i in class_to_idx:
                    idx = class_to_idx[i]
                    if (idx == 0):
                        continue
                    ImgTemp = to_pil_image(OGimage.to(torch.uint8))
                    #Model Output
                    model_output = model_OG[:,idx,:,:].detach().cpu()
                    model_image = to_pil_image(torch.cat([model_output, torch.zeros((2,*model_output.shape[1:]))], dim=0))
                    #combine
                    ImgTemp.paste(model_image, (0, 0), to_pil_image(model_output[0]))
                    ImgTemp.save(folderStr + str(imgIdx) + '_' + str(i) + '_' + str(int(epoch)) + ".png")
                    
            globals().update(locals())

    
    best_loss = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        #Saving a few prediction images during training to visualize training process
        #Skip this part
        for images, masks in test_loader:
            break
        globals().update(locals())
        if ((epoch%test_frequency) == 0) or (epoch < 15):
            LogImg()
            globals().update(locals())
        #Actual training code
        model.train()
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
        torch.cuda.empty_cache()
        #Testing set
        model.eval()
        testing_loss = 0
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)['out']
                loss = criterion(outputs, masks.long())
            testing_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader)}, Testing Loss: {testing_loss/len(test_loader)}")
        if (testing_loss) < best_loss:
            best_loss = testing_loss
            torch.save (model, "models/bestmodel.pt")
    torch.cuda.empty_cache()
    torch.save (model, "models/fuckmodel.pt")
    globals().update(locals()) #STORE ALL VARIABLE TO GLOBAL, the stackoverflow answer hates this though...
    
if __name__ == '__main__':
    startTraining(num_epochs)

