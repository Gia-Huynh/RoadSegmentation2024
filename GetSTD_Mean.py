#https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/
#Fucking chatgpt generated blog post
import os
def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        mean += images.mean(axis=(0, 2, 3))
        std += images.std(axis=(0, 2, 3))
    batch_size, num_channels, height, width = images.shape
    mean /= len(loader)
    std /= len(loader)

    return mean, std

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
#mean, std = get_mean_std(train_loader)
#np.set_printoptions(suppress=True)
#print (mean.numpy(),' ',std.numpy())
