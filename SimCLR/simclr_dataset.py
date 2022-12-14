import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class SimCLRDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.img_dir = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx): # Returns a postive pair of 2 differently augmented images
        img_path = self.img_dir[idx]
        image = read_image(img_path).permute(1, 2, 0).numpy() # transforms.toTensor() wants numpy [H,W,C] [0-255] and goes to tensor [C,H,W] [0-1]

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        return [image1, image2]

if __name__ == '__main__':
    import torchvision.transforms as transforms
    import numpy as np
    import os
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    size = 256

    # Transformations according to paper
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    blur = transforms.GaussianBlur(kernel_size=int(0.1 * size), sigma=np.random.uniform(0.1, 2.0))
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((size, size)),
                                            transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(p=.5),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.RandomApply([blur], p=0.5),
                                            ])
    dataset_dirs = []
    for filename in os.listdir('../images'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('../images', filename)
            dataset_dirs.append(img_path)
    
    dataset = SimCLRDataset(dataset_dirs, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    #img1, img2 = next(iter(loader))
    for img1, img2 in loader:
        plt.imshow(img1.squeeze().T)
        plt.show()
        plt.imshow(img2.squeeze().T)
        plt.show()
        # use 'c' to in pdb to iterate
        import pdb
        pdb.set_trace()
    
