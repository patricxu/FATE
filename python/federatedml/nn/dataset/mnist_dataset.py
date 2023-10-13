import numpy as np
from federatedml.nn.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class MNISTDataset(Dataset):
    
    def __init__(self, flatten_feature=False): # flatten feature or not 
        super(MNISTDataset, self).__init__()
        self.image_folder = None
        self.ids = None
        self.flatten_feature = flatten_feature
        
    def load(self, path):  # read data from path, and set sample ids
        # read using ImageFolder
        self.image_folder = ImageFolder(root=path, transform=transforms.Compose([transforms.ToTensor()]))
        # filename as the image id
        ids = []
        for image_name in self.image_folder.imgs:
            ids.append(image_name[0].split('/')[-1].replace('.jpg', ''))
        self.ids = ids
        return self

    def get_sample_ids(self):  # implement the get sample id interface, simply return ids
        return self.ids
    
    def __len__(self,):  # return the length of the dataset
        return len(self.image_folder)
    
    def __getitem__(self, idx): # get item
        ret = self.image_folder[idx]
        if self.flatten_feature:
            img = ret[0][0].flatten() # return flatten tensor 784-dim
            return img, ret[1] # return tensor and label
        else:
            return ret
