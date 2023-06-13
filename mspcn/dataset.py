import torch
import numpy as np
import torch.utils.data as data
from os import listdir
import os
import scipy.io as sio


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class get_training_set(data.Dataset):
    
    def __init__(self,dataPath='dataset/train', transform=None, target_transform=None):
        super(get_training_set, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_mat_file(x)]
        self.dataPath = dataPath
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        img_path = os.path.join(self.dataPath,self.image_list[index])
        label_path = os.path.join(self.dataPath,'target_train',self.image_list[index])

        input = sio.loadmat(img_path)
        input = np.array(input['I1'])
        input = input.reshape([1,32,32])
        target = sio.loadmat(label_path)
        target = np.array(target['I2'])
        target = target.reshape([1,128,128])
        input, target = (torch.from_numpy(input)).float(), (torch.from_numpy(target)).float()

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return input, target

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

class get_test_set(data.Dataset):

    def __init__(self,dataPath='dataset/test', transform=None, target_transform=None):
        super(get_test_set, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_mat_file(x)]
        self.dataPath = dataPath
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        img_path = os.path.join(self.dataPath,self.image_list[index])
        label_path = os.path.join(self.dataPath,'target_test',self.image_list[index])
        
        input = sio.loadmat(img_path)
        input = np.array(input['I1'])
        input = input.reshape([1,32,32])
        target = sio.loadmat(label_path)
        target = np.array(target['I2'])
        target = target.reshape([1,128,128])
        input, target = (torch.from_numpy(input)).float(), (torch.from_numpy(target)).float()

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return input, target

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
