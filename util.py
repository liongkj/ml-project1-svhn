import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset,TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch import nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from torchvision import transforms, utils
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import seaborn as sns

random_seed = 777
seed_everything(random_seed)

# access to the dict
def load_data(test=False):
    if test:
        train_data = sio.loadmat('./data/test_32x32.mat')    
    else:
        train_data = sio.loadmat('./data/train_32x32.mat')
    x_train = train_data['X']
    y_train = train_data['y']
    x_train = np.moveaxis(x_train,-1,0)
    return x_train,y_train

# show data
class SeedDataset(Dataset):
    def __init__(self,test=False, rgb=True, img_size=32):
        x,y = load_data(test)
        idx=y==10
        y[idx]=0
        self.data = x,y
        self.transform_gray = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),transforms.Resize(img_size),transforms.Normalize((0.4376821, ), (0.19803012,),)])
        self.transform_rgb = transforms.Compose([transforms.ToTensor(),transforms.Resize(img_size), transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614),)])
        self.rgb=rgb
        
    def __len__(self):
        return len(self.data[1])

    def __getitem__(self,idx):
        data = self.data[0][idx,:,:,:]
        label = self.data[1][idx]
        if self.rgb:
            data = self.transform_rgb(data)
        else:
            data = self.transform_gray(data)
        return data, torch.tensor(label,dtype=torch.long)

    def show(self,idx):
        plt.imshow(self.data[0][idx,:,:,:])
        plt.show()
        print(self.data[1][idx])
    
    def plotim(self,idx):
        plt.imshow(torch.squeeze(ds[idx][0]))

class SVHNDataModule(pl.LightningDataModule):

    def __init__(self,batch_size=32,rgb=True,img_size=28):
        super().__init__()
        self.batch_size=batch_size
        self.rgb = rgb
        self.img_size=img_size

    def setup(self,stage=None):

        if stage == 'fit' or stage is None:
            train_full = SeedDataset(test=False, rgb=self.rgb,img_size=self.img_size)
            self.train_set , self.val_set = train_test_split(train_full,test_size=0.1,shuffle=True,random_state=random_seed)
            self.visual_set = self.train_set[:1000]
        
        if stage == 'test' or stage is None:
            self.test_set = SeedDataset(test=True,rgb=self.rgb,img_size=self.img_size)
        

    def train_dataloader(self):
        return [DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True),DataLoader(self.visual_set,batch_size=1000)]

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=1024,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=1)
        
        
def plot_feature(w,number=0):
    no = number-1
    image = w[0][no].to("cpu").numpy()
    min = np.abs(image).max()
    img = (image).reshape(32,32,-1)
    if(img.shape[2]==3):
        img = img * 255
    else:
        img = np.squeeze(img,2)
    plt.imshow(img, vmin=-min,vmax=min, cmap="gray",interpolation='bilinear')
       

def imshow_features(weights,show_plot=False):
    fig = plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plot_feature(weights,i)
    if show_plot:
        plt.show()
    return plt.figure()