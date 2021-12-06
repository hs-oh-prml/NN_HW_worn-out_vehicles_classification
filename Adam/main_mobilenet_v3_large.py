# import
import re 
import os
import PIL
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob
from tqdm import tqdm

# torch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision 
from torchvision import models
import torchvision.transforms as transforms

from torchsummary import summary

import json
from PIL import Image
from pathlib import Path

class AttributeDict(dict):
    def __init__(self):
        self.__dict__ = self
        
class ConfigTree:
    def __init__(self):
        self.DATASET = AttributeDict()
        self.SYSTEM = AttributeDict()
        self.TRAIN = AttributeDict()
        self.MODEL = AttributeDict()
        self.TEST = AttributeDict()

def print_overwrite(step, total_step, loss, acc, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write(f"Train Steps: {step}/{total_step} | Loss: {loss:.4f} | Acc: {acc*100.:.2f} %")   
    else:
        sys.stdout.write(f"Valid Steps: {step}/{total_step} | Loss: {loss:.4f} | Acc: {acc*100.:.2f} %")
    sys.stdout.flush()
    

def get_augmentation(size=224, use_flip=True, use_color_jitter=False, use_gray_scale=False, use_normalize=False):
    resize_crop = transforms.RandomResizedCrop(size=size)
    random_flip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
       
    ], p=0.8)
    gray_scale = transforms.RandomGrayscale(p=0.2)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    to_tensor = transforms.ToTensor()
    
    transforms_array = np.array([resize_crop, random_flip, color_jitter, gray_scale, to_tensor, normalize])
    transforms_mask = np.array([True, use_flip, use_color_jitter, use_gray_scale, True, use_normalize])
    
    transform = transforms.Compose(transforms_array[transforms_mask])
    
    return transform

def encoding_name(filename):
    return os.path.basename(filename).split('.')[0]
    
def submmision(config,file_name,predictions):
    
    PATH = config.SYSTEM.SAVE_DIR
    TEAM_NAME = config.SYSTEM.TEAM_NAME.replace(" ","") # 공백제거
    
    # Make output directory
    # SAVE_PATH = os.path.join(PATH, "output")
    SAVE_PATH = PATH

    os.makedirs(SAVE_PATH, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.join(SAVE_PATH, f'{TEAM_NAME}_{today}.csv')
    
    
    data = np.stack([np.array(file_name),np.array(predictions)],axis=1)
    
    
    # 결과 파일 저장 
    submmision = pd.DataFrame(data=data, columns=['encoded_name','label'])
    submmision['encoded_name'] = submmision['encoded_name'].apply(encoding_name)
    submmision.to_csv(csv_name,index=None)
     
    print(f'|INFO| DATE: {today}')
    print(f'|INFO| 제출 파일 저장 완료: {csv_name}')
    print('하단 주소에 접속하여 캐글Leaderboard 에 업로드 해주세요.')
    print('https://www.kaggle.com/t/16531420f61345978c490712a7a5212b')

"""## Create dataset class"""


class VehicleDataset(Dataset):
    def __init__(self, cfg, mode, transform):
        self.data_root = os.path.join(cfg.DATASET.ROOT, mode)
        self.mode = mode
        self.transform = transform
        self.images = sorted(glob(self.data_root + '/*.jpg'))
        if mode == 'train':
            self.annotations = sorted(glob(self.data_root + '/*.json'))

    def __len__(self):
        if self.mode == 'train':
            assert len(self.images) == len(
                self.annotations), "# of image files and # of json files do not match"
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        image = self.transform(image)

        if self.mode == 'train':
            annotation = self.annotations[index]
            label = self.get_gt(annotation)
            return image, label
        else:
            return image

    def get_gt(self, json_file):
        label_dict = {
            'motorcycle': 0,
            'concrete': 1,
            'bus': 2,
            'benz': 3,
            'suv': 4
        }

        json_file = Path(json_file)
        with open(json_file, 'r') as f:
            annotation = json.load(f)
        gt_class = label_dict[(annotation['label'])]

        return gt_class


config = ConfigTree()
config.DATASET.ROOT = "./dataset/"  # data path
config.DATASET.NUM_CLASSES = 5  # 분류해야 하는 클래스 종류의 수

config.SEED = 2  # seed num


####################################
# 변경 할 부분
####################################

config.SYSTEM.GPU = 0  # GPU 번호
config.SYSTEM.PRINT_FREQ = 2  # 로그를 프린트하는 주기
config.SYSTEM.TEAM_NAME = 'mobilenet_v3_large' # text.replace(" ","")
config.SYSTEM.SAVE_DIR = './save_csv/{}'.format(config.SYSTEM.TEAM_NAME) # 모델 파라미터가 저장되는 위치
config.SYSTEM.SAVE_CHECKPOINT = './checkpoint/{}'.format(config.SYSTEM.TEAM_NAME) # 모델 파라미터가 저장되는 위치

# hyperparameter of experiment
config.TRAIN.EPOCH = 100  # total training epoch
config.TRAIN.BATCH_SIZE = 256
config.TRAIN.BASE_LR = 1e-3
config.TRAIN.WEIGHT_DECAY =0
config.TRAIN.VALID_RATIO = 0.2
config.TRAIN.OPTIM = 'Adam'
config.TRAIN.MODEL = 'mobilenet_v3_large'

# 사용할 augmentation
config.TRAIN.AUGMENTATION = {'size': 112, 'use_flip': True, 'use_color_jitter': True,
                             'use_gray_scale': True, 'use_normalize': True}  
config.TEST.AUGMENTATION = {'size': 224, 'use_normalize': True}

# seed fix 
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# GPU allocation 
device = torch.device(f'cuda:{config.SYSTEM.GPU}' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device()) # check
    with torch.cuda.device(f'cuda:{config.SYSTEM.GPU}'):
        torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""## Train"""

transform = get_augmentation(**config.TRAIN.AUGMENTATION)

dataset = VehicleDataset(config,'train',transform)

len_valid_set = int(config.TRAIN.VALID_RATIO*len(dataset))
len_train_set = len(dataset) - len_valid_set
print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))

train_dataset , valid_dataset,  = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config.TRAIN.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True
                                           )

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                          batch_size=config.TRAIN.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True
                                          )

"""## Testing the shape of input data"""

images, landmarks = next(iter(train_loader))
print(images.shape)

"""## Define the model"""

model_arch=config.TRAIN.MODEL
use_pretrained=False
num_classes=5

model = models.__dict__[model_arch]()

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

# summary(model,(3, 224, 224))

# update_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.__dict__[config.TRAIN.OPTIM]
optimizer = optimizer(model.parameters(), 
                        lr=config.TRAIN.BASE_LR, 
                        betas=[0.9, 0.98],
                        eps=0.000000001,
                        weight_decay=config.TRAIN.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

num_epochs = config.TRAIN.EPOCH
os.makedirs(config.SYSTEM.SAVE_CHECKPOINT, exist_ok=True)

loss_min = np.inf
start_time = time.time()
for epoch in range(1,num_epochs+1):
    
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    
    acc_train = 0
    acc_valid = 0
    running_acc = 0
    
    model.train()
    for step , (images, targets) in enumerate(train_loader):
    
        images, targets = images.to(device), targets.to(device)
        
        predictions = model(images)
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss for the current step
        loss_train_step = criterion(predictions, targets).to(torch.float32)
        
        # calculate the gradients
        loss_train_step.backward()
        
        # update the parameters
        optimizer.step()
        
        loss_train += loss_train_step.item()
        running_loss = loss_train/(step+1)
        
        acc_train += predictions.argmax(-1).eq(targets).float().mean()
        running_acc = acc_train/(step+1)
        
        if step % config.SYSTEM.PRINT_FREQ == 0 :
            print_overwrite(step, len(train_loader), running_loss, running_acc, 'train')
        
    model.eval() 
    with torch.no_grad():
        for step , (images, targets) in enumerate(valid_loader):
            images, targets = images.to(device), targets.to(device)
            
            predictions = model(images)

            # find the loss for the current step
            loss_valid_step = criterion(predictions, targets).to(torch.float32)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/(step+1)

            acc_valid += predictions.argmax(-1).eq(targets).float().mean()
            running_acc = acc_valid/(step+1)
            
            if step % config.SYSTEM.PRINT_FREQ == 0:
                print_overwrite(step, len(valid_loader), running_loss, running_acc, 'valid')
    
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)
    
    acc_train /= len(train_loader)
    acc_valid /= len(valid_loader)
    
    eta = time.time()-start_time
    epoch_eta = time.strftime('%H:%M:%S', time.gmtime(eta))
    print('\n\n'+'-'*90) 
    print('ETA: {} | Epoch: {}/{} | [Train] Loss: {:.4f}, Acc: {:.2f} % | [Valid] Loss: {:.4f}, Acc: {:.2f} %'
      .format(epoch_eta, epoch, num_epochs, loss_train,acc_train*100., loss_valid, acc_valid*100.))
    print('-'*90)
    
    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(model.state_dict(), os.path.join(config.SYSTEM.SAVE_CHECKPOINT, './vehicle_recognition_proposed_best.pth')) 
        print("\nMinimum Validation Loss of {:.4f} & Acc of {:.2f} % at epoch {}/{}".format(loss_min,acc_valid*100., epoch, num_epochs))
        print('Model Saved\n')
        
print('Training Complete')
print("Total Elapsed Time : {}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))))

"""## Submission

### Predict on Test Images
"""

test_transform = get_augmentation(**config.TEST.AUGMENTATION)
test_dataset = VehicleDataset(config,'test', test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=4,
                                          num_workers=4,
                                          pin_memory=True
                                          )

start_time = time.time()

predictions = []

with torch.no_grad():
    
    best_network = model
    best_network.load_state_dict(torch.load(os.path.join(config.SYSTEM.SAVE_CHECKPOINT, './vehicle_recognition_proposed_best.pth'))) 
    best_network.to(device)
    best_network.eval()

    print('Total number of test images: {}'.format(len(test_loader)))
    for batch_idx, images in tqdm(enumerate(test_loader)):
        images = images.to(device)
        
        outputs = best_network(images)
        pred = outputs.argmax(-1).to('cpu').tolist()
        predictions.extend(pred)

end_time = time.time()
print('Testing Complete')
print("Total Elapsed Time : {}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))))

submmision(config, test_loader.dataset.images, predictions)

