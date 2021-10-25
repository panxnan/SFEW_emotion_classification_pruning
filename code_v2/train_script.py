import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import pandas as pd
import numpy as np
import read_data
import os
import shutil
from PIL import Image
import MyModels
import dataset
import random
import time


# set cuda
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')
print('use ', device)

# load dataset
dataset_train, dataset_test = dataset.get_dataset(4)
datasets = {
    'train': dataset_train,
    'val': dataset_test
}

# hyper-parameter
batch_size = 16
learning_rate = 1e-3
epoches = 1

seed = 666
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# models
model = MyModels.Resnet50(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
dataLoaders = {
    'train': trainloader,
    'val': testloader
}

def train_model(model, criterion, optimizer, scheduler, num_epoches):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches-1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'val']:
            since_epoch = time.time()
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            
            running_loss = 0.
            running_corrects = 0

            for data in dataLoaders[phase]:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device).type(torch.LongTensor)

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects / len(dataset[phase])

            time_elapsed_epoch = time.time() - since_epoch
            print('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, epoches)