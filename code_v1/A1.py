#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
from collections import OrderedDict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import math


# ## Define dataloader, Neural network, and other until functions
# - csv2dataset: load data from xlsx file
# - Multip_layer: three layer network, the hidden layer size could be modified
# - get_overall_accuray, get_class_wise_accuracy: function to get accuracy

# In[4]:


"""
Fist step load data from xlsx
"""
def csv2dataset(path='data/SFEW.xlsx'):
    exl = pd.read_excel(path)
    exl.dropna(inplace=True)

    pca = exl.iloc[:, 2:].values
    target = exl.iloc[:, 1].values

    input_tensor = torch.tensor(pca, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.long) - 1

    # apply normalization
    input_mean = torch.mean(input_tensor, dim=0)
    input_std = torch.std(input_tensor, dim=0)
    input_tensor = (input_tensor - input_mean) / input_std

    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    print('load data set: %d samples' % len(dataset))
    return dataset


# In[9]:


class Multi_layer(nn.Module):
    def __init__(self, input_size, n_hiddens, n_class):
        """
        :param input_size:
        :param n_hiddens: numbers of hidden layer
        :param n_class:
        """
        super(Multi_layer, self).__init__()
        layers = OrderedDict()
        current_size = input_size
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_size, n_hidden)
            layers['tanh{}'.format(i+1)] = nn.Tanh()
            current_size = n_hidden
        layers['fc_out'] = nn.Linear(current_size, n_class)
        # layers['softmax_out'] = nn.Softmax(dim=1)
        self.model = nn.Sequential(layers)

    def forward(self, x):
        x = self.model.forward(x)
        return x

def reset_weights(m):
    '''
        try resetting model weights to avoid weight leakage
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters


# In[10]:


def get_overall_accuracy(model, dataLoader):
    correct = 0
    total = 0

    for _, (pca, target) in enumerate(dataLoader, 0):
        outputs = model(pca)
        _, predicted = torch.max(outputs, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def get_class_wise_accuracy(model, dataLoader, n_class=7):
    class_correct = [0 for i in range(n_class)]
    class_total = [0 for i in range(n_class)]
    for (pca, targets) in dataLoader:
        outputs = model(pca)
        _, predicted = torch.max(outputs, dim=1)
        c = (predicted == targets).squeeze()

        for i in range(c.shape[0]):
            target = targets[i]
            class_correct[target] += c[i].item()
            class_total[target] += 1

    return class_correct, class_total


# ## Define the hyper-parameters
# Also save the parameters of kfold id for the later test and retraining

# In[23]:


# hyper-parameters
batch_size = 16
learning_rate = 1e-4
epoches = 4000
k_fold = 5
hidden = [128]

torch.manual_seed(5)
random.seed(5)
np.random.seed(5)
criterion = nn.CrossEntropyLoss()


# In[22]:


# Define the k-fold cross validator
dataset = csv2dataset()
kfold = KFold(n_splits=k_fold, shuffle=True)

# save the training ids for the later experiences
fold_ids = {}
for fold, ids in enumerate(kfold.split(dataset)):
    fold_ids[fold] = ids


# #### Training script
# 
# - Use the tensorboard to record the accuracy and loss during the training
# 
# - save the model after training
# 
# - different hidden size were trained: [16, 32, 64, 128]

# In[211]:


s = '1_SFEW_h128_b16_lr4_ep4000'
writer = SummaryWriter('runs/' + s)

# K-fold cross validation model evaluation
for fold, (train_ids, test_ids) in fold_ids.items():
    print('--------------------')
    print(f'FOLD {fold}')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler = train_subsampler
        )
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler = test_subsampler
        )

    # initial model
    model = Multi_layer(10, hidden, 7)
    model.apply(reset_weights)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    current_loss = 0.
    test_loss = 0.
    for epoch in range(epoches):
        if epoch%1000 == 999:
            print(f'Start epoch{epoch+1}')

        for i, data in enumerate(trainloader, start=0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        current_loss += loss.item()
        
        #compute test loss
        with torch.no_grad():
            #randomly select only one batch
            inputs, targets = iter(testloader).next()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
                

        if epoch%100 == 99:
            train_ac = get_overall_accuracy(model, trainloader)
            test_ac = get_overall_accuracy(model, testloader)

            writer.add_scalar(f'fold {fold} loss/training loss', current_loss/100, epoch)
            writer.add_scalar(f'fold {fold} loss/test loss', test_loss/100, epoch)
            writer.add_scalar(f'fold {fold} AC/training accuracy', train_ac, epoch)
            writer.add_scalar(f'fold {fold} AC/test accuracy', test_ac, epoch)
            
            current_loss = 0.
            test_loss = 0.
    
    # calculate the cw_ac and save in dict
    correct, total = get_class_wise_accuracy(model, testloader)
    m_accuracy = [c/t for (c,t) in zip(correct, total)]
    ac_dict = {}    
    for i in range(len(m_accuracy)):
        print('Accuracy of %d : %2d %%' %
            (i, m_accuracy[i]*100))
        ac_dict[i] = m_accuracy[i]*100  
    final_ac = get_overall_accuracy(model, testloader)
    print(f'final accuray{final_ac}')
    
    ac_dict['average'] = final_ac
    cw_ac_dict[fold] = ac_dict
    
    save_path = 'models/'+ s + f'_model-fold{fold}.pth'
    torch.save(model.state_dict(), save_path)
writer.close()
print('training finished')


# ## Pruning the network
# After training the network, I need to use different technique to pruning the network
# - the distictiveness similarity
# - the distictiveness non-functional units
# 
# The distinctiveness compare the angle between the output of units and also find the non-functional units

# In[380]:


# name of path that the model is saved, along with the hidden unit size
path_set = {
    '2_SFEW_h16_b16_lr4_ep6000': [16],
    '1_SFEW_h32_b16_lr4_ep6000': [32],
    '4_SFEW_h64_b16_lr4_ep6000': [64],
    '2_SFEW_h128_b16_lr4_ep6000': [128],
    
    '1_SFEW_h16_b16_lr4_ep4000': [16],
    '1_SFEW_h32_b16_lr4_ep4000': [32],
    '2_SFEW_h64_b16_lr4_ep4000': [64],    
    '1_SFEW_h128_b16_lr4_ep4000': [128]
}


# In[431]:


# select the second fold as the final test ac is most similar
ids = fold_ids[1]
train_ids, test_ids = ids #len(test_ids) = 135
test_patterns, test_targets = dataset[test_ids]
train_patterns, train_targets = dataset[train_ids]

train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

trainloader = torch.utils.data.DataLoader(
                            dataset, batch_size=batch_size,
                            sampler = train_subsampler)
testloader = torch.utils.data.DataLoader(
                            dataset, batch_size=batch_size,
                            sampler = test_subsampler)


# In[316]:


# data to save
first_layer_output = {}
units_angles = {}
units_output_mean = {}
units_output_std = {}
models = {}


# In[423]:


""" 
Define a pruning function:
    - Prune units the list of index of units that need to remove
    - similar units: a dict[unit_to_keep] = list of other units that are similar to unit_to_keep
"""
def pruning_network(model_old, prune_unites, similar_units=None):
    
    fc = model_old.model[0].weight.clone().detach()
    fc_b = model_old.model[0].bias.clone().detach()
    fc_out = model_old.model[2].weight
    fc_out_b = model_old.model[2].bias
    
    if similar_units != None:
        for idx, unit_list in similar_units.items():
            temp = fc[idx]
            temp_b = fc_b[idx]
            for jdx in unit_list:
                temp += fc[jdx]
                temp_b = fc_b[jdx]
            temp /= len(unit_list) + 1
            temp_b /= len(unit_list) + 1
            
            fc[idx] = temp
            fc_b[idx] = temp_b
            
    h_units = fc.shape[0]
    msk = set(range(h_units)) - prune_unites
    msk = list(msk)
    
    model_new = Multi_layer(10, [len(msk)], 7)
    model_new.model[0].weight = nn.Parameter(fc[msk])
    model_new.model[0].bias = nn.Parameter(fc_b[msk])
    model_new.model[2].weight = nn.Parameter(fc_out[:, msk])
    model_new.model[2].bias = nn.Parameter(fc_out_b)
    
    return model_new


# In[433]:


# calculate the angle between units in different model, and print it
for path in path_set:
    # load model
    print(path)
    load_path = 'models/'+ path + '_model-fold1.pth'
    model = Multi_layer(10, path_set[path], 7)
    model.load_state_dict(torch.load(load_path))
    models[path] = model
    
    # calculate the patterns outputs
    fc = model.model[0]
    tanh = model.model[1]
    output = tanh(fc(train_patterns))
    
    first_layer_output[path] = output
    
    angles = {}
    temp = output.clone().detach().T
    for i in range(temp.shape[0]):
        for j in range(i+1, temp.shape[0]):
            a = temp[i]
            b = temp[j]
            inner = torch.inner(a, b)
            a_norm = a.pow(2).sum().pow(0.5)
            b_norm = b.pow(2).sum().pow(0.5)
            cos = inner / (a_norm * b_norm)
            angle = torch.acos(cos) * (180/math.pi)
            angles[(i, j)] = angle
    
    # save angles in dict
    units_angles[path] = angles
    print(f'number of pairs: {len(angles)}')
    
    # print range of angles
    min_a = 180
    max_a = 0 
    for s, angle in angles.items():
        if angle.item() < min_a:
            min_a = angle.item()
        if angle.item() > max_a:
            max_a = angle.item()
    print('angle range: [%.2f, %.2f]'%(min_a, max_a))
    
    # To find the non-functional units, calculate the mean and std of the output
    mean = output.sum(axis=0)/(output.shape[0])
    std = output.std(axis=0)
    units_output_mean[path] = mean
    units_output_std[path] = std
    print('mean range: [%.4f, %.4f]' % (min(mean), max(mean)))
    print('std range: [%.4f, %.4f]' % (min(std), max(std)))
    
    print('\n')


# In[434]:


# To find the units that bounds in -1, 0, 1 -> [-1, -0.9], [-0.05, 0.05], [0.95, 1]
bound_units = {}
for path in path_set:
    print(path)
    bound_set = set()
    mean = units_output_mean[path]
    std = units_output_std[path]
    for i, value in enumerate(mean):
        if value<-0.9 or (-0.05<value<0.05) or value>0.9:
#             print('%d: mean: %.2f, std: %.2f' % (i, mean[i], std[i]))
            bound_set.add(i)
    bound_units[path] = bound_set
    print(f'set size: {len(bound_set)}')
    print()


# In[435]:


# pruning the network of bound units
models_new = {}

for path in path_set:
    print(path)
    
    model_old = models[path]
    prune_units = bound_units[path]
    print(f'prune units: {len(prune_units)}')
    
    if len(prune_units) == 0:
        model_new = model_old
    else:
        model_new = pruning_network(model_old, prune_units)
    
    models_new[path] = model_new
    
    outputs = model_old(test_patterns)
    loss = criterion(outputs, test_targets).item()
    print('%.2f' % loss)
    
    outputs = model_new(patterns)
    loss = criterion(outputs, test_targets).item()
    print('%.2f' % loss)
    
    print('%.2f' % get_overall_accuracy(model_old, testloader))
    print('%.2f' % get_overall_accuracy(model_new, testloader))
    print('\n')
    print()
    


# In[436]:


## retrain the pruned model for 500 epoches
retrained_epoches = 500
model_bound_retrain = {}

for path in path_set:
    print(path)
    model = models_new[path]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoches in range(retrained_epoches):
        for i, (inputs, targets) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    model_bound_retrain[path] = model
    
    outputs = model(test_patterns)
    loss = criterion(outputs, test_targets).item()
    print('retrained loss: %.2f' % loss)
    
    ac = get_overall_accuracy(model, testloader)
    print('retrain ac: %.2f' % ac)
    print()
    


# In[437]:


# find similar angles and complementary ones
similar_units = {}
complementary_units = {}
for path in path_set:
    similar = []
    complementary = set()
    angles = units_angles[path]
    for pair, value in angles.items():
        if value<30:
            similar.append(pair)
        if value>150:
            a, b = pair
            complementary.add(a)
            complementary.add(b)
    
    similar_units[path] = similar
    complementary_units[path] = complementary


# In[439]:


for path in path_set:
    c = similar_units[path]
    if len(c) ==0: 
        similar_units[path] = {}
        continue
    i2j = dict()
    j2i = dict()

    # merge similar group
    for i, j in c:
        if i not in i2j.keys():
            if j not in j2i.keys():
                i2j[i] = {j}
                j2i[j] = i
            else:
                i2j[j2i[j]].add(j)    
        elif i in i2j.keys():
            i2j[i].add(j)
            j2i[j] = i
    similar_units[path] = i2j
    for s in i2j.values():
        for value in s:
            complementary_units[path].add(value)
similar_units


# In[446]:


# prune network by angles
models_angle = {}
models_angle_retrain = {}
for path in path_set:
    print(path)
    similar = similar_units[path]
    remove = complementary_units[path]
    s = len(similar.keys())
    r = len(remove)
    print(f'{s}')
    print(f'{r}')
    
    model_old = models[path]
    if s == 0 and r == 0:
        model_new = model_old
    else:
        model_new = pruning_network(model_old, remove, similar)
    
    models_angle[path] = model_new
    outputs = model_old(test_patterns)
    loss_o = criterion(outputs, test_targets).item()
    
    
    outputs = model_new(patterns)
    loss_n = criterion(outputs, test_targets).item()
    
    
    ac_o = get_overall_accuracy(model_old, testloader)
    ac_n = get_overall_accuracy(model_new, testloader)
    
    optimizer = optim.SGD(model_new.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoches in range(retrained_epoches):
        for i, (inputs, targets) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model_new(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    models_angle_retrain[path] = model_new
    
    outputs = model_new(test_patterns)
    loss_r = criterion(outputs, test_targets).item()
    
    
    ac = get_overall_accuracy(model_new, testloader)
    print('%.2f' % ac_o)
    print('%.2f' % ac_n)
    print('%.2f' % ac)
    
    print('%.2f' % loss_o)
    print('%.2f' % loss_n)
    print('%.2f' % loss_r)
    
    print('\n')
    


# In[ ]:




