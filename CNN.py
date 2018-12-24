# -*- coding: utf-8 -*-

# http://pytorch.org/
# from os.path import exists
# from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
# platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
# accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

# !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
import torch
############################################################
# Imports
############################################################
# Include your imports here, if any are used.
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms

def extract_data(x_data_filepath, y_data_filepath):
    X = np.load(x_data_filepath)
    y = np.load(y_data_filepath)
    return X, y

##########################ConvolutionalNN on normalized images


############################################################
# Extracting and loading data
############################################################
class Dataset(Dataset):
    """CIFAR-10 image dataset."""
    def __init__(self, X, y, transformations=None):
        self.len = len(X)           
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class FeedForwardNN(nn.Module):

    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(3072, 1500)
        self.fc2 = nn.Linear(1500, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        
        return out

    def get_fc1_params(self):
        return self.fc1.__repr__()
    def get_fc2_params(self):
        return self.fc2.__repr__()

############################################################
# Convolutional Neural Network
############################################################
class ConvolutionalNN(nn.Module):

    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, (3, 3), 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 16, (3, 3), 1, 0)
        self.fc1 = nn.Linear(16 * 13 * 13, 130)
        self.fc2 = nn.Linear(130, 72)
        self.fc3 = nn.Linear(72, 10)       

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        
        return out

    def get_conv1_params(self):
        return self.conv1.__repr__()
    
    def get_pool_params(self):
        return self.pool.__repr__()

    def get_conv2_params(self):
        return self.conv2.__repr__()
    
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()
    
    def get_fc3_params(self):
        return self.fc3.__repr__()

############################################################
# Hyperparameterized Feed Forward Neural Network
############################################################
class HyperParamsFeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(3072, 1500)
        self.fc2 = nn.Linear(1500, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))



def run_experiment(neural_network , train_loader, test_loader, loss_function, optimizer):

    for i, data_train in enumerate(train_loader, 0):
        # Get inputs and labels from data loader
        inputs_train, labels_train = data_train
        inputs_train, labels_train = Variable(inputs_train.cuda()), Variable(labels_train.cuda())
        #inputs_train, labels_train = inputs_train.to(device), labels_train.to(device)
        # Feed the input data into the network 
        print(inputs_train.shape)
        y_train_pred = net(inputs_train)
        
        # Calculate the loss using predicted labels and ground truth labels
        loss = criterion(y_train_pred, labels_train)

        print("epoch: ", "loss: ", loss.data[0])

        # zero gradient
        optimizer.zero_grad()

        # backpropogates to compute gradient
        loss.backward()

        # updates the weghts
        optimizer.step()

        loss_np = loss.data.cpu().numpy()
        print('my original loss', loss_np)

#         running_loss += loss.data[0]
#         if i % 2000 == 1999:   
#             print('new [%5d] loss: %.3f' %
#                   (i + 1, running_loss / 2000))
#             running_loss = 0.0        
        
        # # convert predicted laels into numpy
        # y_train_pred_np = y_train_pred.data.cpu().numpy()

        # # calculate the training accuracy of the current model
        # train_pred_np = np.where(y_train_pred_np>0.5, 1, 0) 
        # train_label_np = labels_train.data.cpu().numpy().reshape(len(labels_train),1)

        # correct = 0
        # for j in range(y_train_pred_np.shape[0]):
        #     if (train_pred_np[j,:] - train_label_np[j,:]).any():
        #         correct += 1

        # train_accuracy = float(correct)/float(len(train_label_np))
        # loss_np = loss.data.cpu().numpy()

#     correct = 0
#     total = 0
#     for data in test_loader:
#         images, labels = data
#         outputs = net(Variable(images.cuda()))
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()

#     print('Accuracy of the network on test images: %d %%' % (
#         100 * correct / total))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            outputs = net(images)
            predicted = torch.max(outputs.data, 1)[1].cuda()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
                    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            outputs = net(images)
            predicted = torch.max(outputs.data, 1)[1].cuda()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total

    return (test_accuracy, train_accuracy, loss_np)

def normalize_image(image):
    """
    Normalizes the RGB pixel values of an image.

    Args:
        image (3D NumPy array): For example, it should take in a single 3x32x32 image from the CIFAR-10 dataset
    Returns:
        tuple: The normalized image
    """
    new_tuple = []
    for i in range(3):
        aver = np.mean(image[i])
        std = np.std(image[i])
        new_tuple.append((image[i] - aver) / std)
    
    return np.array(new_tuple)

############################################################
# Hyperparameterized Convolutional Neural Network
############################################################
class HyperParamsConvNN(nn.Module):
    def __init__(self, kernel_size=3, img_size=32):
        super(HyperParamsConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, kernel_size, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 16, kernel_size, 1, 0)
        self.fc1 = nn.Linear(16 * 10 * 10, 100)
        self.fc2 = nn.Linear(100, 72)
        self.fc3 = nn.Linear(72, 10)  

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        
        return out


############################################################
# Hyperparameterized Feed Forward Neural Network
############################################################
class HyperParamsFeedForwardNN(nn.Module):
    def __init__(self):
        super(HyperParamsFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(3072, 1300)
        self.fc2 = nn.Linear(1300, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        return out
      