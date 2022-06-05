import cv2
import json
import os
from preprocess import *
import random
from resnet import resnet18
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 获取训练数据所需文件夹或者文件名
ori_path = './training_data'
trans_path = './tmp/training_trans'
point_path = './tmp/training_point'
rect_21_path = './tmp/training_21_rect'
small_21_path = './tmp/training_21_small'
rect_7_path = './result/segmentation'
small_7_path = './tmp/training_7_small'

number_path = './tmp/total_number'
alpha_path = './tmp/total_alpha'
label_number_path = './label/total_label_number.txt'
label_alpha_path = './label/total_label_alpha.txt'

train_number_path = './train/train_number'
train_alpha_path = './train/train_alpha'
valid_number_path = './valid/valid_number'
valid_alpha_path = './valid/valid_alpha'

# 训练数据参数设置
num_epochs = 10
batch_size_train = 4
alpha_epochs = 50
lr = 5e-4
momentum = 0.5
interval = 10


# 获取数字数据集
class NumberDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.images = os.listdir(img_path)
        self.img_num = len(self.images)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_index = self.images[index]
        img_path = os.path.join(self.img_path, img_index)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img_name = img_index.split('.')[0]
        label = self.label_dict[img_name]
        return img, label


# 获取字母数据集
class AlphaDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.images = os.listdir(img_path)
        self.img_num = len(self.images)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_index = self.images[index]
        img_path = os.path.join(self.img_path, img_index)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (28, 28))
        img_name = img_index.split('.')[0]
        label = self.label_dict[img_name]
        return img, label


# 交叉验证时获取训练dataloader
def get_train_loader(ori_path, train_path, label_path, dataset):
    number = len(os.listdir(ori_path))
    data_list = list(os.listdir(ori_path))
    random_data_list = list(random.sample(data_list, int(0.8*number)))
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    else:
        shutil.rmtree(train_path)
        os.mkdir(train_path)
    for img_name in random_data_list:
        shutil.copy(os.path.join(ori_path, img_name), os.path.join(train_path, img_name))
    train_set = dataset(train_path, label_path)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True)
    return train_loader, random_data_list


# 交叉验证时获取验证dataloader
def get_valid_loader(ori_path, valid_path, label_path, dataset, random_data_list):
    data_list = list(os.listdir(ori_path))
    valid_data_list = list(set(data_list) - set(random_data_list))
    if not os.path.exists(valid_path):
        os.mkdir(valid_path)
    else:
        shutil.rmtree(valid_path)
        os.mkdir(valid_path)
    for img_name in valid_data_list:
        shutil.copy(os.path.join(ori_path, img_name), os.path.join(valid_path, img_name))
    valid_set = dataset(valid_path, label_path)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, drop_last=True)
    return valid_loader


# 训练数字的网络，训练字母使用resnet81
class Net_number(nn.Module):

    def __init__(self):
        super(Net_number, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.to(torch.float32)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


net_number = Net_number()
net_alpha = resnet18(num_classes=26)
opt_number = optim.SGD(net_number.parameters(), lr=lr, momentum=momentum)
opt_alpha = optim.SGD(net_alpha.parameters(), lr=lr, momentum=momentum)

train_number_loss = []
train_number_cnt = []
valid_number_loss = []
valid_number_cnt = []


# 训练数字
def train_number():
    for epoch in range(num_epochs):
        net_number.train()
        # 交叉验证，每次随机选取80%的数据用来训练，其余用来验证
        train_number_loader, random_list = get_train_loader(number_path, train_number_path, label_number_path, NumberDataset)
        valid_number_loader = get_valid_loader(number_path, valid_number_path, label_number_path, NumberDataset, random_list)
        for idx, (img, label) in enumerate(train_number_loader):
            opt_number.zero_grad()
            output = net_number(img)
            label = torch.Tensor(list(map(eval, label))).to(torch.long)
            loss = F.cross_entropy(output, label)
            loss.backward()
            opt_number.step()
            if (idx+1) % interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx*len(img), len(train_number_loader.dataset),
                    100.*idx/len(train_number_loader), loss.item()
                ))
                train_number_loss.append(loss.item())
                train_number_cnt.append(idx*batch_size_train+((epoch-1)*len(train_number_loader.dataset)))
                torch.save(net_number.state_dict(), './model/model_number.pth')
        
        net_number.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for img, label in valid_number_loader:
                output = net_number(img)
                label = torch.Tensor(list(map(eval, label))).to(torch.long)
                valid_loss += F.cross_entropy(output, label, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).sum()
        valid_loss /= len(valid_number_loader.dataset)
        valid_number_loss.append(valid_loss)
        print('\nValid set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, len(valid_number_loader.dataset), 
        100.*correct/len(valid_number_loader.dataset)
        ))
        result_number = correct/len(valid_number_loader.dataset)
    return result_number


train_alpha_loss = []
train_alpha_cnt = []
valid_alpha_loss = []
valid_alpha_cnt = []


# 训练字母
def train_alpha():
    for epoch in range(alpha_epochs):
        net_alpha.train()
        # 交叉验证，每次选取80%的数据用来训练，其余用来验证
        train_alpha_loader, random_list = get_train_loader(alpha_path, train_alpha_path, label_alpha_path, AlphaDataset)
        valid_alpha_loader = get_valid_loader(alpha_path, valid_alpha_path, label_alpha_path, AlphaDataset, random_list)
        for idx, (img, label) in enumerate(train_alpha_loader):
            opt_alpha.zero_grad()
            img = img.permute(0, 3, 1, 2)
            output = net_alpha(img)
            loss = F.cross_entropy(output, label)
            loss.backward()
            opt_alpha.step()
            if (idx+1) % interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx*len(img), len(train_alpha_loader.dataset),
                    100.*idx/len(train_alpha_loader), loss.item()
                ))
                train_alpha_loss.append(loss.item())
                train_alpha_cnt.append(idx*batch_size_train+((epoch-1)*len(train_alpha_loader.dataset)))
                torch.save(net_alpha.state_dict(), './model/model_alpha.pth')
        
        net_alpha.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for img, label in valid_alpha_loader:
                img = img.permute(0, 3, 1, 2)
                output = net_alpha(img)
                valid_loss += F.cross_entropy(output, label, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).sum()
        valid_loss /= len(valid_alpha_loader.dataset)
        valid_alpha_loss.append(valid_loss)
        print('\nValid set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, len(valid_alpha_loader.dataset), 
        100.*correct/len(valid_alpha_loader.dataset)
        ))
        result_alpha = correct/len(valid_alpha_loader.dataset)
    return result_alpha



if __name__ == '__main__':

    box_21_total, box_7_total = rectangle(ori_path, trans_path, point_path, rect_21_path, small_21_path, rect_7_path, small_7_path)
    segamentation(box_21_total, box_7_total, small_21_path, small_7_path, number_path, alpha_path, label_number_path, label_alpha_path)
    
    result_number = train_number()
    result_alpha = train_alpha()
    print('[Valid]: result_number: {}'.format(result_number))
    print('[Valid]: result_alpha: {}'.format(result_alpha))