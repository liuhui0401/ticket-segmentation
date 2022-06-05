import os
from preprocess import *
from resnet import resnet18
from train import Net_number
import torch

# 获取测试数据所需文件夹和文件名
ori_path = 'test_data'
small_21_path = './tmp/test_small_21'
small_7_path = './tmp/test_small_7'
test_number_21_path = './tmp/test_21_number'
test_alpha_21_path = './tmp/test_21_alpha'
test_number_7_path = './tmp/test_7_number'
test_alpha_7_path = './tmp/test_7_alpha'
prediction = './result/prediction.txt'

# 获取测试数据
test_pre(ori_path, small_21_path, small_7_path, test_number_21_path, test_alpha_21_path, test_number_7_path, test_alpha_7_path)

number_state_dict = torch.load('./model/pre_number.pth')
alpha_state_dict = torch.load('./model/pre_alpha.pth')
net_number = Net_number()
net_number.load_state_dict(number_state_dict)
net_alpha = resnet18(num_classes=26)
net_alpha.load_state_dict(alpha_state_dict)

# 预测结果向prediction.txt输出，每次重新预测时清空txt文件内容
with open(prediction, 'a') as f:
    f.truncate()

test_number_21_result, test_alpha_21_result, test_number_7_result, test_alpha_7_result = [], [], [], []

# 对21位码的数字部分做测试
net_number.eval()
with torch.no_grad():
    files = os.listdir(test_number_21_path)
    files.sort(key= lambda x:int(x[:-4]))
    for img_name in files:
        img = cv2.imread(os.path.join(test_number_21_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = torch.from_numpy(img)
        img = img.unsqueeze_(0)
        img_name = img_name.split('.')[0]
        output = net_number(img)
        pred = output.data.max(1, keepdim=True)[1]
        test_number_21_result.append(str(pred.item()))

# 对21位码的字母部分做测试
net_alpha.eval()
with torch.no_grad():
    files = os.listdir(test_alpha_21_path)
    files.sort(key= lambda x:int(x[:-4]))
    for img_name in files:
        img = cv2.imread(os.path.join(test_alpha_21_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        img_name = img_name.split('.')[0]
        output = net_alpha(img)
        pred = output.data.max(1, keepdim=True)[1]
        test_alpha_21_result.append(str(chr(pred.item()+65)))

# 对7位码的数字部分做测试
net_number.eval()
with torch.no_grad():
    files = os.listdir(test_number_7_path)
    files.sort(key= lambda x:int(x[:-4]))
    for img_name in files:
        img = cv2.imread(os.path.join(test_number_7_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = torch.from_numpy(img)
        img = img.unsqueeze_(0)
        img_name = img_name.split('.')[0]
        output = net_number(img)
        pred = output.data.max(1, keepdim=True)[1]
        test_number_7_result.append(str(pred.item()))

# 对7位码的字母部分做测试
net_alpha.eval()
correct = 0
with torch.no_grad():
    files = os.listdir(test_alpha_7_path)
    files.sort(key= lambda x:int(x[:-4]))
    for img_name in files:
        img = cv2.imread(os.path.join(test_alpha_7_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        img_name = img_name.split('.')[0]
        output = net_alpha(img)
        pred = output.data.max(1, keepdim=True)[1]
        test_alpha_7_result.append(str(chr(pred.item()+65)))

# 将测试结果输出到prediction.txt文件中
with open(prediction, 'w') as f:
    i = 0
    for img_name in os.listdir(ori_path):
        f.write(img_name)
        f.write(' ')
        for j in range(i*20, i*20+14):
            f.write(test_number_21_result[j])
        f.write(test_alpha_21_result[i])
        for j in range(i*20+14, i*20+20):
            f.write(test_number_21_result[j])
        f.write(' ')
        f.write(test_alpha_7_result[i])
        for j in range(i*6, i*6+6):
            f.write(test_number_7_result[j])
        f.write('\n')
        i += 1