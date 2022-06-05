import cv2
import os
from function import *

# 创建分割21位码和7位码所需要的文件夹
def make_dir(trans_path, point_path, rect_21_path, small_21_path, rect_7_path, small_7_path):
    # 存储透视变换后的转正的图片
    if not os.path.exists(trans_path):
        os.makedirs(trans_path)
    # 存储按顺序标注出四个顶点和车票轮廓的做了透视变换后的图片
    if not os.path.exists(point_path):
        os.makedirs(point_path)
    # 存储分割21位码后的结果
    if not os.path.exists(rect_21_path):
        os.makedirs(rect_21_path)
    # 存储对21位码以外区域做mask操作的结果，为了训练/测试数据做准备
    if not os.path.exists(small_21_path):
        os.makedirs(small_21_path)
    # 存储分割21位码和7位码的结果
    if not os.path.exists(rect_7_path):
        os.makedirs(rect_7_path)
    # 存储对7位码以外区域做mask操作的结果，为了训练/测试数据做准备
    if not os.path.exists(small_7_path):
        os.makedirs(small_7_path)


# 对训练数据进行转正、标注轮廓和顶点、识别和标注21位码、识别和标注7位码的操作，并将对应结果存储
def rectangle(ori_path, trans_path, point_path, rect_21_path, small_21_path, rect_7_path, small_7_path):
    # 创建所需文件
    make_dir(trans_path, point_path, rect_21_path, small_21_path, rect_7_path, small_7_path)
    # 读取训练数据
    data = read_data(ori_path)
    i = 0
    box_21_total, box_7_total = [], []
    for img_name in os.listdir(ori_path):
        img = data[i]
        # 透视变换
        img_trans = detection(img)
        cv2.imwrite(os.path.join(trans_path, img_name), img_trans)
        # 标注轮廓和顶点
        point_img, point_box = find_point(img_trans)
        cv2.imwrite(os.path.join(point_path, img_name), point_img)
        # 识别和标注21位码
        rect_21_img, small_21_img, box_21 = rect_21(img_trans, point_box)
        box_21_total.append(box_21)
        cv2.imwrite(os.path.join(rect_21_path, img_name), rect_21_img)
        cv2.imwrite(os.path.join(small_21_path, img_name), small_21_img)
        # 识别和标注7位码
        rect_7_img, small_7_img, box_7 = rect_7(rect_21_img, point_box)
        box_7_total.append(box_7)
        cv2.imwrite(os.path.join(rect_7_path, img_name), rect_7_img)
        cv2.imwrite(os.path.join(small_7_path, img_name), small_7_img)
        i += 1
    
    return box_21_total, box_7_total


# 分割创建训练数据
def segamentation(box_21_total, box_7_total, small_21_path, small_7_path, number_path, alpha_path, label_number_path, label_alpha_path):
    get_seg(box_21_total, box_7_total, small_21_path, small_7_path, number_path, alpha_path, label_number_path, label_alpha_path)


# 分割创建测试数据
def test_pre(ori_path, small_21_path, small_7_path, test_number_21_path, test_alpha_21_path, test_number_7_path, test_alpha_7_path):
    if not os.path.exists(small_21_path):
        os.makedirs(small_21_path)
    if not os.path.exists(small_7_path):
        os.makedirs(small_7_path)
    data = read_data(ori_path)
    i = 0
    box_21_total, box_7_total = [], []
    for img_name in os.listdir(ori_path):
        img = data[i]
        img_trans = detection(img)
        point_img, point_box = find_point(img_trans)
        rect_21_img, small_21_img, box_21 = rect_21(img_trans, point_box)
        box_21_total.append(box_21)
        rect_7_img, small_7_img, box_7 = rect_7(rect_21_img, point_box)
        box_7_total.append(box_7)
        cv2.imwrite(os.path.join(small_21_path, img_name), small_21_img)
        cv2.imwrite(os.path.join(small_7_path, img_name), small_7_img)
        i += 1
    get_seg_test(box_21_total, box_7_total, small_21_path, small_7_path, test_number_21_path, test_alpha_21_path, test_number_7_path, test_alpha_7_path)