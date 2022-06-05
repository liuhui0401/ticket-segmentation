from preprocess import *

ori_path = './training_data'
trans_path = './tmp/training_trans'
point_path = './tmp/training_point'
rect_21_path = './tmp/training_21_rect'
small_21_path = './tmp/training_21_small'
rect_7_path = './result/segmentation'
small_7_path = './tmp/training_7_small'


# 获取识别和标注21位码和7位码的结果，存入segmentation文件夹中
box_21_total, box_7_total = rectangle(ori_path, trans_path, point_path, rect_21_path, small_21_path, rect_7_path, small_7_path)