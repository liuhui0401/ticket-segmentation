import cv2
import cmath
import json
import numpy as np
import os


# 读取图片数据
def read_data(path):
    data = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        data.append(img)
    return data


# 检测原始图片的轮廓，转正图片
def detection(img): 
    # 扩充图片，利于轮廓检测
    expand = 200
    img = cv2.copyMakeBorder(img, expand, expand, expand, expand, cv2.BORDER_CONSTANT, (0, 0, 0))
    
    # 先进行高斯模糊、膨胀、边缘检测
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_dilate = cv2.dilate(img_blur, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    img_edge = cv2.Canny(img_dilate, 50, 150, 3)
    # 再次进行膨胀，把细小的边缘模糊
    img_edge = cv2.dilate(img_edge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 进行车票轮廓检测
    contours, _ = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rectangle = cv2.minAreaRect(np.array(contours[0].squeeze()))
    box = np.int0(cv2.boxPoints(rectangle))
    
    # 在原始未翻转图片中绘制车票轮廓
    # draw_img = img.copy()
    # color = (238, 44, 44)
    # linewidth = 2
    # cv2.line(draw_img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, linewidth)
    # cv2.line(draw_img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color, linewidth)
    # cv2.line(draw_img, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color, linewidth)
    # cv2.line(draw_img, (box[3][0], box[3][1]), (box[0][0], box[0][1]), color, linewidth)

    # 判断车票的长短边，计算角度
    dist1 = np.sum((box[0]-box[1])**2)
    dist2 = np.sum((box[0]-box[3])**2)
    angle = rectangle[2]
    if dist1 > dist2:
        angle = 90 - angle

    # 做透视变换
    new_height, new_width = 540, 860
    if dist1 > dist2:
        dst = np.array([[new_width-6, new_height-6], [5, new_height-6], [5, 5], [new_width-6, 5]], dtype=np.float32)
    else:
        dst = np.array([[5, new_height-6], [5, 5], [new_width-6, 5], [new_width-6, new_height-6]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    img_trans = cv2.warpPerspective(img, M, (new_width, new_height), borderValue=(0, 0, 0))

    # 针对二维码模糊的图片进行数据增强
    tmp = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)
    tmp = cv2.erode(tmp, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    tmp = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80)))
    _, img_thres = cv2.threshold(tmp, 100, 255, cv2.THRESH_BINARY)

    # 根据图片左上角和右下角的灰度值判断二维码的位置，校正票面位置
    left_up = np.sum(img_thres[:int(new_height/2), :int(new_width/4)])
    right_down = np.sum(img_thres[int(new_height/2):, int(3*new_width/4):])
    if left_up < right_down:
        M = cv2.getRotationMatrix2D((new_width/2, new_height/2), 180, 1.0)
        img_trans = cv2.warpAffine(img_trans, M, (new_width, new_height), borderValue=(0, 0, 0))

    return img_trans


# 计算顶点的极坐标幅角
def get_theta(point, middle):
    x, y = point[0], point[1]
    x_m, y_m = middle[0], middle[1]
    _,  theta = cmath.polar(complex(x-x_m, y-y_m))
    return theta


# 按顺序找到转正票面的四个顶点
def find_point(img):
    # 先进行高斯模糊、膨胀、边缘检测
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_dilate = cv2.dilate(img_blur, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    img_edge = cv2.Canny(img_dilate, 50, 150, 3)
    # 再次进行膨胀，把细小的边缘模糊
    img_edge = cv2.dilate(img_edge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 绘制票面轮廓
    draw_img = img.copy()
    cv2.drawContours(draw_img, contours[0], -1, (0,0,255), 3)
    rectangle = cv2.minAreaRect(np.array(contours[0].squeeze()))
    box = np.int0(cv2.boxPoints(rectangle))
    box_list = box.tolist()
    # 按照顺序对四个顶点排序
    middle = np.mean(box_list, axis=0)
    for point in box_list:
        point.append(get_theta(point, middle))
    box_list = sorted(box_list, key=lambda x: -x[2])
    # 删除顶点极坐标计数器
    for point in box_list:
        del point[2]
    # 用不同颜色按照顺序绘制四个顶点
    cv2.circle(draw_img, tuple(box_list[0]), 30, (255,0,0), -1) 
    cv2.circle(draw_img, tuple(box_list[1]), 30, (255,255,0), -1)
    cv2.circle(draw_img, tuple(box_list[2]), 30, (0,255,255), -1)
    cv2.circle(draw_img, tuple(box_list[3]), 30, (0,255,0), -1)

    return draw_img, box_list


# 识别并标注21位码
def rect_21(img, point_box):
    # 选择左下角的顶点作为参考点
    ref_point = point_box[0]
    # 选出21位码的大致区域
    down, up = ref_point[1]-80, ref_point[1]-15
    left, right = ref_point[0]+35, ref_point[0]+410

    # 在原图中只保留21位码区域
    mask = np.zeros_like(img)
    mask[down:up, left:right] = 1
    m_img = (255 - np.clip(img*mask + 255*(mask==0), 0, 255)).astype(np.uint8)
    m_img = np.mean(m_img, axis=-1).astype(np.uint8)
    _, m_img = cv2.threshold(m_img, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作将21位码区域连通后选出
    _, img_thres = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    img_edge = cv2.Canny(img_thres, img_thres.shape[0], img_thres.shape[1])
    kernel = np.ones((5, 19), dtype=np.uint8)
    img_close = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((19, 5), dtype=np.uint8)
    img_open = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对形态学操作后的图片，选择在21位码范围内检测出的边框
    for i in range(len(contours)):
        rectangle = cv2.minAreaRect(np.array(contours[i].squeeze()))
        box = np.int0(cv2.boxPoints(rectangle))
        flag = 0
        for j in range(len(box)):
            if box[j][0] >= left and box[j][0] <= right and box[j][1] >= down and box[j][1] <= up:
                if j == len(box)-1:
                    flag = 1
            else:
                break
        if flag == 1:
            break

    # 将21位码的四个顶点排序，因为边框太紧，进行扩充
    box_list = box.tolist()
    middle = np.mean(box_list, axis=0)
    for point in box_list:
        point.append(get_theta(point, middle))
    box_list = sorted(box_list, key=lambda x: -x[2])
    expand = 6
    box_list[0][0] -= expand
    box_list[0][1] += expand
    box_list[1][0] += expand
    box_list[1][1] += expand
    box_list[2][0] += expand
    box_list[2][1] -= expand
    box_list[3][0] -= expand
    box_list[3][1] -= expand

    # 按顺序描绘出四个顶点，绘制21位码的边框
    draw_img = img.copy()
    color = (238, 44, 44)
    linewidth = 1
    cv2.line(draw_img, (box_list[0][0], box_list[0][1]), (box_list[1][0], box_list[1][1]), color, linewidth)
    cv2.line(draw_img, (box_list[1][0], box_list[1][1]), (box_list[2][0], box_list[2][1]), color, linewidth)
    cv2.line(draw_img, (box_list[2][0], box_list[2][1]), (box_list[3][0], box_list[3][1]), color, linewidth)
    cv2.line(draw_img, (box_list[3][0], box_list[3][1]), (box_list[0][0], box_list[0][1]), color, linewidth)

    # 对21位码进行分割并且描绘出分割线
    box_left, box_up, box_right, box_down = box_list[0][0], box_list[0][1], box_list[2][0], box_list[2][1]
    box_line = []
    box_line.append(box_left)
    img_new_thres = cv2.bitwise_not(img_thres)
    kernel = np.ones((1, 1), dtype=np.uint8)
    img_new_thres = cv2.morphologyEx(img_new_thres, cv2.MORPH_CLOSE, kernel)
    left, right = box_left, box_left
    while left < box_right:
        right = left
        while np.sum(img_new_thres[box_down:box_up, right]) == 0:
            right += 1
        while np.sum(img_new_thres[box_down:box_up, right]) != 0:
            right += 1
        if right < box_right:
            box_line.append(right+2)
        left = right
    del box_line[-1]
    box_line.append(box_right)
    for i in range(len(box_line)):
        cv2.line(draw_img, (box_line[i], box_down), (box_line[i], box_up), color, linewidth)
    box_line.append(box_list)

    return draw_img, m_img, box_line
    

# 识别并标注7位码
def rect_7(img, point_box):
    # 选择左上方的顶点作为参考点
    ref_point = point_box[3]
    # 选择出7位码的大致区域
    down, up = ref_point[1]+20, ref_point[1]+120
    left, right = ref_point[0]+10, ref_point[0]+300

    # 通过两个阈值将7位码分离出来
    mask = np.zeros_like(img)
    mask[down:up, left:right] = 1
    m_img = (255 - np.clip(img*mask + 255*(mask==0), 0, 255)).astype(np.uint8)
    m_img = np.mean(m_img, axis=-1).astype(np.uint8)
    white_pos = np.where(m_img > 220)
    m_img[white_pos] = 0
    m_img = cv2.medianBlur(m_img, 3)
    m_img = cv2.GaussianBlur(m_img, (3, 3), 0)
    _, img_thres = cv2.threshold(m_img, 120, 255, cv2.THRESH_BINARY)

    # 框选出7位码区域
    img_edge = cv2.Canny(img_thres, img_thres.shape[0], img_thres.shape[1])
    kernel = np.ones((5, 19), dtype=np.uint8)
    img_close = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_close = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rectangle = cv2.minAreaRect(np.array(contours[0].squeeze()))
    box = np.int0(cv2.boxPoints(rectangle))

    # 因为7位码区域太紧，进行扩充并绘制边框
    box_list = box.tolist()
    middle = np.mean(box_list, axis=0)
    for point in box_list:
        point.append(get_theta(point, middle))
    box_list = sorted(box_list, key=lambda x: -x[2])
    expand = 6
    box_list[0][0] -= expand
    box_list[0][1] += expand
    box_list[1][0] += expand
    box_list[1][1] += expand
    box_list[2][0] += expand
    box_list[2][1] -= expand
    box_list[3][0] -= expand
    box_list[3][1] -= expand
    draw_img = img.copy()
    color = (238, 44, 44)
    linewidth = 1
    cv2.line(draw_img, (box_list[0][0], box_list[0][1]), (box_list[1][0], box_list[1][1]), color, linewidth)
    cv2.line(draw_img, (box_list[1][0], box_list[1][1]), (box_list[2][0], box_list[2][1]), color, linewidth)
    cv2.line(draw_img, (box_list[2][0], box_list[2][1]), (box_list[3][0], box_list[3][1]), color, linewidth)
    cv2.line(draw_img, (box_list[3][0], box_list[3][1]), (box_list[0][0], box_list[0][1]), color, linewidth)
    
    # 对7位码进行分割
    box_left, box_up, box_right, box_down = box_list[0][0], box_list[0][1], box_list[2][0], box_list[2][1]
    box_line = []
    box_line.append(box_right)
    left, right = box_right, box_right
    while right > box_left:
        left = right
        while np.sum(img_thres[box_down:box_up, left]) <= 510:
            left -= 1
        while np.sum(img_thres[box_down:box_up, left]) > 510:
            left -= 1
        if left > box_left and right-left > 10:
            box_line.append(left-2)
        right = left
    del box_line[-1]
    box_line.append(box_left)
    for i in range(len(box_line)):
        cv2.line(draw_img, (box_line[i], box_down), (box_line[i], box_up), color, linewidth)
    box_line.append(box_list)

    return draw_img, img_thres, box_line


# 读取训练数据标签，对21位码和7位码进行分割，获取数字和字母的训练数据
def get_seg(box_21_total, box_7_total, small_21_path, small_7_path, number_path, alpha_path, label_number_path, label_alpha_path):
    # 创建分别存储数字和字母训练数据的文件夹
    if not os.path.exists(number_path):
        os.mkdir(number_path)
    if not os.path.exists(alpha_path):
        os.mkdir(alpha_path)

    # 读取训练数据标签
    img_name_dict, total_21, total_7 = {}, [], []
    with open('./label/annotation.txt', 'r') as f:
        index = 0
        for line in f.readlines():
            total_21.append(line.split(' ')[1])
            total_7.append(line.split(' ')[2])
            img_name_dict[line.split(' ')[0]] = index
            index += 1

    # 读取21位码的训练数据
    number_index, alpha_index = 0, 0
    label_number, label_alpha = {}, {}
    i_21 = 0
    for img_name in os.listdir(small_21_path):
        small_number_list, small_alpha_list = [], []
        label_n_list, label_a_list = [], []
        box_left, box_up, box_right, box_down = box_21_total[i_21][-1][0][0], box_21_total[i_21][-1][0][1], box_21_total[i_21][-1][2][0], box_21_total[i_21][-1][2][1]
        img_path = os.path.join(small_21_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 获取21位码每一位的对应标签
        index = img_name_dict[img_name]
        for i in range(14):
            label_n_list.append(total_21[index][i])
        label_a_list.append(total_21[index][14])
        for i in range(15, 21):
            label_n_list.append(total_21[index][i])
        # 分割21位码的每一位
        for i in range(14):
            small_img = img[box_down:box_up, box_21_total[i_21][i]:box_21_total[i_21][i+1]]
            small_number_list.append(small_img)
        small_img = img[box_down:box_up, box_21_total[i_21][14]:box_21_total[i_21][15]]
        small_alpha_list.append(small_img)
        for i in range(15, 21):
            small_img = img[box_down:box_up, box_21_total[i_21][i]:box_21_total[i_21][i+1]]
            small_number_list.append(small_img)
        i_21 += 1
        # 将21位码的每一位递增存储
        for i in range(len(small_number_list)):
            cv2.imwrite(os.path.join(number_path, str(number_index)+'.bmp'), small_number_list[i])
            label_number[number_index] = label_n_list[i]
            number_index += 1
        for i in range(len(small_alpha_list)):
            cv2.imwrite(os.path.join(alpha_path, str(alpha_index)+'.bmp'), small_alpha_list[i])
            label_alpha[alpha_index] = ord(label_a_list[i])-65
            alpha_index += 1
    
    # 读取7位码的训练数据
    i_7 = 0
    for img_name in os.listdir(small_7_path):
        small_number_list, small_alpha_list = [], []
        label_n_list, label_a_list = [], []
        box_left, box_up, box_right, box_down = box_7_total[i_7][-1][0][0], box_7_total[i_7][-1][0][1], box_7_total[i_7][-1][2][0], box_7_total[i_7][-1][2][1]
        img_path = os.path.join(small_7_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 获取7位码的每一位的对应标签
        index = img_name_dict[img_name]
        label_a_list.append(total_7[index][0])
        for i in range(1, 7):
            label_n_list.append(total_7[index][i])
        # 分割7位码的每一位
        small_img = img[box_down:box_up, box_7_total[i_7][7]:box_7_total[i_7][6]]
        small_alpha_list.append(small_img)
        for i in range(6, 0, -1):
            small_img = img[box_down:box_up, box_7_total[i_7][i]:box_7_total[i_7][i-1]]
            small_number_list.append(small_img)
        i_7 += 1
        # 将7位码的每一位递增存储
        for i in range(len(small_number_list)):
            cv2.imwrite(os.path.join(number_path, str(number_index)+'.bmp'), small_number_list[i])
            label_number[number_index] = label_n_list[i]
            number_index += 1
        for i in range(len(small_alpha_list)):
            cv2.imwrite(os.path.join(alpha_path, str(alpha_index)+'.bmp'), small_alpha_list[i])
            label_alpha[alpha_index] = ord(label_a_list[i])-65
            alpha_index += 1
    # 存储数字和字母每一位的标签
    with open(label_number_path, 'w') as f:
        json.dump(label_number, f)
    with open(label_alpha_path, 'w') as f:
        json.dump(label_alpha, f)


# 训练集中21位码和7位码的数字和字母都一起存储进行训练，测试数据集需要区分21位码和7位码的数字和字母，才能获取对应输出，因此单独处理测试数据
def get_seg_test(box_21_total, box_7_total, small_21_path, small_7_path, number_21_path, alpha_21_path, number_7_path, alpha_7_path):
    # 创建存储21位码和7位码的数字和字母的四个文件夹
    if not os.path.exists(number_21_path):
        os.mkdir(number_21_path)
    if not os.path.exists(alpha_21_path):
        os.mkdir(alpha_21_path)
    if not os.path.exists(number_7_path):
        os.mkdir(number_7_path)
    if not os.path.exists(alpha_7_path):
        os.mkdir(alpha_7_path)
    
    # 获取21位码的数字和字母
    number_21_index, alpha_21_index, number_7_index, alpha_7_index = 0, 0, 0, 0
    i_21 = 0
    for img_name in os.listdir(small_21_path):
        small_number_list, small_alpha_list = [], []
        box_left, box_up, box_right, box_down = box_21_total[i_21][-1][0][0], box_21_total[i_21][-1][0][1], box_21_total[i_21][-1][2][0], box_21_total[i_21][-1][2][1]
        img_path = os.path.join(small_21_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(14):
            small_img = img[box_down:box_up, box_21_total[i_21][i]:box_21_total[i_21][i+1]]
            small_number_list.append(small_img)
        small_img = img[box_down:box_up, box_21_total[i_21][14]:box_21_total[i_21][15]]
        small_alpha_list.append(small_img)
        for i in range(15, 21):
            small_img = img[box_down:box_up, box_21_total[i_21][i]:box_21_total[i_21][i+1]]
            small_number_list.append(small_img)
        i_21 += 1
        for i in range(len(small_number_list)):
            cv2.imwrite(os.path.join(number_21_path, str(number_21_index)+'.bmp'), small_number_list[i])
            number_21_index += 1
        for i in range(len(small_alpha_list)):
            cv2.imwrite(os.path.join(alpha_21_path, str(alpha_21_index)+'.bmp'), small_alpha_list[i])
            alpha_21_index += 1
    
    # 获取7位码的数字和字母
    i_7 = 0
    for img_name in os.listdir(small_7_path):
        small_number_list, small_alpha_list = [], []
        box_left, box_up, box_right, box_down = box_7_total[i_7][-1][0][0], box_7_total[i_7][-1][0][1], box_7_total[i_7][-1][2][0], box_7_total[i_7][-1][2][1]
        img_path = os.path.join(small_7_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        small_img = img[box_down:box_up, box_7_total[i_7][7]:box_7_total[i_7][6]]
        small_alpha_list.append(small_img)
        for i in range(6, 0, -1):
            small_img = img[box_down:box_up, box_7_total[i_7][i]:box_7_total[i_7][i-1]]
            small_number_list.append(small_img)
        i_7 += 1
        for i in range(len(small_number_list)):
            cv2.imwrite(os.path.join(number_7_path, str(number_7_index)+'.bmp'), small_number_list[i])
            number_7_index += 1
        for i in range(len(small_alpha_list)):
            cv2.imwrite(os.path.join(alpha_7_path, str(alpha_7_index)+'.bmp'), small_alpha_list[i])
            alpha_7_index += 1