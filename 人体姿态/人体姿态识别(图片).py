#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 王鸿菲
"""
# opencv-python
import cv2
# mediapipe包
import mediapipe as mp
# 进度条库
from tqdm import tqdm
# 时间库
import time
# 导入python绘图matplotlib
import matplotlib.pyplot as plt


# 定义可视化图像函数
def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

# 导入solution
mp_pose = mp.solutions.pose

# # 导入绘图函数
mp_drawing = mp.solutions.drawing_utils 

# 导入模型
pose = mp_pose.Pose(static_image_mode=True,        # 是静态图片还是连续视频帧
                    model_complexity=1,            # 选择人体姿态关键点检测模型，0最快但性能差，2性能好但是慢
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)   # 追踪阈值

# 从图片文件读入图像，opencv读入为BGR格式
img = cv2.imread('/Users/apple/Desktop/NLP/人体姿态/test2.jpg')

# BGR转RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 将RGB图像输入模型，获取预测结果
results = pose.process(img_RGB)
h = img.shape[0]
w = img.shape[1]

if results.pose_landmarks: # 若检测出人体关键点
    
    # 可视化关键点及骨架连线
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    for i in range(33): # 遍历所有33个关键点，可视化

        # 获取该关键点的三维坐标
        cx = int(results.pose_landmarks.landmark[i].x * w)
        cy = int(results.pose_landmarks.landmark[i].y * h)
        cz = results.pose_landmarks.landmark[i].z

        radius = 10

        if i == 0: # 鼻尖
            img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
        elif i in [11,12]: # 肩膀
            img = cv2.circle(img,(cx,cy), radius, (223,155,6), -1)
        elif i in [23,24]: # 髋关节
            img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
        elif i in [13,14]: # 胳膊肘
            img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
        elif i in [25,26]: # 膝盖
            img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
        elif i in [15,16,27,28]: # 手腕和脚腕
            img = cv2.circle(img,(cx,cy), radius, (223,155,60), -1)
        elif i in [17,19,21]: # 左手
            img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
        elif i in [18,20,22]: # 右手
            img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
        elif i in [27,29,31]: # 左脚
            img = cv2.circle(img,(cx,cy), radius, (29,123,243), -1)
        elif i in [28,30,32]: # 右脚
            img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
        elif i in [9,10]: # 嘴
            img = cv2.circle(img,(cx,cy), radius, (205,235,255), -1)
        elif i in [1,2,3,4,5,6,7,8]: # 眼及脸颊
            img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
        else: # 其它关键点
            img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)

    # 展示图片
    look_img(img)
    
else:
    print('从图像中未检测出人体关键点，报错。')

# 保存图片
cv2.imwrite('W.jpg',img)