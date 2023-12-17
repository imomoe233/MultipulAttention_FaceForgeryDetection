from mtcnn import MTCNN
import cv2
import random
import os
import numpy as np
import dlib
import csv

def estimate_blur(image):
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的Laplacian（拉普拉斯变换）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 计算Laplacian的方差
    variance = laplacian.var()

    return variance

def extract_random_frames(video_path, num_frames=10):
    num_frames += 1
    
    # 创建视频捕捉对象
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return None

    # 获取视频的总帧数
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("Error: The video has no frames.")
        return None

    frames = []  # 用于存储抽取的帧

    # 随机选择多个帧编号
    random_frame_numbers = random.sample(range(0, total_frames), num_frames) if total_frames > num_frames else range(0, total_frames)

    for frame_number in random_frame_numbers:
        # 设置视频位置
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # 读取当前帧
        success, frame = video.read()
        if not success:
            print(f"Error: Could not extract frame {frame_number}.")
            continue  # 如果无法读取帧，则继续下一个

        frames.append(frame)

    # 释放视频对象
    video.release()

    return frames

def extract_and_save_face(i, frame, result, save_path, padding_percentage=0.2):
    """
    从给定的帧中提取脸部并保存。

    参数:
        frame: 包含脸部的图像帧。
        result: MTCNN 检测到的脸部结果（单个脸部，为字典格式）。
        save_path: 保存脸部图像的完整路径。
        padding_percentage: 用于扩展边界框的百分比，值为0到1之间。
    """
    # 如果没有检测到脸部，直接返回
    if not result:
        return

    x, y, width, height = result['box']
    # 计算扩展边界的像素数
    x_padding = int(width * padding_percentage)
    y_padding = int(height * padding_percentage)

    # 计算带有额外空间的新边界框
    x1, y1 = max(x - x_padding, 0), max(y - y_padding, 0)  # 避免负坐标
    x2, y2 = min(x + width + x_padding, frame.shape[1]), min(y + height + y_padding, frame.shape[0])  # 避免超出图像边界

    # 提取带有背景的脸部区域
    face_with_background = frame[y1:y2, x1:x2]

    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        #print(f"Directory '{save_path}' created")
    #print(estimate_blur(face_with_background))
    #if estimate_blur(face_with_background) >= 60:
    #    cv2.imwrite(save_path + f'{i}.png', face_with_background)
    cv2.imwrite(save_path + f'{i}.png', face_with_background)

def get_most_centered_face(faces, frame_width, frame_height):
    """
    在多个检测到的脸中找出最中间的一张。

    参数:
        faces: MTCNN 检测到的脸部信息列表。
        frame_width: 帧的宽度。
        frame_height: 帧的高度。

    返回:
        最中心脸的信息。
    """
    if not faces:
        return None

    image_center = np.array([frame_width / 2, frame_height / 2])

    most_centered_face = None
    smallest_distance = float('inf')

    for face in faces:
        box = face['box']
        # 计算脸部边界框的中心点
        face_center_x = box[0] + box[2] / 2
        face_center_y = box[1] + box[3] / 2
        face_center = np.array([face_center_x, face_center_y])

        # 计算脸部中心与图像中心的欧氏距离
        distance = np.linalg.norm(face_center - image_center)

        # 选择距离最小（即最中心）的脸
        if distance < smallest_distance:
            most_centered_face = face
            smallest_distance = distance

    return most_centered_face


file_name = 'F:\datasets\Face\DFDC\original.csv'
file_list = []

detector = dlib.get_frontal_face_detector()

with open(file_name, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        file_list.append(row[1])

for individual_file in file_list:
    path_to_video = 'F:\datasets\Face\DFDC\original/' + individual_file  # 修改为你的视频文件路径
    print(path_to_video)
    random_frame = extract_random_frames(path_to_video, num_frames=5)

    faces = []

    for frame in random_frame:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用 dlib 检测人脸
        detections = detector(img_rgb, 1)  # 1 表示将图像上采样一次，可以增加检测的准确性
        
        # 转换 dlib 的检测结果格式，使其与之前的代码相兼容
        current_faces = [{'box': [d.left(), d.top(), d.width(), d.height()]} for d in detections]
        
        faces.append(current_faces)

    for i in range(len(random_frame)):
        frame_height, frame_width = random_frame[i].shape[:2]
        centered_face = get_most_centered_face(faces[i], frame_width, frame_height)
        extract_and_save_face(i, random_frame[i], centered_face, 'F:\datasets\Face\DFDC\original/face/'+individual_file+'/', padding_percentage=0.2)
    
    
    
    


    