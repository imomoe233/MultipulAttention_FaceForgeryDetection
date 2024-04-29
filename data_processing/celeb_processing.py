import random
import sys
import zipfile
import os
from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image
import json
import shutil
import dlib
from imutils.face_utils import FaceAligner, rect_to_bb


# 加载dlib的人脸检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # 替换为你的shape_predictor路径
predictor = dlib.shape_predictor(predictor_path)
folder_path = 'F:\FaceForensics++\data\\vedio'
output_path = 'F:\FaceForensics++\data\\face'
# 初始化面部对齐器
fa = FaceAligner(predictor, desiredFaceWidth=1024)

def get_frame(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(3 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_indices = [random.randint(start_frame, total_frames) for _ in range(800)]

    frame_count = start_frame
    frame = []
    while frame_count < total_frames:
        ret, frame_tmp = cap.read()

        if frame_count in frame_indices:
            frame.append(frame_tmp)

        frame_count += 1

        if frame_count > max(frame_indices):
            break

    cap.release()

    return frame


device = 'cuda:0'
face_detector = MTCNN(select_largest=True)



file_list = os.listdir(folder_path)
print(file_list)
print(file_list[782:])
file_list = file_list[782:]
#sys.exit()
count_frame = 0
num = 0
num_total = len(file_list)
for file in file_list:
    video_name = os.path.splitext(file)[0]
    print(f'# [{num} / {num_total}] 提取帧从 {file}')
    frame = get_frame(os.path.join(folder_path, file))

    continue_flag = 0
    count_frame_temp = count_frame

    for i in range(0, len(frame)):
        if continue_flag == 120:
            break
        try:
            bboxes, prob = face_detector.detect(frame[i])
            w0, h0, w1, h1 = bboxes[0]

        except:
            #print("Could not detect faces in the image")
            continue

        hc, wc = (h0+h1)/2, (w0+w1)/2
        crop = int(((h1-h0) + (w1-w0)) /2/2 *1.5)
        frame[i] = np.pad(frame[i], ((crop,crop),(crop,crop),(0,0)), mode='edge')  # allow cropping outside by replicating borders
        h0 = int(hc-crop+crop + crop*0.15)
        w0 = int(wc-crop+crop)
        face = frame[i][h0:h0+crop*2, w0:w0+crop*2]

        if face is not None:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face).convert("RGB")
            face = face.resize((1024, 1024))
            np_face = np.array(face)
            gray = cv2.cvtColor(np_face, cv2.COLOR_RGB2GRAY)
            # 使用检测器检测图像中的人脸
            faces = detector(gray)

            # 如果没有检测到人脸，跳过这张图片
            if len(faces) == 0:
                #print(f"No faces detected. Skipping...")
                continue

            # 获取面部标志
            shape = predictor(gray, faces[0])

            # 对齐并保存人脸
            aligned_face = fa.align(np_face, gray, faces[0])
            aligned_face = np.array(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
            save_filename = f"{video_name}_{i}.png"
            cv2.imwrite(f'{output_path}/{save_filename}.png', aligned_face)

            count_frame += 1
            continue_flag += 1
        else:
            print("无法进行颜色空间转换，因为源图像为空。")
    num += 1
    print(f'这个视频保存了{count_frame-count_frame_temp}张脸')
