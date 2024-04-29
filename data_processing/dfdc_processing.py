import random
import zipfile
import os
from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image
import json
import shutil

def get_frame(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #middle_frame_index = total_frames // 2
    #frame_indices = [middle_frame_index-500, middle_frame_index-400, middle_frame_index-300, middle_frame_index-200, middle_frame_index-100, middle_frame_index, middle_frame_index+100, middle_frame_index+200, middle_frame_index+300, middle_frame_index+400, middle_frame_index+500]
    
    # 上限和下限
    lower_limit = 0
    upper_limit = total_frames

    # 创建包含20个元素的列表，每个元素在上限和下限之间
    frame_indices = [random.randint(lower_limit, upper_limit) for _ in range(20)]

    frame_count = 0
    frame = []
    while frame_count < total_frames:
        ret, frame_tmp = cap.read()
        
        if frame_count in frame_indices:
            frame.append(frame_tmp)
        
        frame_count += 1

        if frame_count > max(frame_indices):
            break

    # 释放视频捕捉对象
    cap.release()
    
    return frame


device = 'cuda:0'
face_detector = MTCNN(select_largest=True, device=device)

folder_path = 'F:\datasets\Face\DFDC\DFDC_zip_org_data'

file_list = os.listdir(folder_path)
count_frame = 5000

for file in file_list:
    '''
    if file == 'dfdc_train_part_00.zip':
        print(f"pass {file}")
        continue
    '''
    print(f'当前处理{file}')
    zip_file_path = 'F:\\datasets\\Face\\DFDC\\DFDC_zip_org_data\\' + file
    output_dir = 'E:\code_xwd\dataset\DFDC\\temp\\'
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        #print('提取'+zip_file_path)
        #print('保存'+output_dir)
        zip_file_list = zip_file.namelist()
        #print(zip_file_list)
        zip_file.extractall(output_dir)
        items = os.listdir(output_dir)
        # JSON 文件路径
        json_file_path = f'{output_dir}{items[0]}/metadata.json'

        # 打开 JSON 文件并加载数据
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    count_file = 0
    
    for sub_file in zip_file_list:
        parts = sub_file.split('/')
        filename = parts[1]
        
        print(f'{count_file} / {len(zip_file_list)} + {parts[0]}')
        if sub_file[-4:] == '.mp4':
            
            #print('读取文件夹中的视频' + output_dir + sub_file)
            print(f'# 提取帧从 {output_dir + sub_file}')
            frame = get_frame(output_dir + sub_file)
            
            print('# 抽脸保存')
            video_data = data[filename]
            label_mp4 = video_data['label']
            
                
            count_file += 1
            continue_flag = 0
            count_frame_temp = count_frame
            
            for i in range(0, len(frame)):
                if continue_flag == 2:
                    print('已抽两张 跳过')
                    continue 
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
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = Image.fromarray(face).convert("RGB")
                face = face.resize((1024, 1024))
                if label_mp4 == 'REAL':
                    face.save(f'E:\code_xwd\dataset\DFDC/train/real/{count_frame}.jpg')
                else:
                    face.save(f'E:\code_xwd\dataset\DFDC/train/fake/{count_frame}.jpg')
                count_frame += 1
                continue_flag += 1
            print(f'这个视频保存了{count_frame-count_frame_temp}张脸')
    # 清空 output_dir 文件夹中的文件
    shutil.rmtree(output_dir + parts[0])
