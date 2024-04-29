'''''
从youtube原始视频文件夹提取所有的视频文件
'''''
import os
import shutil

# 设置输入和输出目录
input_dir = 'F:\FaceForensics++\data\downloaded_videos5'
output_dir = 'F:\FaceForensics++\data\\vedio'

for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    # 遍历输入目录下的所有.mp4文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp4'):
            file_path = os.path.join(folder_path, file_name)
            output_file = os.path.join(output_dir, file_name)
            shutil.move(file_path, output_file)


