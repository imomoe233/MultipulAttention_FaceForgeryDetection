import os
import csv

# 指定要搜索视频文件的目录路径
directory_path = 'F:\datasets\Face\DFDC\FaceSwap'

# 创建一个列表来保存视频文件的名称和默认值
video_files = []

# 遍历目录中的每个文件/文件夹
for foldername, subfolders, filenames in os.walk(directory_path):
    for filename in filenames:
        # 检查文件扩展名是否为视频文件
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):  # 你可以根据需要添加更多的视频格式
            # 获取完整的文件路径
            file_path = os.path.join(foldername, filename)
            # 将文件名和对应的默认值0添加到列表中
            video_files.append([1, filename])

# 将视频文件的名称以及默认值写入CSV文件中
with open('F:\datasets\Face\DFDC\FaceSwap.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行（可选）
    #writer.writerow(['Default Value', 'Video File Name'])
    # 按行写入视频文件的名称及默认值
    for entry in video_files:
        writer.writerow(entry)

print(f'Finished! Found {len(video_files)} video files and wrote them to video_files.csv')
