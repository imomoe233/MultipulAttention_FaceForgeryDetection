import os
import csv
import shutil
from datetime import datetime
import sys

# 定义CSV文件名
csv_filename = 'F:\ECCV\data_preprocessing\processed\FF++\\test_offical.labels.csv'

# 定义源文件夹路径
source_folder = 'F:\\ECCV\data_preprocessing\\processed\\FF++\\test_offical'

# 定义目标文件夹路径
destination_folder = 'F:\ECCV\data_preprocessing\processed\FF++\\test_lap'

# 如果目标文件夹不存在，则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 打开CSV文件以写入模式
with open(csv_filename, 'w', newline='') as csvfile:
    # 创建CSV写入器
    csv_writer = csv.writer(csvfile)

    # 遍历源文件夹中的所有文件
    for file_name in os.listdir(source_folder):
        # 获取文件的完整路径
        file_path = os.path.join(source_folder, file_name)
        # 检查文件是否存在且为普通文件
        if os.path.isfile(file_path):
            # 获取文件的修改日期
            modification_time = os.path.getmtime(file_path)
            # 将修改日期转换为datetime对象
            modification_datetime = datetime.fromtimestamp(modification_time)
            # 检查文件是否为2024年的文件
            if modification_datetime.year == 2024:
                # 写入文件名到CSV文件中
                csv_writer.writerow([1, file_name])#1为真图，0为伪造

                # 检查F:\FaceForensics++\data\face_lap文件夹中是否存在相同名称的文件夹
                face_lap_folder = os.path.join('F:\FaceForensics++\data\\face_lap', file_name[:-4])
                if os.path.exists(face_lap_folder) and os.path.isdir(face_lap_folder):
                    # 构建目标文件夹路径
                    destination_path = os.path.join(destination_folder, file_name)
                    # 复制文件夹到目标文件夹中
                    shutil.copytree(face_lap_folder, destination_path)
                    print(f"Folder '{file_name[:-4]}' copied successfully.")
                    sys.exit(0)
                else:
                    print(f"No folder named '{file_name}' found in F:\FaceForensics++\data\\face_lap.")

print("All files processed.")
