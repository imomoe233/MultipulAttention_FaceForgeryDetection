import os
import csv

folder_path = 'F:\ECCV\data_preprocessing\processed\FF++\\test_offical'
csv_file = 'F:\ECCV\data_preprocessing\processed\FF++\\test_offical.labels.csv'

# 获取文件夹中的文件名
file_names = os.listdir(folder_path)

# 将文件名保存到CSV文件
with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for index, file_name in enumerate(file_names, start=1):
        if file_name[0] == 'O':
            csv_writer.writerow([1, file_name])
        else:
            csv_writer.writerow([0, file_name])
