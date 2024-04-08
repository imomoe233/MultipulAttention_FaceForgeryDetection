import os
import shutil
import random
import csv

def random_split_and_copy(source_folder, train_folder, test_folder, train_csv, test_csv, train_percentage=0.8):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # 计算训练集和测试集的数量
    train_size = int(len(files) * train_percentage)
    
    # 随机选择训练集
    train_files = random.sample(files, train_size)
    
    # 其余文件为测试集
    test_files = list(set(files) - set(train_files))

    # 创建训练集和测试集文件夹
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 复制文件到训练集和测试集
    for file in train_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(train_folder, file)
        shutil.copyfile(source_path, destination_path)
        # 写入训练集CSV文件
        with open(train_csv, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([0 if 'fake' in source_folder else 1, file])

    for file in test_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(test_folder, file)
        shutil.copyfile(source_path, destination_path)
        # 写入测试集CSV文件
        with open(test_csv, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([0 if 'fake' in source_folder else 1, file])

# 定义文件夹路径
fake_folder = 'F:\ECCV\data_preprocessing\processed\DFDC/fake'
real_folder = 'F:\ECCV\data_preprocessing\processed\DFDC/real'
train_folder = 'F:\ECCV\data_preprocessing\processed\DFDC/train'
test_folder = 'F:\ECCV\data_preprocessing\processed\DFDC/test'
train_csv = 'F:\ECCV\data_preprocessing\processed\DFDC/trains.labels.csv'
test_csv = 'F:\ECCV\data_preprocessing\processed\DFDC/tests.labels.csv'

# 清空之前可能存在的CSV文件
with open(train_csv, 'w', newline='') as csvfile:
    pass

with open(test_csv, 'w', newline='') as csvfile:
    pass

# 分别处理fake和real文件夹
random_split_and_copy(fake_folder, train_folder, test_folder, train_csv, test_csv)
random_split_and_copy(real_folder, train_folder, test_folder, train_csv, test_csv)

print("文件复制和CSV文件生成完成。")
