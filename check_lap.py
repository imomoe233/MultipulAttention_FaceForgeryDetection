import os

# 获取lap文件夹中的所有文件夹名字
train_lap_folder = "F:\ECCV\data_preprocessing\processed\FF++\\train_lap"  # 你的lap文件夹路径
train_lap_folders = [f for f in os.listdir(train_lap_folder) if os.path.isdir(os.path.join(train_lap_folder, f))]

test_lap_folder = "F:\ECCV\data_preprocessing\processed\FF++\\test_lap"  # 你的lap文件夹路径
test_lap_folders = [f for f in os.listdir(test_lap_folder) if os.path.isdir(os.path.join(test_lap_folder, f))]

all_lap = train_lap_folders + test_lap_folders


# 在train文件夹中查找lap文件夹中的每个文件夹
train_folder = "F:\ECCV\data_preprocessing\processed\FF++\\test_offical"  # 你的train文件夹路径
files = [f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]

num = 0

for file in files:
    if file[:-4] in all_lap:
        #print(f"{folder} 存在于 lap 文件夹中")
        continue
    else:
        num += 1
        os.remove(train_folder + '/' + file)
        print(f"{file} 不存在于 lap 文件夹中,删除{train_folder + '/' + file}")
    
print("sum = ", num)
