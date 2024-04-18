import csv
import cv2
import os

# 输入文件夹路径
input_folder = 'F:\ECCV\data_preprocessing\processed\FF++\\test'

# 输出视频文件夹路径
output_video_folder = 'F:\ECCV\data_preprocessing\processed\FF++\\test_vedio'

# 压缩后视频文件夹路径
output_compressed_video_folder_c23 = 'F:\ECCV\data_preprocessing\processed\FF++\\test_vedio_c23'
output_compressed_video_folder_c40 = 'F:\ECCV\data_preprocessing\processed\FF++\\test_vedio_c40'

# 提取图片文件夹路径
output_image_folder_c23 = 'F:\ECCV\data_preprocessing\processed\FF++\\test_c23'
output_image_folder_c40 = 'F:\ECCV\data_preprocessing\processed\FF++\\test_c40'

# 每个视频中的帧数
frames_per_video = 6000

# 读取输入文件夹中的所有图片文件名
image_files = os.listdir(input_folder)

nums = (len(image_files) // 6000) + 1

# 记录视频的索引
video_index = 1

# 记录视频开始的图片文件名
video_start_frame_name = ''

frame_name = []

# 逐个读取图片并生成视频
for i, image_file in enumerate(image_files, 1):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    frame_name.append(image_file[:-4])
    
    if i + 1 == len(image_files):
        print('已经是最后一张图片了喔')
        csv_file = output_video_folder + f'/video_{video_index-1}.csv'

        # 将一维数组转换为二维数组（每个元素作为单独的行）
        data_as_2d = [[item] for item in frame_name]

        # 使用 CSV 模块打开文件并写入数据
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_as_2d)
    
    # 如果当前图片是每个视频的起始帧，则创建一个新视频
    if i % frames_per_video == 1:
        video_start_frame_name = image_file
        video_writer = cv2.VideoWriter(os.path.join(output_video_folder, f'video_{video_index}.mp4'), 
                                       cv2.VideoWriter_fourcc(*'mp4v'), 60, (image.shape[1], image.shape[0]))
        video_index += 1
        print(f'正在制作第{video_index-1}/{nums}个视频')
    
    # 写入视频帧
    video_writer.write(image)
    
    # 如果当前图片是每个视频的最后一帧，则释放视频对象并进行压缩
    if i % frames_per_video == 0 or i == len(image_files):
        video_writer.release()
        
        # 使用ffmpeg进行视频压缩，crf 23
        os.system(f'ffmpeg -y -i {os.path.join(output_video_folder, f"video_{video_index-1}.mp4")} '
                  f'-c:v libx264 -crf 23 {os.path.join(output_compressed_video_folder_c23, f"video_{video_index-1}.mp4")}')
        
        # 使用ffmpeg进行视频压缩，crf 40
        os.system(f'ffmpeg -y -i {os.path.join(output_video_folder, f"video_{video_index-1}.mp4")} '
                  f'-c:v libx264 -crf 40 {os.path.join(output_compressed_video_folder_c40, f"video_{video_index-1}.mp4")}')
        
        # 提取压缩后视频的每一帧并保存为图片
        os.system(f'ffmpeg -y -i {os.path.join(output_compressed_video_folder_c23, f"video_{video_index-1}.mp4")} '
                f'{output_image_folder_c23}/%d.png')
        
        os.system(f'ffmpeg -y -i {os.path.join(output_compressed_video_folder_c40, f"video_{video_index-1}.mp4")} '
                f'{output_image_folder_c40}/%d.png')
        
        for k in range(frames_per_video):
            # 将上面保存的每一帧都重命名
            os.rename(f'{output_image_folder_c23}/{k+1}.png', f'{output_image_folder_c23}/{frame_name[k]}.png')
            os.rename(f'{output_image_folder_c40}/{k+1}.png', f'{output_image_folder_c40}/{frame_name[k]}.png')
            if k ==  frames_per_video - 1:
                # 要保存的 CSV 文件路径
                csv_file = output_video_folder + f'/video_{video_index-1}.csv'

                # 将一维数组转换为二维数组（每个元素作为单独的行）
                data_as_2d = [[item] for item in frame_name]

                # 使用 CSV 模块打开文件并写入数据
                with open(csv_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data_as_2d)

                frame_name = []
            
        print(f'Processed video {video_index-1} from {video_start_frame_name} to {image_file}')
