import subprocess

#ffmpeg_path = r"F:/ffmpeg-7.0-full_build/binffmpeg.exe"

# 定义FFmpeg命令
ffmpeg_command = [
    'ffmpeg',             # 命令名
    '-i', 'F:\ECCV\data_preprocessing\processed\FF++/test/D036_035_8.png',    # 输入文件名
    '-c:v', 'libx264',    # 视频编码器
    '-crf', '23',         # 压缩质量级别
    '-preset', 'medium',  # 编码器预设
    '-frames:v', '1',
    'F:\ECCV\data_preprocessing\processed\FF++/test(c23)/D036_035_8.mp4'  # 输出文件名
]

# 调用FFmpeg命令
subprocess.run(ffmpeg_command)
