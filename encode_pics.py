import os
import sys
from PIL import Image
from models.inversion_coach import InversionCoach

coach = InversionCoach()
print('load encoder.pt successfully pretrained')

# 指定文件夹路径
folder_path = "E:/code_xwd/dataset/Face_Forgery_Detection/train"

# 获取文件夹中所有图片文件的列表
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# 遍历读取图片
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image)
    
    x = coach.run(image)
    print(x)
    print(type(x))
    sys.exit()
