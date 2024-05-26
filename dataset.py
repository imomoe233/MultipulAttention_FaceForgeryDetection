import os
from sys import argv
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import dlib
import torch
import cv2


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


class LoadData(Dataset):
    def __init__(self, arg, img, label_dict, mode='train', transform=None):
        #self.img = img
        self.img = sorted(img)
        self.label_dict = label_dict
        self.mode = mode
        self.args = arg
        self.transform = transform

    def __getitem__(self, item):
        input_img = self.img[item]

        #print(input_img)
        
        t_list = [transforms.ToTensor()]
        composed_transform = transforms.Compose(t_list)
        try:
            if self.mode == 'train':
                face_detect = dlib.get_frontal_face_detector()
                img = cv2.imread(self.args.train_dir + '/' + input_img)
                
                if self.args.train_dir == 'F:/ECCV/data_preprocessing/processed/FF++/train_offical':
                    img_lap = cv2.imread(self.args.train_dir[:-8] + '_lap/' + input_img[:-4] + '/recon_normal_lap.png')
                else:
                    img_lap = cv2.imread(self.args.train_dir + '_lap/' + input_img[:-4] + '/recon_normal_lap.png')     
                    
                #img_lap = cv2.resize(img_lap, (256, 256))
                img_lap = composed_transform(img_lap)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detect(gray, 1)
            
                if len(faces) != 0:
                    face = faces[0]
                    x, y, size = get_boundingbox(face, 256, 256)
                    cropped_face = img[y:y+size, x:x+size]
                    #cropped_face = cv2.resize(cropped_face, (256, 256))
                    cropped_face = composed_transform(cropped_face)

                    label = self.label_dict[input_img]
                    label = torch.LongTensor([label])
                else:
                    #cropped_face = cv2.resize(img, (256, 256))
                    #cropped_face = composed_transform(cropped_face)
                    #从↑修改为↓
                    #cropped_face = composed_transform(img)
                    #print(f'========================== {input_img} no face =============================================')
                    #os.remove(self.args.train_dir + '\\' + input_img)
                    #print(f'删除{self.args.train_dir}\{input_img}')
                    
                    # 直接下一张图
                    #print(f"Error processing image  {input_img} no face ")
                    return self.__getitem__(item + 1) 
                    
                    label = self.label_dict[input_img]
                    label = torch.LongTensor([label])
                if self.args.train_dir == 'F:/ECCV/data_preprocessing/processed/FF++/train_offical':
                    encoder_save_path = self.args.train_dir[:-8] + '_lap/' + input_img[:-4] + '/encoder.pt'
                    
                else:
                    encoder_save_path = self.args.train_dir + '_lap/' + input_img[:-4] + '/encoder.pt'

                encoder_save = torch.load(encoder_save_path)
       
            if self.mode == 'val':
                face_detect = dlib.get_frontal_face_detector()
                img = cv2.imread(self.args.test_dir + '/' + input_img)
                
                
                if self.args.test_dir == 'F:/ECCV/data_preprocessing/processed/FF++/test_offical':
                    # 为了方便，将test_official下的_lap全放在train_lap中
                    img_lap = cv2.imread(self.args.train_dir[:-8] + '_lap/' + input_img[:-4] + '/recon_normal_lap.png')
                else:
                    # 为了方便，将test_official下的_lap全放在train_lap中
                    img_lap = cv2.imread(self.args.train_dir + '_lap/' + input_img[:-4] + '/recon_normal_lap.png')
                
                
                #img_lap = cv2.resize(img_lap, (256, 256))
                img_lap = composed_transform(img_lap)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detect(gray, 1)
                if len(faces) != 0:
                    face = faces[0]
                    x, y, size = get_boundingbox(face, 256, 256)
                    cropped_face = img[y:y+size, x:x+size]
                    #cropped_face = cv2.resize(cropped_face, (256, 256))
                    cropped_face = composed_transform(cropped_face)

                    label = self.label_dict[input_img]
                    label = torch.LongTensor([label])
                else:
                    #cropped_face = cv2.resize(img, (256, 256))
                    #cropped_face = composed_transform(cropped_face)
                    #从↑修改为↓
                    #cropped_face = composed_transform(img)
                    #print(f'========================== {input_img} no face =============================================')
                    #os.remove(self.args.train_dir + '\\' + input_img)
                    #print(f'删除{self.args.train_dir}\{input_img}')
                    
                    # 直接下一张图
                    #print(f"Error processing image  {input_img} no face ")
                    return self.__getitem__(item + 1)
                    label = self.label_dict[input_img]
                    label = torch.LongTensor([label])
                if self.args.test_dir == 'F:/ECCV/data_preprocessing/processed/FF++/test_offical':
                    # 为了方便，将test_official下的_lap全放在train_lap中
                    encoder_save_path = self.args.train_dir[:-8] + '_lap/' + input_img[:-4] + '/encoder.pt'
                else:
                    # 为了方便，将test_official下的_lap全放在train_lap中
                    encoder_save_path = self.args.train_dir + '_lap/' + input_img[:-4] + '/encoder.pt'

                encoder_save = torch.load(encoder_save_path)
                
            if self.transform is not None:
                cropped_face = self.transform(cropped_face)
                img_lap = self.transform(img_lap)
                
        except Exception as e:
            print(f"Error processing image {input_img}: {str(e)}")
            return self.__getitem__(item + 1) 
        
        return cropped_face, label, img_lap, encoder_save_path, encoder_save

    def __len__(self):
        return len(self.img)


