import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))
import wandb
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import utils as vutils
from torchvision import transforms
from torch.utils.data import Dataset
import csv
import dataset
import argparse
import model_core
from loss import am_softmax
from PIL import Image



def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")

    parser.add_argument("--test_label", type=str, default='../dataset/Celeb-DF-v2/test.labels.csv',
                        help="The traindata label path")
    
    parser.add_argument("--test_dir", type=str, default='../dataset/Celeb-DF-v2/test-align',
                        help="The real_test_data path ")
    
    parser.add_argument("--load_model", type=bool, default=False,
                        help="Whether load pretraining model")

    parser.add_argument("--pre_model", type=str, default='F:\Face Forge Detection\checkpoints\Celeb-DF-v2\k8v96v71 celeb-align-11-shuffle True-adamw-lr0.0001-attention_ca_depth_drop0.2/checkpoint_6.tar',
                        help="the path of pretraining model")

    return parser.parse_args()


if __name__ == '__main__':
    
    args = input_args()
    
    torch.cuda.set_device(args.cuda_id)
    device = torch.device("cuda:%d" % (args.cuda_id) if torch.cuda.is_available() else "cpu")
    
    csvFile = open(args.test_label, "r")
    reader = csv.reader(csvFile)
    test_label_dict = dict()
    
    for item in reader:
        # key: filename
        key = item[-1]
        # value: the label (0 or 1) of file 
        value = item[0]
        if value != 'l':
            value = int(value)
            test_label_dict.update({key: value})

    transform_256 = transforms.Compose([
            transforms.Resize((256, 256), antialias=None),  # 上采样为 256x256
            #transforms.ToTensor(),  # 转换为张量
        ])
    

    test_list = [file for file in os.listdir(args.test_dir) if file.endswith('.jpg')]

    ValData = torch.utils.data.DataLoader(dataset.LoadData(args, test_list, test_label_dict, mode='val', transform=transform_256),
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=8,
                                            drop_last=True)
    

    model = model_core.Two_Stream_Net()
    model = model.cuda()
    if args.load_model:
        model_state_dict = torch.load(args.pre_model, map_location='cuda:0')['state_dict']
        model.load_state_dict(model_state_dict)

    epoch = 0 
    
    while epoch < 1000:
        count = 0
        total_loss = 0
        correct = 0
        total = 0

        val_correct = 0
        val_total = 0
        val_bar = tqdm(ValData)
        for batch_idx, (val_input, val_label, img_lap, encoder_save_path) in enumerate(val_bar):
            model.eval()

            val_input = val_input.to(device)
            val_label = val_label.to(device)
            val_label = val_label.squeeze()
            img_lap = img_lap.to(device)

            with torch.no_grad():
                val_output = model(val_input, img_lap, encoder_save_path)
            _, val_predict = torch.max(val_output.data, 1)
            val_correct += val_predict.eq(val_label.data).cpu().sum()
            
            print(val_predict.eq(val_label.data))
            for i in range(len(val_predict.eq(val_label.data))):
                if val_predict.eq(val_label.data)[i] == False:
                    print(encoder_save_path[i])
            
            val_total = val_total + val_label.size(0)
            val_ac = 100.0 * val_correct / val_total

            desc = 'Validation  : Epoch %d, AC = %.4f' % (epoch, val_ac)
            
            val_bar.set_description(desc)
            val_bar.update()
        
        epoch = epoch + 1

        
