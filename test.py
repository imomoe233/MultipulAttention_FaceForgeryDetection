import random
import sys
import os
import cv2
from torch.serialization import load
import dlib
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import csv
import dataset
import argparse
import model_core
from loss import am_softmax


def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")

    parser.add_argument("--test_dir", type=str, default='F:\Celeb-DF-v2\\test',
                        help="The testdata path")
    
    parser.add_argument("--train_dir", type=str, default='F:\Celeb-DF-v2\\test',
                        help="The testdata path")
    
    parser.add_argument("--test_label", type=str, default='F:\Celeb-DF-v2\\test.labels.csv',
                        help="The traindata label path")
    
    parser.add_argument("--pre_model", type=str, default='F:\Face Forgery Detection\checkpoints\FF++(raw)/checkpoint_1.tar',
                        help="the path of pretraining model")
    
    parser.add_argument("--results_save_path", type=str, default='F:/ECCV/data_preprocessing/processed/FF++/FF++_offical2FF++_offical_results',
                        help="The real_test_data path ")
    return parser.parse_args()


if __name__ == '__main__':
    args = input_args()

    torch.cuda.set_device(args.cuda_id)
    device = torch.device("cuda:%d" % (args.cuda_id) if torch.cuda.is_available() else "cpu")

    model = model_core.Two_Stream_Net()
    model = model.cuda()

    model_state_dict = torch.load(args.pre_model, map_location='cuda:0')['state_dict']
    model.load_state_dict(model_state_dict)
    
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
        # transforms.ToTensor(),  # 转换为张量
    ])
    
    # 生成列表
    test_list = [file for file in os.listdir(args.test_dir) if file.endswith('.jpg')]
    # 打乱列表顺序
    random.shuffle(test_list)
    # 取前 100 个元素
    test_list = test_list[:300]
    
    ValData = torch.utils.data.DataLoader(
        dataset.LoadData(args, test_list, test_label_dict, mode='val', transform=transform_256),
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    

    # Initialize lists to store true labels and predicted probabilities
    true_labels = []
    predicted_probs = []

    val_bar = tqdm(ValData)
    
    val_correct = 0
    val_total = 0
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
            for i in range(len(val_predict.eq(val_label.data))):
                if val_predict.eq(val_label.data)[i] == False:
                    print(encoder_save_path[i])
                    #print(val_label.data[i])
            val_total = val_total + val_label.size(0)
            val_ac = 100.0 * val_correct / val_total

            desc = 'Validation  :AC = %.4f' % (val_ac)
            
            val_bar.set_description(desc)
            val_bar.update()
        # 计算概率
        probabilities = torch.softmax(val_output, dim=1)
        # 获取预测概率
        #print(probabilities)
        predicted_prob = probabilities[:, 1].cpu().numpy()  # 取第二类的概率作为正类的预测概率
        
        # 获取真实标签
        true_label = val_label.cpu().numpy()
        
        # 将真实标签和预测概率添加到列表中
        #true_labels.extend(true_label)
        true_labels.extend(val_label.cpu().numpy().tolist())
        predicted_probs.extend(predicted_prob)
        
    # Calculate TPR and FPR
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs, pos_label=None)

    # print(fpr, tpr, thresholds)
    # print(true_labels, predicted_probs)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % (roc_auc*100))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save ROC curve plot
    if not os.path.exists(args.results_save_path):
        os.makedirs(args.results_save_path)
        print(f"Folder '{args.results_save_path}' created.")
    plt.savefig(args.results_save_path + '/test_roc_curve.png')
    
    # plt.show()
    
    # Save AUC value
    print("AUC:", (roc_auc*100))