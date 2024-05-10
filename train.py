
import argparse
import csv
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import model_core
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# if __name__ == '__main__':
#     sys.path.append(os.path.dirname(sys.path[0]))
import wandb
from loss import am_softmax
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

import dataset


def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")

    parser.add_argument("--train_label", type=str, default='F:/ECCV/data_preprocessing/processed/FF++/train_offical.labels.csv',
                        help="The traindata label path")

    parser.add_argument("--test_label", type=str, default='F:/ECCV/data_preprocessing/processed/FF++/test_offical.labels.csv',
                        help="The traindata label path")

    parser.add_argument("--train_dir", type=str, default='F:/ECCV/data_preprocessing/processed/FF++/train_offical',
                        help="The real_train_data path ")

    parser.add_argument("--test_dir", type=str, default='F:/ECCV/data_preprocessing/processed/FF++/test_offical',
                        help="The real_test_data path ")
    
    parser.add_argument("--results_save_path", type=str, default='F:/ECCV/data_preprocessing/processed/FF++/FF++_offical2FF++_offical_results',
                        help="The real_test_data path ")

    parser.add_argument("--load_model", type=bool, default=False,
                        help="Whether load pretraining model")

    parser.add_argument("--pre_model", type=str, default='F:\Face Forgery Detection\checkpoints\FF++/checkpoint_2.tar',
                        help="the path of pretraining model")

    parser.add_argument("--save_model", type=str, default='F:/Face Forgery Detection/checkpoints/FF++/',
                        help="the path of saving model")

    return parser.parse_args()


if __name__ == '__main__':
    # wandb = None
    if wandb is not None:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Face Forge Detection",

            # track hyperparameters and run metadata
            config={
            }
        )

    args = input_args()

    torch.cuda.set_device(args.cuda_id)
    device = torch.device("cuda:%d" % (args.cuda_id) if torch.cuda.is_available() else "cpu")

    csvFile = open(args.train_label, "r")
    reader = csv.reader(csvFile)
    train_label_dict = dict()

    for item in reader:
        # print(item)
        # key: filename
        key = item[-1]
        # value: the label (0 or 1) of file 
        value = item[0]
        if value != 'l':
            value = int(value)
            train_label_dict.update({key: value})

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

    train_list = [file for file in os.listdir(args.train_dir) if file.endswith('.png')]
    # 打乱列表顺序
    random.shuffle(train_list)
    # 取前 100 个元素
    #train_list = train_list[:len(train_list)//10]
    
    test_list = [file for file in os.listdir(args.test_dir) if file.endswith('.png')]
    # 打乱列表顺序
    random.shuffle(test_list)
    # 取前 100 个元素
    #test_list = test_list[:len(train_list)//2]
    
    TrainData = torch.utils.data.DataLoader(
        dataset.LoadData(args, train_list, train_label_dict, mode='train', transform=transform_256),
        batch_size=32,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    ValData = torch.utils.data.DataLoader(
        dataset.LoadData(args, test_list, test_label_dict, mode='val', transform=transform_256),
        batch_size=32,
        shuffle=True,
        num_workers=8,
        drop_last=True)

    model = model_core.Two_Stream_Net()
    
    # 替换模型
    
    model = model.cuda()
    if args.load_model:
        model_state_dict = torch.load(args.pre_model, map_location='cuda:0')['state_dict']
        model.load_state_dict(model_state_dict)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=4e-3)

    epoch = 0

    while epoch < 1000:
        count = 0
        total_loss = 0
        correct = 0
        train_bar = tqdm(TrainData)
        total = 0

        for batch_idx, (input_img, img_label, img_lap, encoder_save_path) in enumerate(train_bar):
            count = count + 1

            model.train()
            input_img = input_img.to(device)
            img_label = img_label.to(device)
            img_lap = img_lap.to(device)

            outputs = model(input_img, img_lap, encoder_save_path)
            optimizer.zero_grad()

            amloss = am_softmax.AMSoftmaxLoss()
            img_label = img_label.squeeze()
            loss = amloss(outputs, img_label)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss
            avg_loss = total_loss / count
            _, predict = torch.max(outputs.data, 1)
            # print(outputs)
            correct += predict.eq(img_label.data).cpu().sum()
            for i in range(len(predict.eq(img_label.data))):
                if predict.eq(img_label.data)[i] == False:
                    print("识别错误的人脸: " + encoder_save_path[i])
            total = total + img_label.size(0)
            correct_per = 100.0 * correct / total

            desc = 'Training : Epoch %d, AvgLoss = %.4f, AC = %.4f' % (epoch, avg_loss, correct_per)

            # correct_per是包含了整个batch之前出现过后的均值，所以作为结果，只需要看他在这个batch的最后一次的值即可
            wandb.log({"Epoch": epoch, "Train_AvgLoss": avg_loss, "Train_AC": correct_per})
            train_bar.set_description(desc)
            train_bar.update()

        # Initialize lists to store true labels and predicted probabilities
        true_labels = []
        predicted_probs = []

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
            for i in range(len(val_predict.eq(val_label.data))):
                if val_predict.eq(val_label.data)[i] == False:
                    print(encoder_save_path[i])
                    print(val_label.data[i])
            val_total = val_total + val_label.size(0)
            val_ac = 100.0 * val_correct / val_total

            desc = 'Validation  : Epoch %d, AC = %.4f' % (epoch, val_ac)

            # val_ac是包含了整个batch之前出现过后的均值，所以作为结果，只需要看他在这个batch的最后一次的值即可
            wandb.log({"Epoch": epoch, "Test_AC": val_ac})
            val_bar.set_description(desc)
            val_bar.update()
            
            # 计算概率
            probabilities = torch.softmax(val_output, dim=1)
            # 获取预测概率
            predicted_prob = probabilities[:, 0].cpu().numpy()  # 取第二类的概率作为正类的预测概率
            
            # 获取真实标签
            true_label = val_label.cpu().numpy()
            
            # 将真实标签和预测概率添加到列表中
            true_labels.extend(true_label)
            predicted_probs.extend(predicted_prob)

        savename = args.save_model + '/checkpoint' + '_' + str(epoch) + '.tar'
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
            print(f"Folder '{args.save_model}' created.")
        else:
            print(f"Folder '{args.save_model}' already exists.")
        try:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, savename)
        except Exception as e:
            print(f"Error during model save: {e}")
        epoch = epoch + 1

        # Calculate TPR and FPR
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs, pos_label=1)

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
        plt.savefig(args.results_save_path + '/roc_curve' + str(epoch) + '.png')
        
        # plt.show()

        # Save AUC value
        print("AUC:", (roc_auc*100))
        
        wandb.log({"Epoch": epoch, "AUC": (roc_auc*100)})

        
