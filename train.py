import sys
import os
# if __name__ == '__main__':
#     sys.path.append(os.path.dirname(sys.path[0]))
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
from torch.utils.data import WeightedRandomSampler


def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")

    parser.add_argument("--train_label", type=str, default='../Celeb-DF-v2/train.labels.csv',
                        help="The traindata label path")

    parser.add_argument("--test_label", type=str, default='../Celeb-DF-v2/test.labels.csv',
                        help="The traindata label path")

    parser.add_argument("--train_dir", type=str, default='../Celeb-DF-v2/train',
                        help="The real_train_data path ")

    parser.add_argument("--test_dir", type=str, default='../Celeb-DF-v2/test',
                        help="The real_test_data path ")

    parser.add_argument("--load_model", type=bool, default=False,
                        help="Whether load pretraining model")

    parser.add_argument("--pre_model", type=str, default='',
                        help="the path of pretraining model")

    parser.add_argument("--save_model", type=str, default='checkpoints/Celeb-DF-v2/',
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

    train_list = [file for file in os.listdir(args.train_dir) if file.endswith('.jpg')]
    test_list = [file for file in os.listdir(args.test_dir) if file.endswith('.jpg')]

    ######################################################################################################
    #
    #   添加sampler，并在读取数据时传入，以平衡数据
    #
    # 计算每个类别的样本数
    label_counts = {0: 135016, 1: 21210}
    for label in train_label_dict.values():
        label_counts[label] += 1

    # 计算类别权重（真的类别数量是假的四倍）
    total_count = sum(label_counts.values())
    class_weights = {cls: total_count / count for cls, count in label_counts.items()}

    # 计算每个样本的权重
    sample_weights = [class_weights[label] for label in train_label_dict.values()]

    # 创建WeightedRandomSampler对象
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    ######################################################################################################

    TrainData = torch.utils.data.DataLoader(
        dataset.LoadData(args, train_list, train_label_dict, mode='train', transform=transform_256),
        batch_size=16,
        # shuffle=True,
        sampler=sampler,
        num_workers=8,
        drop_last=True)
    ValData = torch.utils.data.DataLoader(
        dataset.LoadData(args, test_list, test_label_dict, mode='val', transform=transform_256),
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=True)

    model = model_core.Two_Stream_Net()
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
