import os
import logging
import torch
import numpy as np
import torch.nn as nn
from model.DNM import Vgg16_DNM, Vgg16_net
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ignite.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Vgg16"
batch_size = 32
class_num = 2
LR = 0.001

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform=transforms.Compose([
    transforms.RandomResizedCrop(224),#Resizes all images into same dimension
    # transforms.RandomRoation(10),# Rotates the images upto Max of 10 Degrees
    transforms.RandomHorizontalFlip(p=0.4),#Performs Horizantal Flip over images

    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),# Coverts into Tensors
    transforms.Normalize(mean = mean_nums, std=std_nums) # Normalizes
    # normalize
])

transform_test = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 224
    transforms.ToTensor(),
    transforms.Normalize(mean = mean_nums, std=std_nums) # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

basePath =  os.getcwd()
train_path = os.path.join(basePath, "dataset2", "train")
test_path = os.path.join(basePath, "dataset2", "test")

train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform_test)

print(train_dataset.class_to_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_size = len(train_loader.dataset)
test_size = len(test_loader.dataset) #incorrect
train_num_batches = len(train_loader)
test_num_batches = len(test_loader)

model = Vgg16_net(num_classes=class_num)
model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# setting logging
logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename=model_name+"_lr-"+str(LR)+".log" #log日志输出的文件位置和文件名
                    ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )



def get2Svalues(y_ture, y_predict):  # 返回sen和spe
    posNum = (np.array(y_ture).sum())
    negNum = len(y_ture) - posNum
    if posNum == 0 or negNum == 0:
        print("data unbalences")
        return False
    TP = 0
    FN = 0
    for i in range(len(y_predict)):
        if y_predict[i] == 1 and y_ture[i] == 1:
            TP = TP + 1
        if y_predict[i] == 0 and y_ture[i] == 0:
            FN = FN + 1
    print("sen spe info:")
    print(TP, FN, posNum, negNum)
    return float(TP) / float(posNum), float(FN) / float(negNum)


def train(model, loss_func, optimizer):
    train_acc = Accuracy()
    correct = 0
    running_loss = 0
    preds = []
    targets = []
    model = model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        out = model(batch_x)    # 模型输出

        loss = loss_func(out, batch_y) # 计算loss
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        _, pred = torch.max(out.data, 1)
        correct += (pred == batch_y).type(torch.float).sum().item()
        train_acc.update((pred, batch_y))

        pred = out.argmax(axis=1)  # 获得预测值
        lable = batch_y.squeeze().data.cpu().numpy()  # 获得标签
        preds.extend(pred)
        targets.extend(lable)


    sen, spe = (get2Svalues(targets, preds))

        # if i%10 == 0:
        #     print("{}/{}----AccNum:{}".format(i, train_num_batches, correct))
    
    running_loss /= train_num_batches
    correct /= train_size
    

    print("Train: \nAccuracy: {:0.4f}% ".format(100 * correct))
    print("Avg loss: {:8f}, ".format(running_loss))
    # print("ignite acc: {:0.4f}%".format(100 * train_acc.compute()))
    print("sen: {:0.4f} ".format(sen))
    print("spe: {:0.4f}, ".format(spe))

    logging.info("Train:")
    logging.info("Accuracy: {:0.4f}% ".format(100 * correct))
    logging.info("Avg loss: {:8f}, ".format(running_loss))
    # logging.info("ignite acc: {:0.4f}%".format(100 * train_acc.compute()))
    logging.info("sen:{:0.4f} ".format(sen))
    logging.info("spe:{:0.4f} ".format(spe))
    
    train_acc.reset()

def test(model):
    correct = 0
    running_loss = 0
    test_acc = Accuracy()
    preds = []
    targets = []
    model = model.eval()
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        out = model(batch_x)

        loss = loss_func(out, batch_y)
        running_loss += loss.item()

        _, pred = torch.max(out.data, 1)
        correct += (pred == batch_y).type(torch.float).sum().item()
        test_acc.update((pred, batch_y))
        pred = out.argmax(axis=1)  # 获得预测值
        lable = batch_y.squeeze().data.cpu().numpy()  # 获得标签
        preds.extend(pred)
        targets.extend(lable)


    sen, spe = (get2Svalues(targets, preds))
    running_loss /= test_num_batches
    correct /= test_size
    

    print("Test Error: \nAccuracy: {:0.4f}% ".format(100 * correct))
    print("Avg loss: {:8f}, ".format(running_loss))
    # print("ignite acc: {:0.4f}%".format(100 * train_acc.compute()))
    print("sen: {:0.4f} ".format(sen))
    print("spe: {:0.4f}, ".format(spe))

    logging.info("Test:")
    logging.info("Accuracy: {:0.4f}% ".format(100 * correct))
    logging.info("Avg loss: {:8f}, ".format(running_loss))
    # logging.info("ignite acc: {:0.4f}%".format(100 * train_acc.compute()))
    logging.info("sen:{:0.4f} ".format(sen))
    logging.info("spe:{:0.4f} ".format(spe))

    test_acc.reset()
    
    

def main():
    epochs = 100

    # test_recall = Recall()
    # test_precision = Precision()

    for epoch in range(epochs):
        print(("Epoch:{}".format(epoch)))
        logging.info("Epoch:{}".format(epoch))
        train(model, loss_func, optimizer)
        test(model)
    


if __name__ == '__main__':
    main()